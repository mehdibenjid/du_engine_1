"""
Step 7 — UMAP + HDBSCAN Clustering.

Clusters incoming query embeddings, computes medoids and diverse representatives
per cluster, and writes cluster and representative parquet files.
"""
import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import hdbscan
import numpy as np
import pandas as pd
import umap

from config import Config, load_config
from helpers.clustering import (
    compute_medoid_indices,
    load_and_normalize_vectors,
    mmr_select,
    top_metadata_distributions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_clustering(
    cfg: Config,
    incoming_embeddings_path: str,
    topk_parquet_path: Optional[str] = None,
    artifacts_dir: Optional[str] = None,
    use_umap: Optional[bool] = None,
    umap_n_components: Optional[int] = None,
    umap_n_neighbors: Optional[int] = None,
    umap_min_dist: Optional[float] = None,
    hdbscan_min_cluster_size: Optional[int] = None,
    hdbscan_min_samples: Optional[int] = None,
    random_state: Optional[int] = None,
    rep_closest_n: Optional[int] = None,
    rep_diverse_n: Optional[int] = None,
    rep_mmr_lambda: Optional[float] = None,
    max_text_chars_for_llm: Optional[int] = None,
    incoming_id_col: str = "incoming_reference",
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Clusters incoming embeddings with UMAP + HDBSCAN and selects per-cluster representatives.

    Args:
        cfg:                       Global configuration object.
        incoming_embeddings_path:  Path to incoming_embeddings.parquet (must contain 'qvec' column).
        topk_parquet_path:         Optional path to topk results parquet for metadata enrichment.
        artifacts_dir:             Output directory. Defaults to cfg.paths.artifacts_dir.
        use_umap … random_state:   Override clustering parameters (fall back to config values if None).
        rep_closest_n … rep_mmr_lambda: Override representative selection parameters.
        max_text_chars_for_llm:    Truncate rep texts to this many chars.
        incoming_id_col:           ID column name in the embeddings parquet.

    Returns:
        cluster_representatives DataFrame.

    Side effects:
        Writes:
          - <artifacts_dir>/datasets/incoming_clusters.parquet
          - <artifacts_dir>/datasets/cluster_representatives.parquet
    """
    if artifacts_dir is None:
        artifacts_dir = cfg.paths.artifacts_dir

    cl = cfg.clustering
    use_umap               = use_umap               if use_umap               is not None else cl.use_umap
    umap_n_components      = umap_n_components      if umap_n_components      is not None else cl.umap_n_components
    umap_n_neighbors       = umap_n_neighbors       if umap_n_neighbors       is not None else cl.umap_n_neighbors
    umap_min_dist          = umap_min_dist          if umap_min_dist          is not None else cl.umap_min_dist
    hdbscan_min_cluster_size = hdbscan_min_cluster_size if hdbscan_min_cluster_size is not None else cl.hdbscan_min_cluster_size
    hdbscan_min_samples    = hdbscan_min_samples    if hdbscan_min_samples    is not None else cl.hdbscan_min_samples
    random_state           = random_state           if random_state           is not None else cl.random_state
    rep_closest_n          = rep_closest_n          if rep_closest_n          is not None else cl.rep_closest_n
    rep_diverse_n          = rep_diverse_n          if rep_diverse_n          is not None else cl.rep_diverse_n
    rep_mmr_lambda         = rep_mmr_lambda         if rep_mmr_lambda         is not None else cl.rep_mmr_lambda
    max_text_chars_for_llm = max_text_chars_for_llm if max_text_chars_for_llm is not None else cl.max_text_chars_for_llm

    ds_dir = os.path.join(artifacts_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    out_clusters = os.path.join(ds_dir, "incoming_clusters.parquet")
    out_reps = os.path.join(ds_dir, "cluster_representatives.parquet")

    abs_emb_path = os.path.abspath(incoming_embeddings_path)
    if not os.path.exists(abs_emb_path):
        raise FileNotFoundError(
            f"Embeddings parquet not found: {abs_emb_path}. "
            "Ensure step 6 ran with --save_embeddings."
        )
    logger.info("Step 7 - loading embeddings: %s", abs_emb_path)
    emb_df = pd.read_parquet(abs_emb_path)

    # Validate the ID column before any access.
    if incoming_id_col not in emb_df.columns:
        raise ValueError(
            f"Column '{incoming_id_col}' not found in embeddings parquet. "
            f"Available columns: {list(emb_df.columns)}"
        )

    emb_df[incoming_id_col] = emb_df[incoming_id_col].astype(str).str.strip()
    emb_df = emb_df.drop_duplicates(subset=[incoming_id_col]).reset_index(drop=True)
    # emb_df now has a clean RangeIndex(0, N). All positional access below uses
    # .iloc to make this dependency explicit and safe against future re-indexing.

    # Metadata enrichment from TopK parquet
    if topk_parquet_path and os.path.exists(topk_parquet_path):
        topk_df = pd.read_parquet(topk_parquet_path)
        if "incoming_reference" in topk_df.columns:
            meta_cols_raw = ["incoming_reference", "incoming_intent", "incoming_category", "incoming_system", "incoming_reason", "incoming_action", "incoming_rubric"]
            meta_cols_present = [c for c in meta_cols_raw if c in topk_df.columns]
            topk_meta = topk_df[meta_cols_present].drop_duplicates(subset=["incoming_reference"]).copy()
            topk_meta["incoming_reference"] = topk_meta["incoming_reference"].astype(str).str.strip()
            topk_meta = topk_meta.rename(columns={c: c.replace("incoming_", "") for c in topk_meta.columns if c != "incoming_reference"})
            emb_df = emb_df.merge(topk_meta, how="left", left_on=incoming_id_col, right_on="incoming_reference", suffixes=("", "_topk"))
            # Re-establish clean RangeIndex after merge.
            emb_df = emb_df.reset_index(drop=True)
            logger.info("Enriched with TopK metadata.")

    X = load_and_normalize_vectors(emb_df["qvec"])
    n = X.shape[0]
    logger.info("Vectors loaded: %d | dims=%d", n, X.shape[1])

    # UMAP
    if use_umap and n >= 5:
        safe_components = min(umap_n_components, max(2, n - 2))
        safe_neighbors = min(umap_n_neighbors, max(2, n - 1))
        umap_kwargs = dict(
            n_neighbors=safe_neighbors, n_components=safe_components,
            min_dist=umap_min_dist, metric="cosine",
        )
        if random_state is not None:
            umap_kwargs["random_state"] = random_state
        if n_jobs is not None:
            umap_kwargs["n_jobs"] = n_jobs
        reducer = umap.UMAP(**umap_kwargs)
        Z = np.asarray(reducer.fit_transform(X), dtype=np.float32)
        umap2_kwargs = {**umap_kwargs, "n_components": 2}
        reducer2 = umap.UMAP(**umap2_kwargs)
        Z2 = np.asarray(reducer2.fit_transform(X), dtype=np.float32)
        logger.info("UMAP done: Z=%s Z2=%s", Z.shape, Z2.shape)
    else:
        Z = X
        Z2 = np.zeros((n, 2), dtype=np.float32)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples, metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(Z)
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("Clustering: %d clusters | %d noise (%d total)", n_clusters, n_noise, n)

    medoid_idx = compute_medoid_indices(X, labels)
    dist_to_medoid = np.full((n,), np.nan, dtype=np.float32)
    is_medoid = np.zeros((n,), dtype=bool)
    for c, midx in medoid_idx.items():
        idx = np.where(labels == c)[0]
        mvec = X[midx]
        sims = (X[idx] @ mvec).astype(np.float32)
        dist_to_medoid[idx] = (1.0 - sims).astype(np.float32)
        is_medoid[midx] = True

    clusters_df = pd.DataFrame({
        incoming_id_col: emb_df[incoming_id_col].tolist(),
        "cluster_id": labels.astype(int),
        "umap_x": Z2[:, 0],
        "umap_y": Z2[:, 1],
        "dist_to_medoid": dist_to_medoid,
        "is_medoid": is_medoid,
    })
    clusters_df.to_parquet(out_clusters, index=False)
    logger.info("Clusters saved: %s", out_clusters)

    # Representatives
    target_meta_fields = ["intent", "category", "system", "reason", "action", "rubric"]
    meta_fields = [col for col in emb_df.columns if str(col).lower() in target_meta_fields]
    text_col = "dense_text" if "dense_text" in emb_df.columns else None

    rep_rows = []
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        midx = medoid_idx.get(c, int(idx[0]))
        mvec = X[midx]
        sims = (X[idx] @ mvec).astype(np.float32)
        order = idx[np.argsort(-sims)]
        closest = [int(i) for i in order[:max(1, rep_closest_n)]]

        subX = X[idx]
        rel_full = (subX @ mvec).astype(np.float32)
        # Find midx's local position within this cluster's idx array.
        local_positions = np.where(idx == midx)[0]
        if local_positions.size == 0:
            # midx is not in this cluster — should never happen by construction.
            logger.warning("Medoid index %d not found in cluster %d members; using position 0.", midx, c)
            local_medoid = 0
        else:
            local_medoid = int(local_positions[0])
        mmr_k = min(idx.size, max(1, rep_diverse_n + 1))
        mmr_sel_local = mmr_select(subX, rel_full, k=mmr_k, lambda_=rep_mmr_lambda, already_selected=[local_medoid])
        diverse = [int(idx[j]) for j in mmr_sel_local if int(idx[j]) not in closest]
        reps = (closest + diverse)[:max(1, rep_closest_n + rep_diverse_n)]

        # Use .iloc for positional access — emb_df has a guaranteed RangeIndex here.
        medoid_ref = str(emb_df.iloc[midx][incoming_id_col])
        rep_refs = [str(emb_df.iloc[i][incoming_id_col]) for i in reps]
        texts = []
        if text_col:
            for i in reps:
                t = str(emb_df.iloc[i][text_col])
                if max_text_chars_for_llm and len(t) > max_text_chars_for_llm:
                    t = t[:max_text_chars_for_llm] + "..."
                texts.append(t)

        meta_dist = top_metadata_distributions(emb_df.iloc[idx], meta_fields, top_n=5)
        coherence = float(np.mean(sims)) if sims.size else float("nan")

        rep_rows.append({
            "cluster_id": int(c), "cluster_size": int(idx.size),
            "medoid_reference": medoid_ref, "rep_references": ";".join(rep_refs),
            "rep_texts": json.dumps(texts, ensure_ascii=False),
            "meta_distributions_json": json.dumps(meta_dist, ensure_ascii=False),
            "coherence_mean_cosine_to_medoid": coherence,
        })

    reps_df = pd.DataFrame(rep_rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]).reset_index(drop=True)
    reps_df.to_parquet(out_reps, index=False)
    logger.info("Representatives saved: %s | %d clusters", out_reps, len(reps_df))

    return reps_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 7 - Cluster incoming query embeddings.")
    parser.add_argument("--incoming_embeddings_parquet", required=True)
    parser.add_argument("--incoming_topk_parquet", type=str, default="")
    parser.add_argument("--artifacts_dir", type=str, default=None)
    parser.add_argument("--use_umap", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable UMAP (--use_umap / --no-use_umap). Defaults to config value.")
    parser.add_argument("--umap_n_components", type=int, default=None)
    parser.add_argument("--umap_n_neighbors", type=int, default=None)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=None)
    parser.add_argument("--hdbscan_min_samples", type=int, default=None)
    parser.add_argument("--rep_closest_n", type=int, default=None)
    parser.add_argument("--rep_diverse_n", type=int, default=None)
    parser.add_argument("--rep_mmr_lambda", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    run_clustering(
        cfg, incoming_embeddings_path=args.incoming_embeddings_parquet,
        topk_parquet_path=args.incoming_topk_parquet or None,
        artifacts_dir=args.artifacts_dir or cfg.paths.artifacts_dir,
        use_umap=args.use_umap,
        umap_n_components=args.umap_n_components, umap_n_neighbors=args.umap_n_neighbors,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size, hdbscan_min_samples=args.hdbscan_min_samples,
        rep_closest_n=args.rep_closest_n, rep_diverse_n=args.rep_diverse_n, rep_mmr_lambda=args.rep_mmr_lambda,
    )
