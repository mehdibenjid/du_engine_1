"""
Step 10 — Medoid-based Classification.

Compares each incoming query vector against the corpus cluster medoids
saved during TRAIN/RETRAIN (step 9) and assigns an existing class or
marks the query as a new discovery.
"""
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from config import Config
from helpers.clustering import load_and_normalize_vectors

logger = logging.getLogger(__name__)


def _load_medoids(medoids_path: str):
    """
    Loads cluster_medoids.json and returns (cluster_ids, medoid_matrix, medoid_meta).

    Returns:
        cluster_ids:   List[int] — ordered cluster IDs.
        medoid_matrix: np.ndarray of shape (n_clusters, dims), L2-normalized.
        medoid_meta:   Dict[int, dict] — per-cluster metadata (reference, cluster_size).
    """
    with open(medoids_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    medoids_raw = data.get("medoids", {})
    if not medoids_raw:
        return [], np.empty((0, 0), dtype=np.float32), {}

    cluster_ids = sorted(int(k) for k in medoids_raw.keys())
    vecs = [np.asarray(medoids_raw[str(cid)]["vector"], dtype=np.float32) for cid in cluster_ids]
    medoid_matrix = np.stack(vecs)

    # L2-normalize for cosine similarity via dot product
    norms = np.linalg.norm(medoid_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    medoid_matrix = medoid_matrix / norms

    medoid_meta = {
        cid: {
            "reference": medoids_raw[str(cid)].get("reference", ""),
            "cluster_size": medoids_raw[str(cid)].get("cluster_size", 0),
        }
        for cid in cluster_ids
    }

    return cluster_ids, medoid_matrix, medoid_meta


def run_classify(
    cfg: Config,
    incoming_embeddings_path: str,
    artifacts_dir: str,
    threshold: Optional[float] = None,
    incoming_id_col: str = "incoming_reference",
) -> pd.DataFrame:
    """
    Classifies incoming queries against corpus cluster medoids.

    For each incoming query vector, computes cosine similarity to every
    medoid. If the maximum similarity exceeds the threshold, the query
    is assigned to that existing class. Otherwise it is marked as a
    new discovery.

    Args:
        cfg:                       Global config.
        incoming_embeddings_path:  Path to incoming_embeddings.parquet (from step 6).
        artifacts_dir:             Active version directory containing metadata/.
        threshold:                 Cosine similarity threshold. Defaults to
                                   cfg.clustering.classification_threshold.
        incoming_id_col:           Column name for query ID.

    Returns:
        DataFrame with columns: incoming_reference, existing_class_id,
        best_medoid_similarity, best_medoid_reference, is_new_discovery.

    Side effects:
        Writes: <artifacts_dir>/datasets/incoming_classification.parquet
    """
    if threshold is None:
        threshold = cfg.clustering.classification_threshold

    # --- Load medoids ---
    medoids_path = os.path.join(artifacts_dir, "metadata", "cluster_medoids.json")
    if not os.path.exists(medoids_path):
        raise FileNotFoundError(
            f"Medoids file not found: {medoids_path}. "
            "Run 'train' or 'retrain' first to generate cluster medoids."
        )

    cluster_ids, medoid_matrix, medoid_meta = _load_medoids(medoids_path)
    logger.info("Step 10 — loaded %d medoids from: %s", len(cluster_ids), medoids_path)

    if len(cluster_ids) == 0:
        logger.warning("No medoids available — all rows will be marked as new discoveries.")

    # --- Load incoming embeddings ---
    abs_emb_path = os.path.abspath(incoming_embeddings_path)
    if not os.path.exists(abs_emb_path):
        raise FileNotFoundError(
            f"Incoming embeddings not found: {abs_emb_path}. "
            "Ensure step 6 ran with save_embeddings=True."
        )

    emb_df = pd.read_parquet(abs_emb_path)
    if incoming_id_col not in emb_df.columns:
        raise ValueError(
            f"Column '{incoming_id_col}' not found in embeddings parquet. "
            f"Available columns: {list(emb_df.columns)}"
        )

    emb_df[incoming_id_col] = emb_df[incoming_id_col].astype(str).str.strip()
    emb_df = emb_df.drop_duplicates(subset=[incoming_id_col]).reset_index(drop=True)
    n = len(emb_df)
    logger.info("Incoming queries: %d", n)

    # --- Vectorized classification ---
    if len(cluster_ids) == 0 or n == 0:
        # No medoids or no queries — everything is new
        out_df = pd.DataFrame({
            incoming_id_col: emb_df[incoming_id_col].tolist() if n > 0 else [],
            "existing_class_id": [-1] * n,
            "best_medoid_similarity": [0.0] * n,
            "best_medoid_reference": [""] * n,
            "is_new_discovery": [True] * n,
        })
    else:
        X = load_and_normalize_vectors(emb_df["qvec"])  # (n, dims)

        # Cosine similarity via dot product (both matrices are L2-normalized)
        sim_matrix = X @ medoid_matrix.T  # (n, n_clusters)

        best_cluster_pos = np.argmax(sim_matrix, axis=1)  # position index
        best_sim = np.max(sim_matrix, axis=1)

        # Map position back to actual cluster ID
        cluster_id_arr = np.array(cluster_ids)
        best_class_id = cluster_id_arr[best_cluster_pos]

        is_new = best_sim < threshold
        # New discoveries get class_id = -1
        best_class_id = np.where(is_new, -1, best_class_id)

        # Look up the medoid reference for the best match
        best_medoid_ref = [
            medoid_meta[cluster_ids[pos]]["reference"] if not new else ""
            for pos, new in zip(best_cluster_pos, is_new)
        ]

        out_df = pd.DataFrame({
            incoming_id_col: emb_df[incoming_id_col].tolist(),
            "existing_class_id": best_class_id.astype(int),
            "best_medoid_similarity": np.round(best_sim.astype(float), 6),
            "best_medoid_reference": best_medoid_ref,
            "is_new_discovery": is_new.astype(bool),
        })

    n_new = int(out_df["is_new_discovery"].sum())
    n_known = n - n_new
    logger.info(
        "Classification: %d known / %d new discoveries (threshold=%.3f)",
        n_known, n_new, threshold,
    )

    # --- Save ---
    ds_dir = os.path.join(artifacts_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, "incoming_classification.parquet")
    out_df.to_parquet(out_path, index=False)
    logger.info("Classification saved: %s", out_path)

    return out_df
