"""
Step 9 — Global Corpus Clustering.

Runs UMAP + HDBSCAN on the full corpus vectors produced by step 5,
computes cluster medoids, and persists them as "class anchors" for
future inference-time classification.

Used by TRAIN and RETRAIN modes.
"""
import json
import logging
import warnings
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import hdbscan
import numpy as np
import pandas as pd
import umap


from config import Config
from helpers.clustering import (
    compute_medoid_indices,
    extract_medoid_vectors,
)

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


def run_global_clustering(
    cfg: Config,
    version_dir: str,
    random_state: Optional[int] = None,
) -> Dict[int, List[float]]:
    """
    Clusters the full corpus embeddings and persists medoid vectors.

    Steps:
        1. Load vectors and doc_ids from version_dir/indexes/
        2. L2-normalize the matrix
        3. UMAP dimensionality reduction (with random_state for reproducibility)
        4. HDBSCAN clustering
        5. Compute medoid vectors per cluster
        6. Save cluster_medoids.json to version_dir/metadata/
        7. Save corpus_clusters.parquet to version_dir/datasets/

    Args:
        cfg:          Global configuration.
        version_dir:  Versioned artifact directory (e.g. artifacts/v1).
        random_state: Override; defaults to cfg.clustering.random_state.

    Returns:
        Dict mapping cluster_id -> medoid vector as list of floats.
    """
    cl = cfg.clustering
    strategy = cfg.processing.embedding_strategy
    dims = cfg.api.embedding_dims

    if random_state is None:
        random_state = cl.random_state

    # --- 1. Load vectors and doc IDs ---
    index_dir = os.path.join(version_dir, "indexes")
    vectors_path = os.path.join(index_dir, f"vectors_{strategy}_{dims}.npy")
    doc_ids_path = os.path.join(index_dir, f"doc_ids_{strategy}_{dims}.npy")

    if not os.path.exists(vectors_path):
        raise FileNotFoundError(
            f"Vectors file not found: {vectors_path}. "
            "Ensure step 5 (index build) completed successfully."
        )
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(
            f"Doc IDs file not found: {doc_ids_path}. "
            "Ensure step 5 (index build) completed successfully."
        )

    logger.info("Step 9 — loading corpus vectors: %s", vectors_path)
    X = np.load(vectors_path).astype(np.float32)
    doc_ids = np.load(doc_ids_path, allow_pickle=True).tolist()

    n, d = X.shape
    logger.info("Corpus: %d vectors | %d dims", n, d)

    if n < 2:
        logger.warning("Too few vectors (%d) for clustering — skipping.", n)
        return {}

    # --- 2. L2-normalize (defensive — step 5 stores raw FAISS vectors) ---
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # --- 3. UMAP reduction ---
    if cl.use_umap and n >= 5:
        safe_components = min(cl.umap_n_components, max(2, n - 2))
        safe_neighbors = min(cl.umap_n_neighbors, max(2, n - 1))
        logger.info(
            "UMAP: n_components=%d, n_neighbors=%d, min_dist=%.2f, random_state=%s",
            safe_components, safe_neighbors, cl.umap_min_dist, random_state,
        )
        reducer = umap.UMAP(
            n_neighbors=safe_neighbors,
            n_components=safe_components,
            min_dist=cl.umap_min_dist,
            metric="cosine",
            random_state=random_state,
        )
        Z = np.asarray(reducer.fit_transform(X), dtype=np.float32)
        logger.info("UMAP done: %s -> %s", X.shape, Z.shape)
    else:
        Z = X

    # --- 4. HDBSCAN ---
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cl.hdbscan_min_cluster_size,
        min_samples=cl.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(Z)
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("HDBSCAN: %d clusters | %d noise (%d total)", n_clusters, n_noise, n)

    if n_clusters == 0:
        logger.warning("No clusters found — medoids file will be empty.")

    # --- 5. Compute medoids (on original high-dim space for accurate anchors) ---
    medoid_indices = compute_medoid_indices(X, labels)
    medoid_vectors = extract_medoid_vectors(X, labels)

    # Compute cluster sizes
    cluster_sizes: Dict[int, int] = {}
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        cluster_sizes[c] = int((labels == c).sum())

    # --- 6. Save cluster_medoids.json ---
    metadata_dir = os.path.join(version_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    medoids_payload: Dict[str, dict] = {}
    for cid, vec in medoid_vectors.items():
        midx = medoid_indices[cid]
        medoids_payload[str(cid)] = {
            "vector": vec.tolist(),
            "reference": str(doc_ids[midx]),
            "cluster_size": cluster_sizes.get(cid, 0),
        }

    # Extract the version label from the directory name (e.g. "v1" from ".../artifacts/v1")
    version_label = os.path.basename(version_dir)

    medoids_json = {
        "version": version_label,
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "n_clusters": n_clusters,
        "embedding_dims": d,
        "clustering_params": {
            "use_umap": cl.use_umap,
            "umap_n_components": cl.umap_n_components,
            "umap_n_neighbors": cl.umap_n_neighbors,
            "umap_min_dist": cl.umap_min_dist,
            "hdbscan_min_cluster_size": cl.hdbscan_min_cluster_size,
            "hdbscan_min_samples": cl.hdbscan_min_samples,
            "random_state": random_state,
        },
        "medoids": medoids_payload,
    }

    medoids_path = os.path.join(metadata_dir, "cluster_medoids.json")
    with open(medoids_path, "w", encoding="utf-8") as f:
        json.dump(medoids_json, f, indent=2, ensure_ascii=False)
    logger.info("Medoids saved: %s | %d clusters", medoids_path, n_clusters)

    # --- 7. Save corpus_clusters.parquet ---
    datasets_dir = os.path.join(version_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    clusters_df = pd.DataFrame({
        "reference": [str(r) for r in doc_ids],
        "cluster_id": labels.astype(int),
    })
    clusters_path = os.path.join(datasets_dir, "corpus_clusters.parquet")
    clusters_df.to_parquet(clusters_path, index=False)
    logger.info("Corpus clusters saved: %s | %d rows", clusters_path, len(clusters_df))

    return {cid: vec.tolist() for cid, vec in medoid_vectors.items()}
