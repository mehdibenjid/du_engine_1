"""
Clustering utilities: cosine similarity, medoid selection, MMR diversity selection,
metadata distribution summaries, and vector loading/normalization.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Computes a pairwise cosine similarity matrix from row-normalized vectors."""
    return np.clip(X @ X.T, -1.0, 1.0)


def extract_medoid_vectors(X: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Returns the actual medoid vector for each cluster.

    Args:
        X:      Row-normalized embedding matrix, shape (n, dims).
        labels: Cluster label per row, shape (n,).

    Returns:
        Dict mapping cluster_id -> medoid vector (1-D ndarray of shape (dims,)).
    """
    medoid_indices = compute_medoid_indices(X, labels)
    return {cid: X[idx].copy() for cid, idx in medoid_indices.items()}


def compute_medoid_indices(X: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    """
    Returns the global index of the medoid (minimum average distance point) for each cluster.
    Noise points (label == -1) are skipped.

    Args:
        X:      Row-normalized embedding matrix, shape (n, dims).
        labels: Cluster label per row, shape (n,).

    Returns:
        Dict mapping cluster_id → row index of its medoid in X.
    """
    medoids: Dict[int, int] = {}
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        if idx.size == 1:
            medoids[c] = int(idx[0])
            continue
        sub = X[idx]
        dists = 1.0 - cosine_sim_matrix(sub)
        medoids[c] = int(idx[int(np.argmin(dists.mean(axis=1)))])
    return medoids


def mmr_select(
    X: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lambda_: float = 0.6,
    already_selected: Optional[List[int]] = None,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) selection.
    Returns up to k indices balancing relevance (lambda_) vs. diversity (1 - lambda_).

    Args:
        X:                Row-normalized embeddings for the candidate set.
        relevance:        Relevance score per row (e.g. cosine similarity to query/medoid).
        k:                Number of items to select.
        lambda_:          Trade-off weight: 1.0 = pure relevance, 0.0 = pure diversity.
        already_selected: Indices to treat as already selected (e.g. the medoid).

    Returns:
        List of selected local indices (within X).
    """
    n = int(X.shape[0])
    if n == 0:
        return []
    k = min(int(k), n)
    selected: List[int] = list(already_selected) if already_selected else []
    if len(selected) >= k:
        return selected[:k]
    sims = cosine_sim_matrix(X)
    candidates = [i for i in range(n) if i not in selected]
    while len(selected) < k and candidates:
        best_i, best_score = None, -1e18
        for i in candidates:
            score = (
                float(relevance[i])
                if not selected
                else lambda_ * float(relevance[i]) - (1.0 - lambda_) * max(float(sims[i, j]) for j in selected)
            )
            if score > best_score:
                best_score, best_i = score, i
        if best_i is None:
            break
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


def top_metadata_distributions(df: pd.DataFrame, fields: List[str], top_n: int = 5) -> Dict:
    """
    Returns top-N value distributions (value, percentage, count) per field.
    Used to populate cluster metadata summaries for LLM prompts.
    """
    out: Dict = {}
    for f in fields:
        if f not in df.columns:
            continue
        vc = df[f].fillna("NA").astype(str).str.strip().value_counts(dropna=False)
        total = int(vc.sum())
        out[f] = [
            (str(val), 0.0 if total == 0 else int(cnt) / total * 100.0, int(cnt))
            for val, cnt in vc.head(top_n).items()
        ]
    return out


def load_and_normalize_vectors(qvec_series: pd.Series) -> np.ndarray:
    """
    Loads vectors from a Series of array-like values, fills missing rows with zeros,
    and L2-normalizes the matrix.

    Args:
        qvec_series: pd.Series where each element is a list or ndarray of floats.

    Returns:
        Float32 ndarray of shape (n, dims), L2-normalized.

    Raises:
        ValueError: If all values are missing or invalid.
    """
    vecs = []
    for v in qvec_series.tolist():
        try:
            arr = np.asarray(v, dtype=np.float32) if v is not None else None
        except Exception:
            arr = None
        vecs.append(arr)

    dims = next((arr.shape[0] for arr in vecs if arr is not None and arr.ndim == 1), None)
    if dims is None:
        raise ValueError("All qvec values are missing — cannot cluster.")

    X = np.zeros((len(vecs), dims), dtype=np.float32)
    missing = 0
    for i, arr in enumerate(vecs):
        if arr is None or arr.ndim != 1 or arr.shape[0] != dims:
            missing += 1
        else:
            X[i] = arr

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    if missing:
        logger.warning("%d vectors missing/invalid — kept as zeros.", missing)
    return X
