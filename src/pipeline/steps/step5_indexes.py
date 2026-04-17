"""
Step 5 — Index Compilation (FAISS + BM25).

Provides two modes:
  - run_build_indexes:  Build fresh indexes from scratch (train mode).
  - run_merge_indexes:  Merge new embeddings into an existing versioned index (retrain mode).
"""
import argparse
import logging
import os
import pickle
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from config import Config, load_config
from helpers.du_engine import bm25_tokenize

logger = logging.getLogger(__name__)


def _artifact_paths(index_dir: str, strategy: str, dims: int) -> dict:
    return {
        "faiss":     os.path.join(index_dir, f"faiss_{strategy}_{dims}.index"),
        "doc_ids":   os.path.join(index_dir, f"doc_ids_{strategy}_{dims}.npy"),
        "bm25":      os.path.join(index_dir, f"bm25_{strategy}.pkl"),
        "meta":      os.path.join(index_dir, f"meta_{strategy}.parquet"),
        "inv_index": os.path.join(index_dir, f"inv_index_ids_{strategy}.pkl"),
        "vectors":   os.path.join(index_dir, f"vectors_{strategy}_{dims}.npy"),
    }


def _write_indexes(df_aligned: pd.DataFrame, index_dir: str, dims: int, strategy: str) -> dict:
    """
    Builds and writes FAISS, BM25, inverted index, metadata, and vector matrix
    for the given aligned DataFrame.
    """
    os.makedirs(index_dir, exist_ok=True)
    paths = _artifact_paths(index_dir, strategy, dims)

    # FAISS
    vecs = df_aligned["vector"].tolist()
    mat = np.ascontiguousarray(np.stack(vecs).astype("float32"))
    if mat.shape[1] != dims:
        raise ValueError(f"Vector dimension mismatch: got {mat.shape[1]}, expected {dims}")

    index = faiss.IndexFlatIP(dims)
    index.add(mat)
    faiss.write_index(index, paths["faiss"])
    logger.info("FAISS index saved: %s | ntotal=%d", paths["faiss"], index.ntotal)

    np.save(paths["vectors"], mat)
    logger.info("Vector matrix saved: %s", paths["vectors"])

    # Doc ID mapping
    doc_ids = df_aligned["Reference"].astype(str).tolist()
    np.save(paths["doc_ids"], np.array(doc_ids, dtype=object))
    logger.info("Doc IDs saved: %s", paths["doc_ids"])

    # BM25
    lex = df_aligned["lex_text"].fillna("").astype(str).tolist()
    tokenized = [bm25_tokenize(t) for t in lex]
    bm25 = BM25Okapi(tokenized)
    with open(paths["bm25"], "wb") as f:
        pickle.dump(bm25, f)
    logger.info("BM25 saved: %s", paths["bm25"])

    # Inverted ID index
    inv_index = defaultdict(list)
    for idx, text in enumerate(lex):
        for tok in bm25_tokenize(text):
            if ":" in tok:
                inv_index[tok].append(idx)
    with open(paths["inv_index"], "wb") as f:
        pickle.dump(dict(inv_index), f)
    logger.info("Inverted index saved: %s", paths["inv_index"])

    # Metadata parquet
    keep_cols = [
        "Reference", "Title", "Text", "dense_text", "lex_text",
        "Intent", "Category", "Reason", "Action", "Rubric", "System",
        "Side", "FAL", "has_match_code_header", "feature_status",
    ]
    keep_cols = [c for c in keep_cols if c in df_aligned.columns]
    df_aligned[keep_cols].to_parquet(paths["meta"], index=False)
    logger.info("Metadata parquet saved: %s", paths["meta"])

    return paths


def _load_and_align(emb_parquet: str, ready_parquet: str) -> pd.DataFrame:
    df_emb = pd.read_parquet(emb_parquet)
    df_emb["Reference"] = df_emb["Reference"].astype(str).str.strip()
    df_emb = df_emb.drop_duplicates(subset=["Reference"], keep="first").dropna(subset=["vector"])

    df_ready = pd.read_parquet(ready_parquet)
    df_ready["Reference"] = df_ready["Reference"].astype(str).str.strip()
    df_ready = df_ready.drop_duplicates(subset=["Reference"], keep="first")

    df = df_emb.merge(df_ready, on="Reference", how="inner")
    logger.info("Aligned rows: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_build_indexes(
    cfg: Config,
    df_embeddings: pd.DataFrame = None,
    df_texts: pd.DataFrame = None,
    version_dir: str = None,
) -> dict:
    """
    Builds FAISS + BM25 indexes from scratch (train mode).

    Args:
        cfg:           Global configuration object.
        df_embeddings: Embeddings DataFrame (Reference + vector). Loaded from disk if None.
        df_texts:      Vectorization-ready DataFrame (dense_text + lex_text). Loaded from disk if None.
        version_dir:   Version artifact directory (e.g. artifacts/v1). Defaults to cfg.paths.artifacts_dir.

    Returns:
        Dict of artifact paths written.
    """
    if version_dir is None:
        version_dir = cfg.paths.artifacts_dir

    strategy = cfg.processing.embedding_strategy
    dims = cfg.api.embedding_dims
    index_dir = os.path.join(version_dir, "indexes")
    embeddings_dir = os.path.join(version_dir, "embeddings")
    datasets_dir = os.path.join(version_dir, "datasets")

    emb_path = os.path.join(embeddings_dir, f"embeddings_{strategy}_{dims}d_DU.parquet")
    ready_path = os.path.join(datasets_dir, "ready_for_vectorization_DU.parquet")

    # Fall back to default artifacts layout if version_dir == artifacts_dir
    if not os.path.exists(emb_path):
        emb_path = os.path.join(cfg.paths.artifacts_dir, "embeddings", f"embeddings_{strategy}_{dims}d_DU.parquet")
    if not os.path.exists(ready_path):
        ready_path = os.path.join(cfg.paths.artifacts_dir, "datasets", "ready_for_vectorization_DU.parquet")

    if df_embeddings is None or df_texts is None:
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Missing embeddings: {emb_path}")
        if not os.path.exists(ready_path):
            raise FileNotFoundError(f"Missing ready dataset: {ready_path}")
        df_aligned = _load_and_align(emb_path, ready_path)
    else:
        df_emb = df_embeddings.copy()
        df_emb["Reference"] = df_emb["Reference"].astype(str).str.strip()
        df_emb = df_emb.drop_duplicates(subset=["Reference"], keep="first").dropna(subset=["vector"])
        df_txt = df_texts.copy()
        df_txt["Reference"] = df_txt["Reference"].astype(str).str.strip()
        df_aligned = df_emb.merge(df_txt, on="Reference", how="inner")
        logger.info("Aligned rows (in-memory): %d", len(df_aligned))

    logger.info("Step 5 (build) — building indexes in: %s", index_dir)
    return _write_indexes(df_aligned, index_dir, dims, strategy)


def run_merge_indexes(
    cfg: Config,
    df_new_embeddings: pd.DataFrame = None,
    df_new_texts: pd.DataFrame = None,
    current_version_dir: str = None,
    new_version_dir: str = None,
) -> dict:
    """
    Merges new embeddings into existing indexes (retrain mode).

    Steps:
      1. Load existing vectors from current_version_dir.
      2. Merge new embeddings, deduplicating on Reference (new wins).
      3. Write full combined indexes to new_version_dir.

    Args:
        cfg:                  Global configuration object.
        df_new_embeddings:    New embeddings DataFrame. Loaded from disk if None.
        df_new_texts:         New vectorization-ready DataFrame. Loaded from disk if None.
        current_version_dir:  Directory of the currently active version (e.g. artifacts/v1).
        new_version_dir:      Directory to write the merged version to (e.g. artifacts/v2).

    Returns:
        Dict of artifact paths written.
    """
    if current_version_dir is None:
        current_version_dir = cfg.paths.artifacts_dir
    if new_version_dir is None:
        raise ValueError("new_version_dir must be specified for retrain (merge) mode.")

    strategy = cfg.processing.embedding_strategy
    dims = cfg.api.embedding_dims

    # Load current (existing) embeddings
    cur_emb_path = os.path.join(current_version_dir, "embeddings", f"embeddings_{strategy}_{dims}d_DU.parquet")
    cur_ready_path = os.path.join(current_version_dir, "datasets", "ready_for_vectorization_DU.parquet")

    # Fallback to flat layout
    if not os.path.exists(cur_emb_path):
        cur_emb_path = os.path.join(cfg.paths.artifacts_dir, "embeddings", f"embeddings_{strategy}_{dims}d_DU.parquet")
    if not os.path.exists(cur_ready_path):
        cur_ready_path = os.path.join(cfg.paths.artifacts_dir, "datasets", "ready_for_vectorization_DU.parquet")

    logger.info("Step 5 (merge) — loading current version from: %s", current_version_dir)
    df_cur = _load_and_align(cur_emb_path, cur_ready_path)

    # Prepare new data
    if df_new_embeddings is None or df_new_texts is None:
        new_emb_path = os.path.join(cfg.paths.artifacts_dir, "embeddings", f"embeddings_{strategy}_{dims}d_DU.parquet")
        new_ready_path = os.path.join(cfg.paths.artifacts_dir, "datasets", "ready_for_vectorization_DU.parquet")
        df_new = _load_and_align(new_emb_path, new_ready_path)
    else:
        df_ne = df_new_embeddings.copy()
        df_ne["Reference"] = df_ne["Reference"].astype(str).str.strip()
        df_ne = df_ne.drop_duplicates(subset=["Reference"], keep="first").dropna(subset=["vector"])
        df_nt = df_new_texts.copy()
        df_nt["Reference"] = df_nt["Reference"].astype(str).str.strip()
        df_new = df_ne.merge(df_nt, on="Reference", how="inner")

    logger.info("Current docs: %d | New docs: %d", len(df_cur), len(df_new))

    # Merge: new records win on duplicate Reference
    df_merged = pd.concat([df_cur, df_new], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["Reference"], keep="last").reset_index(drop=True)
    logger.info("Merged total: %d docs", len(df_merged))

    # Write new version indexes
    new_index_dir = os.path.join(new_version_dir, "indexes")
    logger.info("Writing merged indexes to: %s", new_index_dir)
    return _write_indexes(df_merged, new_index_dir, dims, strategy)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Step 5 — Build or merge FAISS/BM25 indexes.")
    parser.add_argument("--mode", choices=["build", "merge"], default="build")
    parser.add_argument("--version_dir", type=str, default=cfg.paths.artifacts_dir, help="Output version directory.")
    parser.add_argument("--current_version_dir", type=str, default=None, help="Current version dir (merge mode only).")
    args = parser.parse_args()

    if args.mode == "build":
        run_build_indexes(cfg, version_dir=args.version_dir)
    else:
        if not args.current_version_dir:
            parser.error("--current_version_dir is required for merge mode.")
        run_merge_indexes(cfg, current_version_dir=args.current_version_dir, new_version_dir=args.version_dir)
