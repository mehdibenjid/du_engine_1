"""
Step 4 — Embedding Generation.

Sends dense_text strings to the Titan embedding API using concurrent threads
and writes the resulting vectors to a parquet file.
"""
import argparse
import concurrent.futures
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import Config, load_config
from helpers.auth import get_valid_token
from helpers.du_engine import build_session, embed_one_safe

logger = logging.getLogger(__name__)

_EXPECTED_COLUMNS = ["Reference", "vector", "model_input", "error"]

class TqdmToLogger:
    """Outputs tqdm progress to a standard python logger."""
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, buf):
        # Strip carriage returns and whitespace
        msg = buf.strip('\r\n\t ')
        if msg: # Avoid logging empty strings
            self.logger.log(self.level, msg)

    def flush(self):
        # flush is required by tqdm, but we don't need to do anything here
        pass
# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_generate_embeddings(
    cfg: Config,
    df: Optional[pd.DataFrame] = None,
    artifacts_dir: Optional[str] = None,
    strategy: Optional[str] = None,
    max_workers: Optional[int] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Embeds dense_text strings and writes the vectors to a parquet file.

    Args:
        cfg:           Global configuration object.
        df:            DataFrame with 'Reference' and 'dense_text' columns.
                       Loaded from disk (ready_for_vectorization_DU.parquet) if None.
        artifacts_dir: Overrides cfg.paths.artifacts_dir.
        strategy:      Embedding strategy label (default: cfg.processing.embedding_strategy).
        max_workers:   Thread count (default: cfg.retrieval.max_workers).
        timeout:       API timeout seconds (default: cfg.retrieval.timeout).
        retries:       Retry attempts per document (default: cfg.retrieval.retries).
        resume:        If True, skip already-embedded references found in the output file.

    Returns:
        DataFrame with columns: Reference, vector, model_input, error.

    Side effects:
        Writes:
          - <artifacts_dir>/embeddings/embeddings_<strategy>_<dims>d_DU.parquet
          - <artifacts_dir>/embeddings/embedding_run_report_<strategy>_<dims>d.json
    """
    if artifacts_dir is None:
        artifacts_dir = cfg.paths.artifacts_dir
    if strategy is None:
        strategy = cfg.processing.embedding_strategy
    if max_workers is None:
        max_workers = cfg.retrieval.max_workers
    if timeout is None:
        timeout = cfg.retrieval.timeout
    if retries is None:
        retries = cfg.retrieval.retries

    dims = cfg.api.embedding_dims
    embeddings_dir = os.path.join(artifacts_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    out_path = os.path.join(embeddings_dir, f"embeddings_{strategy}_{dims}d_DU.parquet")

    # Load data
    if df is None:
        ready_path = os.path.join(artifacts_dir, "datasets", "ready_for_vectorization_DU.parquet")
        if not os.path.exists(ready_path):
            raise FileNotFoundError(f"Missing input parquet: {ready_path}")
        df = pd.read_parquet(ready_path)
        logger.info("Step 4 - loaded parquet: %s", ready_path)

    df = df.copy()
    for _required_col in ("Reference", "dense_text"):
        if _required_col not in df.columns:
            raise ValueError(
                f"Missing required column '{_required_col}'. Available: {list(df.columns)}"
            )
    df["Reference"] = df["Reference"].astype(str).str.strip()

    if strategy == "C":
        df["model_input"] = df["dense_text"].fillna("").astype(str).str.strip()
    else:
        title = df.get("Title", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        text = df.get("Text", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        df["model_input"] = ("Title: " + title + "\n" + text).str.strip()

    valid_mask = df["model_input"].str.len() > 0
    logger.info("Valid inputs: %d / %d", int(valid_mask.sum()), len(df))

    # Resume: skip already processed references
    if resume and os.path.exists(out_path):
        prev = pd.read_parquet(out_path)
        existing_refs = set(prev["Reference"].astype(str).tolist())
        keep_mask = ~df["Reference"].isin(existing_refs)
        df = df[keep_mask].copy()
        valid_mask = df["model_input"].str.len() > 0
        logger.info("Resume: %d already embedded | %d remaining", len(existing_refs), len(df))

    if len(df) == 0 or valid_mask.sum() == 0:
        logger.info("Nothing to embed.")
        # Return the existing file if present; otherwise return an empty DataFrame
        # with the expected schema so callers never receive a bare, column-less frame.
        if os.path.exists(out_path):
            return pd.read_parquet(out_path)
        return pd.DataFrame(columns=_EXPECTED_COLUMNS)

    vectors = [None] * len(df)
    errors = [None] * len(df)
    indices = df[valid_mask].index.tolist()
    idx_to_pos = {idx: pos for pos, idx in enumerate(df.index.tolist())}

    effective_workers = min(max_workers, len(indices))
    session = build_session(effective_workers)
    token = get_valid_token()
    if token is None:
        raise RuntimeError(
            "Failed to obtain a valid authentication token. "
            "Check certificate files and SSO connectivity."
        )
    api_url = cfg.api.base_url
    model_id = cfg.api.embedding_model_id
    namespace = cfg.api.namespace
    logger.info("Embedding %d documents with %d workers...", len(indices), effective_workers)
    start = time.perf_counter()

    def _submit_one(i):
        vec, _, err = embed_one_safe(
            session, df.at[i, "model_input"], token,
            timeout=timeout, retries=retries,
            api_url=api_url, model_id=model_id, namespace=namespace, dims=dims,
        )
        return i, vec.tolist() if vec is not None else None, err

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as ex:
        futs = [ex.submit(_submit_one, i) for i in indices]
        
        # 3. Pass the wrapper to 'file' and set 'mininterval'
        for fut in tqdm(
            concurrent.futures.as_completed(futs), 
            total=len(futs),
            file=tqdm_out,        # Redirect output to logger
            mininterval=20.0,     # ONLY log progress every 30 seconds to prevent spam
            ascii=True            # Use basic ASCII characters to prevent weird log artifacts
        ):
            i, vec, err = fut.result()
            pos = idx_to_pos[i]
            vectors[pos] = vec
            errors[pos] = err

    dur = time.perf_counter() - start
    out_df = pd.DataFrame({"Reference": df["Reference"].astype(str).tolist(), "vector": vectors, "model_input": df["model_input"].astype(str).tolist(), "error": errors})
    success = int(out_df["vector"].notna().sum())
    fail = len(out_df) - success
    logger.info("Embedding done in %.2fs | success=%d | fail=%d", dur, success, fail)
    if fail > 0:
        fail_refs = out_df[out_df["vector"].isna()]["Reference"].tolist()
        logger.warning("Failed references (%d, first 20): %s", fail, fail_refs[:20])

    # Merge with previous if resuming
    if resume and os.path.exists(out_path):
        prev = pd.read_parquet(out_path)
        out_df = pd.concat([prev, out_df], ignore_index=True)
        out_df["has_vec"] = out_df["vector"].notna()
        out_df = out_df.sort_values(["Reference", "has_vec"], ascending=[True, False]).drop_duplicates(subset=["Reference"], keep="first").drop(columns=["has_vec"])

    out_df.to_parquet(out_path, index=False)
    logger.info("Saved embeddings: %s", out_path)

    report = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "strategy": strategy,
        "dims": dims,
        "rows": int(len(out_df)),
        "success": success,
        "fail": fail,
        "duration_sec": round(dur, 2),
    }
    rep_path = os.path.join(embeddings_dir, f"embedding_run_report_{strategy}_{dims}d.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return out_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4 - Generate embeddings via Titan API.")
    parser.add_argument("--artifacts_dir", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None, choices=["C", "D"])
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    run_generate_embeddings(
        cfg,
        artifacts_dir=args.artifacts_dir or cfg.paths.artifacts_dir,
        strategy=args.strategy or cfg.processing.embedding_strategy,
        max_workers=args.max_workers or cfg.retrieval.max_workers,
        timeout=args.timeout or cfg.retrieval.timeout,
        retries=args.retries or cfg.retrieval.retries,
        resume=args.resume,
    )
