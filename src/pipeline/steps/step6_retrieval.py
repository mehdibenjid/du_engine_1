"""
Step 6 — Hybrid Batch Retrieval.

Loads the DUEngine, prepares incoming queries, runs hybrid search (FAISS + BM25
+ RRF + MMR), and writes the top-K results parquet.
"""
import argparse
import json
import logging
import os
from typing import Optional

import pandas as pd

from config import Config, load_config
from engine.du_engine import DUEngine
from helpers.processing import detect_csv_separator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_retrieval(
    cfg: Config,
    incoming_path: str,
    artifacts_dir: Optional[str] = None,
    match_code: Optional[bool] = None,
    top_k: Optional[int] = None,
    candidate_k: Optional[int] = None,
    bm25_candidate_k: Optional[int] = None,
    use_hybrid: bool = True,
    rrf_k: Optional[int] = None,
    w_dense: Optional[float] = None,
    w_bm25: Optional[float] = None,
    boost_cat: Optional[float] = None,
    boost_int: Optional[float] = None,
    use_mmr: Optional[bool] = None,
    mmr_lambda: Optional[float] = None,
    max_workers: Optional[int] = None,
    timeout: Optional[int] = None,
    retries: Optional[int] = None,
    exclude_self: bool = False,
    save_embeddings: bool = False,
    debug_timing: bool = False,
    show_progress: bool = True,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Runs hybrid retrieval on an incoming CSV and writes top-K results.

    Args:
        cfg:             Global configuration object.
        incoming_path:   Path to the incoming queries CSV (comma-separated).
        artifacts_dir:   Artifacts directory to load indexes from. Defaults to cfg.paths.artifacts_dir.
        match_code:      Filter to regex-header rows only.
        top_k … retries: Override search parameters (fall back to config values if None).
        exclude_self:    Exclude self-matches (useful for evaluating on training data).
        save_embeddings: If True, also saves incoming embeddings to parquet.
        debug_timing:    If True, also saves timing metrics.
        show_progress:   Show TQDM progress bar.
        output_path:     Override default output parquet path.

    Returns:
        Results DataFrame.

    Side effects:
        Writes:
          - <artifacts_dir>/datasets/incoming_topk_results.parquet
          - <artifacts_dir>/datasets/incoming_embeddings.parquet (if save_embeddings)
          - <artifacts_dir>/datasets/incoming_timings.csv (if debug_timing)
    """
    if artifacts_dir is None:
        artifacts_dir = cfg.paths.artifacts_dir
    if match_code is None:
        match_code = cfg.processing.match_code

    # Fill from config where not overridden
    r = cfg.retrieval
    top_k            = top_k            if top_k            is not None else r.top_k
    candidate_k      = candidate_k      if candidate_k      is not None else r.candidate_k
    bm25_candidate_k = bm25_candidate_k if bm25_candidate_k is not None else r.bm25_candidate_k
    rrf_k            = rrf_k            if rrf_k            is not None else r.rrf_k
    w_dense          = w_dense          if w_dense          is not None else r.w_dense
    w_bm25           = w_bm25           if w_bm25           is not None else r.w_bm25
    boost_cat        = boost_cat        if boost_cat        is not None else r.boost_cat
    boost_int        = boost_int        if boost_int        is not None else r.boost_int
    use_mmr          = use_mmr          if use_mmr          is not None else r.use_mmr
    mmr_lambda       = mmr_lambda       if mmr_lambda       is not None else r.mmr_lambda
    max_workers      = max_workers      if max_workers      is not None else r.max_workers
    timeout          = timeout          if timeout          is not None else r.timeout
    retries          = retries          if retries          is not None else r.retries

    datasets_dir = os.path.join(artifacts_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(datasets_dir, "incoming_topk_results.parquet")

    abs_incoming = os.path.abspath(incoming_path)
    if not os.path.exists(abs_incoming):
        raise FileNotFoundError(f"Incoming CSV not found: {abs_incoming}")
    logger.info("Step 6 - loading incoming CSV: %s", abs_incoming)
    _sep = cfg.processing.csv_separator or detect_csv_separator(abs_incoming)
    df_raw = pd.read_csv(abs_incoming, sep=_sep, low_memory=False)

    logger.info("Initializing DUEngine from: %s", artifacts_dir)
    engine = DUEngine(artifacts_dir)

    logger.info("Preparing %d incoming queries...", len(df_raw))
    df_in = engine.prepare_incoming(df_raw, match_code=match_code)
    logger.info("Queries after filtering: %d", len(df_in))

    result = engine.retrieve_batch(
        df_incoming_prepared=df_in,
        top_k=top_k,
        candidate_k=candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        filters=None,
        use_hybrid=use_hybrid,
        exclude_self=exclude_self,
        rrf_k=rrf_k,
        w_dense=w_dense,
        w_bm25=w_bm25,
        boost_cat=boost_cat,
        boost_int=boost_int,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        debug_timing=debug_timing,
        max_workers=max_workers,
        timeout=timeout,
        retries=retries,
        show_progress=show_progress,
        return_query_embeddings=save_embeddings,
    )

    # Unpack result tuple based on flags. Guard against unexpected return shapes
    # so a future API change in retrieve_batch produces a clear error.
    emb_df = None
    timing = None
    try:
        if debug_timing and save_embeddings:
            res, timing, emb_df = result
        elif debug_timing:
            res, timing = result
        elif save_embeddings:
            res, emb_df = result
        else:
            res = result
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"retrieve_batch returned an unexpected structure for "
            f"debug_timing={debug_timing}, save_embeddings={save_embeddings}. "
            f"Got: {type(result)}. Original error: {exc}"
        ) from exc

    if not res.empty:
        res["run_arguments"] = json.dumps({
            "top_k": top_k, "candidate_k": candidate_k, "bm25_candidate_k": bm25_candidate_k,
            "rrf_k": rrf_k, "w_dense": w_dense, "w_bm25": w_bm25,
            "boost_cat": boost_cat, "boost_int": boost_int,
            "use_mmr": use_mmr, "mmr_lambda": mmr_lambda,
        })

    res.to_parquet(output_path, index=False)
    logger.info("Results saved: %s | rows=%d", output_path, len(res))

    if save_embeddings and emb_df is not None and not emb_df.empty:
        emb_out = os.path.join(datasets_dir, "incoming_embeddings.parquet")
        emb_df.to_parquet(emb_out, index=False)
        logger.info("Incoming embeddings saved: %s", emb_out)

    if debug_timing and timing is not None:
        timing_out = os.path.join(datasets_dir, "incoming_timings.csv")
        timing.to_csv(timing_out, index=False)
        logger.info("Timings saved: %s", timing_out)

    return res


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 6 - Hybrid batch retrieval.")
    parser.add_argument("--incoming_csv", required=True)
    parser.add_argument("--artifacts_dir", type=str, default=None)
    parser.add_argument("--match_code", action=argparse.BooleanOptionalAction, default=None,
                        help="Override match_code filter (--match_code / --no-match_code). Defaults to config value.")
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--bm25_candidate_k", type=int, default=None)
    parser.add_argument("--no_hybrid", action="store_true")
    parser.add_argument("--rrf_k", type=int, default=None)
    parser.add_argument("--w_dense", type=float, default=None)
    parser.add_argument("--w_bm25", type=float, default=None)
    parser.add_argument("--boost_cat", type=float, default=None)
    parser.add_argument("--boost_int", type=float, default=None)
    parser.add_argument("--use_mmr", action=argparse.BooleanOptionalAction, default=None,
                        help="Override MMR (--use_mmr / --no-use_mmr). Defaults to config value.")
    parser.add_argument("--mmr_lambda", type=float, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--exclude_self", action="store_true")
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--debug_timing", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    run_retrieval(
        cfg, incoming_path=args.incoming_csv,
        artifacts_dir=args.artifacts_dir or cfg.paths.artifacts_dir,
        match_code=args.match_code,
        top_k=args.top_k, candidate_k=args.candidate_k,
        bm25_candidate_k=args.bm25_candidate_k, use_hybrid=not args.no_hybrid,
        rrf_k=args.rrf_k, w_dense=args.w_dense, w_bm25=args.w_bm25,
        boost_cat=args.boost_cat, boost_int=args.boost_int,
        use_mmr=args.use_mmr, mmr_lambda=args.mmr_lambda,
        max_workers=args.max_workers, timeout=args.timeout, retries=args.retries,
        exclude_self=args.exclude_self, save_embeddings=args.save_embeddings,
        debug_timing=args.debug_timing, show_progress=not args.no_progress,
        output_path=args.out,
    )
