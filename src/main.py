"""
DU Engine — single CLI entry point.

Usage:
    python main.py train   --path data/raw/input.csv [--match_code]
    python main.py retrain --path data/raw/new.csv   [--match_code]
    python main.py infer   --path data/incoming.csv  [--topk] [--clustering] [--summarize]
                           [--top_k N] [--candidate_k N] [--no_hybrid] [--exclude_self]
"""
import argparse
import logging
import sys
import colorlog

from config import load_config
from pipeline.runner import run_infer, run_retrain, run_train


def _configure_logging(level: str = "INFO") -> None:
    # 1. Create a stream handler for stdout
    handler = colorlog.StreamHandler(stream=sys.stdout)
    
    # 2. Define the colored formatter (keeping your exact layout)
    formatter = colorlog.ColoredFormatter(
        # The %(log_color)s variable dictates the color for the whole line
        "%(log_color)s%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    handler.setFormatter(formatter)
    
    # 3. Apply it to the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Prevent adding multiple handlers if the script is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="du-engine",
        description="DU Engine — train, retrain, or infer.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logging level (default: INFO).",
    )
    parser.add_argument(
        "--config_dir",
        default=None,
        help="Override path to the config/ directory.",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ------------------------------------------------------------------ train
    train_p = subparsers.add_parser("train", help="Full corpus build (steps 1–5).")
    train_p.add_argument("--path", required=True, help="Path to raw CSV input.")
    train_p.add_argument("--match_code", action="store_true", help="Filter to rows with a regex header.")

    # ---------------------------------------------------------------- retrain
    retrain_p = subparsers.add_parser("retrain", help="Incremental retrain + index merge (steps 1–5).")
    retrain_p.add_argument("--path", required=True, help="Path to new raw CSV input.")
    retrain_p.add_argument("--match_code", action="store_true")

    # ------------------------------------------------------------------ infer
    infer_p = subparsers.add_parser("infer", help="Inference pipeline (steps 6–8).")
    infer_p.add_argument("--path", required=True, help="Path to incoming queries CSV.")

    # Stage flags (all default off so callers opt-in explicitly)
    infer_p.add_argument("--topk", action="store_true", default=True,
                         help="Run step 6 hybrid retrieval (default: on).")
    infer_p.add_argument("--no_topk", dest="topk", action="store_false",
                         help="Skip step 6.")
    infer_p.add_argument("--clustering", action="store_true",
                         help="Run step 7 UMAP + HDBSCAN clustering.")
    infer_p.add_argument("--summarize", action="store_true",
                         help="Run step 8 LLM cluster summarization.")
    infer_p.add_argument("--language", choices=["fr", "en"], default=None,
                         help="LLM output language for summaries (default: from config, 'fr').")

    # Retrieval overrides forwarded as **retrieval_kwargs
    infer_p.add_argument("--top_k", type=int, default=None)
    infer_p.add_argument("--candidate_k", type=int, default=None)
    infer_p.add_argument("--bm25_candidate_k", type=int, default=None)
    infer_p.add_argument("--no_hybrid", action="store_true",
                         help="Disable hybrid mode (dense only).")
    infer_p.add_argument("--rrf_k", type=int, default=None)
    infer_p.add_argument("--w_dense", type=float, default=None)
    infer_p.add_argument("--w_bm25", type=float, default=None)
    infer_p.add_argument("--boost_cat", type=float, default=None)
    infer_p.add_argument("--boost_int", type=float, default=None)
    infer_p.add_argument("--use_mmr", action="store_true", default=None)
    infer_p.add_argument("--mmr_lambda", type=float, default=None)
    infer_p.add_argument("--max_workers", type=int, default=None)
    infer_p.add_argument("--exclude_self", action="store_true")
    infer_p.add_argument("--classify", action="store_true",
                         help="Run medoid-based classification against corpus clusters.")
    infer_p.add_argument("--threshold", type=float, default=0.75,
                         help="Cosine similarity threshold for 'new discovery' marking (default: 0.75).")
    infer_p.add_argument("--save_embeddings", action="store_true",
                         help="Save incoming embeddings parquet (required if --clustering or --classify).")
    infer_p.add_argument("--debug_timing", action="store_true")
    infer_p.add_argument("--no_progress", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    cfg = load_config(config_dir=args.config_dir)
    logger.info("Config loaded | mode=%s", args.mode)

    if args.mode == "train":
        run_train(cfg, input_path=args.path, match_code=args.match_code)

    elif args.mode == "retrain":
        run_retrain(cfg, input_path=args.path, match_code=args.match_code)

    elif args.mode == "infer":
        # Automatically enable save_embeddings when clustering or classify is requested
        save_emb = args.save_embeddings or args.clustering or args.classify

        retrieval_kwargs = {k: v for k, v in {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "bm25_candidate_k": args.bm25_candidate_k,
            "use_hybrid": not args.no_hybrid,
            "rrf_k": args.rrf_k,
            "w_dense": args.w_dense,
            "w_bm25": args.w_bm25,
            "boost_cat": args.boost_cat,
            "boost_int": args.boost_int,
            "use_mmr": args.use_mmr,
            "mmr_lambda": args.mmr_lambda,
            "max_workers": args.max_workers,
            "exclude_self": args.exclude_self,
            "save_embeddings": save_emb,
            "debug_timing": args.debug_timing,
            "show_progress": not args.no_progress,
        }.items() if v is not None}

        run_infer(
            cfg,
            incoming_path=args.path,
            run_topk=args.topk,
            run_classify_flag=args.classify,
            run_clustering=args.clustering,
            run_summarize=args.summarize,
            threshold=args.threshold,
            language=args.language,
            **retrieval_kwargs,
        )


if __name__ == "__main__":
    main()
