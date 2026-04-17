"""
Pipeline runner — orchestrates the train / retrain / infer modes.

The runner is the only component that:
  - reads and writes manifest.json
  - resolves version directories
  - chains step functions together
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional

from config import Config
from pipeline.steps.step1_process import run_process_data
from pipeline.steps.step2_features import run_stage_features
from pipeline.steps.step3_texts import run_prepare_texts
from pipeline.steps.step4_embeddings import run_generate_embeddings
from pipeline.steps.step5_indexes import run_build_indexes, run_merge_indexes
from pipeline.steps.step6_retrieval import run_retrieval
from pipeline.steps.step7_clustering import run_clustering as run_step7_clustering
from pipeline.steps.step8_summarize import run_summarize as run_step8_summarize
from pipeline.steps.step9_global_clustering import run_global_clustering
from pipeline.steps.step10_classify import run_classify as run_step10_classify
from pipeline.steps.step11_export import run_export as run_step11_export

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "manifest.json"


# ---------------------------------------------------------------------------
# File logging helper
# ---------------------------------------------------------------------------

def _setup_file_logging(version_dir: str) -> None:
    """
    Adds a rotating FileHandler that writes all log records to
    <version_dir>/logs/pipeline.log.  Safe to call multiple times —
    a handler for the same path is only added once.
    """
    log_dir = os.path.join(version_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "pipeline.log")

    root = logging.getLogger()
    # Avoid duplicate handlers if the function is somehow called twice
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_path):
            return

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)
    logger.info("File logging active: %s", log_path)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, _MANIFEST_FILE)


def _load_manifest(artifacts_dir: str) -> dict:
    path = _manifest_path(artifacts_dir)
    if not os.path.exists(path):
        return {"current": None, "versions": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(artifacts_dir: str, manifest: dict) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    path = _manifest_path(artifacts_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Manifest updated: %s", path)


def _next_version(manifest: dict) -> str:
    """Returns the next version label (v1, v2, …)."""
    existing = manifest.get("versions", {})
    if not existing:
        return "v1"
    last_n = max(int(k[1:]) for k in existing if k.startswith("v") and k[1:].isdigit())
    return f"v{last_n + 1}"


def _current_artifacts_dir(cfg: Config) -> str:
    """Returns the artifact directory for the currently active version."""
    manifest = _load_manifest(cfg.paths.artifacts_dir)
    current = manifest.get("current")
    if current:
        versions = manifest.get("versions", {})
        if current in versions:
            return versions[current].get("artifacts_dir", cfg.paths.artifacts_dir)
    return cfg.paths.artifacts_dir


# ---------------------------------------------------------------------------
# Train mode (steps 1–5, fresh build)
# ---------------------------------------------------------------------------

def run_train(cfg: Config, input_path: str, match_code: bool = False) -> None:
    """
    Full corpus build: runs steps 1–5 and registers a new versioned artifact set.

    Args:
        cfg:         Global configuration object.
        input_path:  Path to the raw CSV input.
        match_code:  Filter rows to those with a regex header.
    """
    artifacts_dir = cfg.paths.artifacts_dir
    manifest = _load_manifest(artifacts_dir)
    version = _next_version(manifest)
    version_dir = os.path.join(artifacts_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    _setup_file_logging(version_dir)
    logger.info("=== TRAIN [%s] ===", version)

    # All steps write into the versioned directory so each version is self-contained.
    df = run_process_data(cfg, input_path=input_path, artifacts_dir=version_dir, match_code=match_code)
    df_sem, df_lex = run_stage_features(cfg, df=df, artifacts_dir=version_dir)
    df_ready = run_prepare_texts(cfg, df_semantic=df_sem, df_lexical=df_lex, artifacts_dir=version_dir)
    df_emb = run_generate_embeddings(cfg, df=df_ready, artifacts_dir=version_dir)
    run_build_indexes(cfg, df_embeddings=df_emb, df_texts=df_ready, version_dir=version_dir)

    # Step 9: Global clustering on full corpus — produces class anchors (medoids)
    logger.info("--- Step 9: Global Clustering ---")
    run_global_clustering(cfg, version_dir=version_dir)

    # Register version
    manifest["current"] = version
    manifest.setdefault("versions", {})[version] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "train",
        "source": os.path.abspath(input_path),
        "artifacts_dir": version_dir,
        "doc_count": int(len(df_emb[df_emb["vector"].notna()])),
    }
    _save_manifest(artifacts_dir, manifest)
    logger.info("=== TRAIN DONE [%s] — %d docs indexed ===", version, manifest["versions"][version]["doc_count"])


# ---------------------------------------------------------------------------
# Retrain mode (steps 1–4 on new data, step 5 merge)
# ---------------------------------------------------------------------------

def run_retrain(cfg: Config, input_path: str, match_code: bool = False) -> None:
    """
    Incremental retrain: runs steps 1–4 on new data, merges into current FAISS/BM25,
    and registers a new versioned artifact set.

    Args:
        cfg:         Global configuration object.
        input_path:  Path to the new raw CSV input.
        match_code:  Filter rows to those with a regex header.
    """
    artifacts_dir = cfg.paths.artifacts_dir
    manifest = _load_manifest(artifacts_dir)
    current = manifest.get("current")
    if not current:
        logger.warning("No current version found in manifest — running full train instead.")
        run_train(cfg, input_path=input_path, match_code=match_code)
        return

    current_version_info = manifest["versions"][current]
    current_version_dir = current_version_info["artifacts_dir"]

    new_version = _next_version(manifest)
    new_version_dir = os.path.join(artifacts_dir, new_version)
    os.makedirs(new_version_dir, exist_ok=True)

    _setup_file_logging(new_version_dir)
    logger.info("=== RETRAIN [%s → %s] ===", current, new_version)

    # All steps write into the new versioned directory; the new version is self-contained.
    df = run_process_data(cfg, input_path=input_path, artifacts_dir=new_version_dir, match_code=match_code)
    df_sem, df_lex = run_stage_features(cfg, df=df, artifacts_dir=new_version_dir)
    df_ready = run_prepare_texts(cfg, df_semantic=df_sem, df_lexical=df_lex, artifacts_dir=new_version_dir)
    df_emb = run_generate_embeddings(cfg, df=df_ready, artifacts_dir=new_version_dir)

    # Step 5: merge new into existing version
    run_merge_indexes(
        cfg,
        df_new_embeddings=df_emb,
        df_new_texts=df_ready,
        current_version_dir=current_version_dir,
        new_version_dir=new_version_dir,
    )

    # Step 9: Re-run global clustering on merged corpus
    logger.info("--- Step 9: Global Clustering (post-merge) ---")
    run_global_clustering(cfg, version_dir=new_version_dir)

    current_count = manifest["versions"][current].get("doc_count", 0)
    new_count = int(len(df_emb[df_emb["vector"].notna()]))

    manifest["current"] = new_version
    manifest["versions"][new_version] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "retrain",
        "source": os.path.abspath(input_path),
        "artifacts_dir": new_version_dir,
        "doc_count": current_count + new_count,
        "merged_from": current,
        "new_docs_added": new_count,
    }
    _save_manifest(artifacts_dir, manifest)
    logger.info("=== RETRAIN DONE [%s] — total %d docs ===", new_version, manifest["versions"][new_version]["doc_count"])


# ---------------------------------------------------------------------------
# Infer mode (steps 6–8 with early-stop flags)
# ---------------------------------------------------------------------------

def run_infer(
    cfg: Config,
    incoming_path: str,
    run_topk: bool = True,
    run_classify_flag: bool = False,
    run_clustering: bool = False,
    run_summarize: bool = False,
    threshold: float = 0.75,
    language: Optional[str] = None,
    **retrieval_kwargs,
) -> None:
    """
    Runs the inference pipeline on a new incoming CSV.

    Args:
        cfg:               Global configuration object.
        incoming_path:     Path to the incoming queries CSV.
        run_topk:          Run step 6 (hybrid retrieval).
        run_classify_flag: Run step 10 (medoid-based classification).
        run_clustering:    Run step 7 (UMAP + HDBSCAN monthly clustering).
        run_summarize:     Run step 8 (LLM summarization).
        threshold:         Cosine similarity threshold for classification.
        **retrieval_kwargs: Additional keyword arguments forwarded to run_retrieval.
    """
    artifacts_dir = _current_artifacts_dir(cfg)
    _setup_file_logging(artifacts_dir)
    logger.info("=== INFER — active version dir: %s ===", artifacts_dir)

    ds_dir = os.path.join(artifacts_dir, "datasets")

    # Step A: Retrieval — force save_embeddings when classify or clustering needs vectors
    if run_topk:
        logger.info("--- Step A (6): Retrieval ---")
        retrieval_kwargs.setdefault("save_embeddings", run_classify_flag or run_clustering)
        run_retrieval(cfg, incoming_path=incoming_path, artifacts_dir=artifacts_dir, **retrieval_kwargs)

    # Step B: Medoid-based classification
    if run_classify_flag:
        logger.info("--- Step B (10): Medoid Classification (threshold=%.3f) ---", threshold)
        emb_path = os.path.join(ds_dir, "incoming_embeddings.parquet")
        run_step10_classify(cfg, incoming_embeddings_path=emb_path, artifacts_dir=artifacts_dir, threshold=threshold)

    # Step C: Monthly batch clustering + summarization
    if run_clustering:
        logger.info("--- Step C (7): Monthly Batch Clustering ---")
        emb_path = os.path.join(ds_dir, "incoming_embeddings.parquet")
        topk_path = os.path.join(ds_dir, "incoming_topk_results.parquet")
        run_step7_clustering(
            cfg,
            incoming_embeddings_path=emb_path,
            topk_parquet_path=topk_path,
            artifacts_dir=artifacts_dir,
            random_state=None,  # No determinism needed for monthly batches
            n_jobs=-1,          # Speed over reproducibility
        )

    if run_summarize:
        logger.info("--- Step C (8): Summarization ---")
        run_step8_summarize(cfg, artifacts_dir=artifacts_dir, language=language)

    # Step D: Consolidated export
    logger.info("--- Step D (11): Export ---")
    run_step11_export(cfg, artifacts_dir=artifacts_dir)

    logger.info("=== INFER DONE ===")
