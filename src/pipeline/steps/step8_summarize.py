"""
Step 8 — Cluster Summarization via Nova Pro LLM.

Sends cluster representative texts and metadata distributions to Nova Pro,
parses the structured JSON response, and writes the cluster summaries parquet.
"""
import argparse
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import Config, load_config
from helpers.modelops_llm import build_cluster_prompt, call_nova_pro, safe_json_extract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cluster prompt hash (step-local utility)
# ---------------------------------------------------------------------------

def _hash_prompt(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-cluster worker
# ---------------------------------------------------------------------------

def _process_cluster(row: pd.Series, cfg: Config, max_reps_sent: int, truncate_chars: int, language: str = "fr") -> Dict[str, Any]:
    cluster_id = int(row["cluster_id"])
    cluster_size = int(row["cluster_size"])
    coherence = row["coherence_mean_cosine_to_medoid"]
    coherence = float(coherence) if not pd.isna(coherence) else float("nan")

    try:
        meta_dist = json.loads(row["meta_distributions_json"]) if isinstance(row["meta_distributions_json"], str) else {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("cluster=%d: failed to parse meta_distributions_json: %s", cluster_id, exc)
        meta_dist = {}

    medoid_ref = str(row["medoid_reference"])
    rep_refs = [x for x in str(row["rep_references"]).split(";") if x.strip()]
    rep_refs = [medoid_ref] + [x for x in rep_refs if x != medoid_ref]

    try:
        rep_texts = json.loads(row["rep_texts"]) if isinstance(row["rep_texts"], str) else []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("cluster=%d: failed to parse rep_texts: %s", cluster_id, exc)
        rep_texts = []

    reps_pairs: List[Tuple[str, str]] = []
    for i, ref in enumerate(rep_refs[:max_reps_sent]):
        txt = str(rep_texts[i]) if i < len(rep_texts) else ""
        if truncate_chars and len(txt) > truncate_chars:
            txt = txt[:truncate_chars] + "..."
        reps_pairs.append((ref, txt))

    system, user_prompt = build_cluster_prompt(cluster_id, cluster_size, coherence, meta_dist, reps_pairs, language=language)
    prompt_hash = _hash_prompt(system + "\n" + user_prompt)

    llm = cfg.llm
    try:
        gen_text, _ = call_nova_pro(
            user_text=user_prompt, system_text=system,
            model_id=llm.model_id, max_new_tokens=llm.max_new_tokens,
            temperature=llm.temperature, timeout=llm.timeout, retries=llm.retries,
        )
        parsed = safe_json_extract(gen_text)
    except Exception as e:
        gen_text = ""
        parsed = {"label": "", "summary": "", "llm_confidence": None, "outliers_or_notes": f"LLM error: {e}"}

    label = str(parsed.get("label", "")).strip()[:120]
    summary = str(parsed.get("summary", "")).strip()
    routing = str(parsed.get("routing_suggestion", "")).strip()
    notes = str(parsed.get("outliers_or_notes", "")).strip()
    keywords = parsed.get("keywords", [])
    common_systems = parsed.get("common_systems", [])
    common_actions = parsed.get("common_actions", [])

    llm_conf = parsed.get("llm_confidence", None)
    try:
        llm_conf = min(1.0, max(0.0, float(llm_conf)))
    except (TypeError, ValueError):
        llm_conf = float("nan")

    coh01 = min(1.0, max(0.0, (coherence + 1.0) / 2.0)) if not pd.isna(coherence) else float("nan")
    if not pd.isna(llm_conf) and not pd.isna(coh01):
        combined = 0.6 * llm_conf + 0.4 * coh01
    elif not pd.isna(llm_conf):
        combined = llm_conf
    elif not pd.isna(coh01):
        combined = coh01
    else:
        combined = float("nan")

    logger.info(
        "cluster=%d size=%d label='%s' llm_conf=%.2f combined=%.2f",
        cluster_id, cluster_size, label,
        llm_conf if not pd.isna(llm_conf) else 0.0,
        combined if not pd.isna(combined) else 0.0,
    )

    return {
        "cluster_id": cluster_id, "cluster_size": cluster_size, "label": label, "summary": summary,
        "keywords_json": json.dumps(keywords, ensure_ascii=False),
        "common_systems_json": json.dumps(common_systems, ensure_ascii=False),
        "common_actions_json": json.dumps(common_actions, ensure_ascii=False),
        "routing_suggestion": routing, "outliers_or_notes": notes,
        "llm_confidence": llm_conf, "coherence_mean_cosine_to_medoid": coherence,
        "combined_confidence": combined, "llm_model_id": llm.model_id,
        "llm_prompt_hash": prompt_hash, "llm_raw_snippet": gen_text[:800],
    }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_summarize(
    cfg: Config,
    reps_df: Optional[pd.DataFrame] = None,
    artifacts_dir: Optional[str] = None,
    max_workers: Optional[int] = None,
    max_reps_sent: Optional[int] = None,
    truncate_chars: Optional[int] = None,
    output_path: Optional[str] = None,
    language: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates LLM summaries for each cluster using Nova Pro.

    Args:
        cfg:            Global configuration object.
        reps_df:        Cluster representatives DataFrame. Loaded from disk if None.
        artifacts_dir:  Overrides cfg.paths.artifacts_dir.
        max_workers:    Concurrent LLM API threads (default: cfg.llm.max_workers).
        max_reps_sent:  Max representative texts per prompt (default: cfg.llm.max_reps_sent).
        truncate_chars: Character limit per representative text (default: cfg.llm.truncate_chars).
        output_path:    Override default output path.

    Returns:
        Cluster summaries DataFrame.

    Side effects:
        Writes:
          - <artifacts_dir>/datasets/cluster_summaries.parquet
    """
    if artifacts_dir is None:
        artifacts_dir = cfg.paths.artifacts_dir
    if max_workers is None:
        max_workers = cfg.llm.max_workers
    if max_reps_sent is None:
        max_reps_sent = cfg.llm.max_reps_sent
    if truncate_chars is None:
        truncate_chars = cfg.llm.truncate_chars
    if language is None:
        language = cfg.llm.language

    ds_dir = os.path.join(artifacts_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(ds_dir, "cluster_summaries.parquet")

    if reps_df is None:
        reps_path = os.path.join(ds_dir, "cluster_representatives.parquet")
        if not os.path.exists(reps_path):
            raise FileNotFoundError(
                f"Missing cluster representatives parquet: {reps_path}. "
                "Ensure step 7 (clustering) completed successfully."
            )
        logger.info("Step 8 - loading representatives: %s", reps_path)
        reps_df = pd.read_parquet(reps_path)

    required = {"cluster_id", "cluster_size", "medoid_reference", "rep_references", "rep_texts", "meta_distributions_json", "coherence_mean_cosine_to_medoid"}
    missing = required - set(reps_df.columns)
    if missing:
        raise ValueError(f"Missing columns in representatives parquet: {missing}")

    logger.info("Step 8 - summarizing %d clusters with %d workers (language=%s)...", len(reps_df), max_workers, language)
    rows = []
    n_failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cid = {
            executor.submit(_process_cluster, row, cfg, max_reps_sent, truncate_chars, language): row["cluster_id"]
            for _, row in reps_df.iterrows()
        }
        for future in as_completed(future_to_cid):
            cid = future_to_cid[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                logger.error("Cluster %d failed: %s", cid, exc)
                n_failed += 1

    if n_failed > 0:
        logger.warning(
            "%d / %d clusters failed LLM summarization and are absent from the output.",
            n_failed, len(reps_df),
        )

    out_df = (
        pd.DataFrame(rows)
        .sort_values(["cluster_size", "cluster_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out_df.to_parquet(output_path, index=False)
    logger.info("Summaries saved: %s | %d clusters", output_path, len(out_df))

    return out_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 8 - LLM cluster summarization.")
    parser.add_argument("--artifacts_dir", type=str, default=None)
    parser.add_argument("--cluster_reps_parquet", type=str, default="")
    parser.add_argument("--out_summaries_parquet", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--max_reps_sent", type=int, default=None)
    parser.add_argument("--truncate_chars", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    reps_df = pd.read_parquet(args.cluster_reps_parquet) if args.cluster_reps_parquet else None
    run_summarize(
        cfg, reps_df=reps_df,
        artifacts_dir=args.artifacts_dir or cfg.paths.artifacts_dir,
        max_workers=args.max_workers or cfg.llm.max_workers,
        max_reps_sent=args.max_reps_sent or cfg.llm.max_reps_sent,
        truncate_chars=args.truncate_chars or cfg.llm.truncate_chars,
        output_path=args.out_summaries_parquet or None,
    )
