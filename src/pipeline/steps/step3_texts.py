"""
Step 3 — Corpus Text Preparation.

Merges the semantic and lexical parquet files, constructs dense_text (for
embedding) and lex_text (for BM25), and writes the final vectorization-ready
parquet.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import Config, load_config
from helpers.du_engine import build_dense_text, build_lex_text, sanitize_text

logger = logging.getLogger(__name__)

_COL_ALIASES = {
    "Reason_Code": "Reason", "Reason_Description": "Reason",
    "Action_Code": "Action", "Action_Description": "Action",
    "Narrative": "Text",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: _COL_ALIASES[c] for c in df.columns if c in _COL_ALIASES}
    if not rename:
        return df
    # Drop target columns that already exist to prevent duplicate column names
    # after rename (e.g. both "Narrative" and "Text" present simultaneously).
    targets_already_present = {tgt for tgt in rename.values() if tgt in df.columns}
    if targets_already_present:
        sources_to_drop = [src for src, tgt in rename.items() if tgt in targets_already_present]
        logger.warning(
            "Dropping source columns %s before rename — target column(s) %s already exist.",
            sources_to_drop,
            sorted(targets_already_present),
        )
        df = df.drop(columns=sources_to_drop)
        rename = {c: _COL_ALIASES[c] for c in df.columns if c in _COL_ALIASES}
    return df.rename(columns=rename) if rename else df


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_prepare_texts(
    cfg: Config,
    df_semantic: Optional[pd.DataFrame] = None,
    df_lexical: Optional[pd.DataFrame] = None,
    artifacts_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Builds dense_text and lex_text columns and writes the vectorization-ready parquet.

    Args:
        cfg:           Global configuration object.
        df_semantic:   Semantic DataFrame (from step 2). Loaded from disk if None.
        df_lexical:    Lexical DataFrame (from step 2). Loaded from disk if None.
        artifacts_dir: Overrides cfg.paths.artifacts_dir.

    Returns:
        Combined DataFrame with dense_text and lex_text columns.

    Side effects:
        Writes:
          - <artifacts_dir>/datasets/ready_for_vectorization_DU.parquet
          - <artifacts_dir>/stats/text_builder_report.json
    """
    if artifacts_dir is None:
        artifacts_dir = cfg.paths.artifacts_dir

    datasets_dir = os.path.join(artifacts_dir, "datasets")
    stats_dir = os.path.join(artifacts_dir, "stats")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    sem_path = os.path.join(datasets_dir, "final_DU_semantic.parquet")
    lex_path = os.path.join(datasets_dir, "final_DU_lexical.parquet")
    out_path = os.path.join(datasets_dir, "ready_for_vectorization_DU.parquet")
    report_path = os.path.join(stats_dir, "text_builder_report.json")

    if df_semantic is None:
        if not os.path.exists(sem_path):
            raise FileNotFoundError(f"Missing semantic parquet: {sem_path}")
        df_semantic = pd.read_parquet(sem_path)
        logger.info("Step 3 - loaded semantic parquet: %s", sem_path)

    if df_lexical is None:
        if not os.path.exists(lex_path):
            raise FileNotFoundError(f"Missing lexical parquet: {lex_path}")
        df_lexical = pd.read_parquet(lex_path)
        logger.info("Step 3 - loaded lexical parquet: %s", lex_path)

    df_sem = _normalize_columns(df_semantic.copy())
    df_lex = _normalize_columns(df_lexical.copy())

    # Validate that both DataFrames carry the join key before proceeding.
    for name, df in (("semantic", df_sem), ("lexical", df_lex)):
        if "Reference" not in df.columns:
            raise ValueError(
                f"Missing 'Reference' column in {name} DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    df_sem["Reference"] = df_sem["Reference"].astype(str).str.strip()
    df_lex["Reference"] = df_lex["Reference"].astype(str).str.strip()

    # Warn early about duplicates — reindex will silently duplicate lex rows otherwise.
    dup_refs = df_sem["Reference"][df_sem["Reference"].duplicated()].unique()
    if len(dup_refs) > 0:
        logger.warning(
            "Duplicate References in semantic DataFrame (%d values); "
            "lex_text alignment may be incorrect: %s",
            len(dup_refs),
            dup_refs[:10].tolist(),
        )

    df_lex = df_lex.set_index("Reference").reindex(df_sem["Reference"]).reset_index()

    logger.info("Building dense_text...")
    df_sem["dense_text"] = df_sem.apply(
        lambda r: build_dense_text(r, max_ids_to_keep=cfg.processing.max_ids_to_keep), axis=1
    )

    logger.info("Building lex_text...")
    # Use .values to avoid pandas index-label alignment: df_lex was reset to
    # RangeIndex(0, N) by reindex+reset_index, while df_sem may carry a
    # non-contiguous index from upstream — aligning by label would produce NaN.
    df_sem["lex_text"] = df_lex.apply(build_lex_text, axis=1).values

    # Fallback for empty dense_text
    empty_dense = df_sem["dense_text"].fillna("").str.strip() == ""
    if empty_dense.any():
        if "Title" in df_sem.columns:
            title_fallback = (
                df_sem.loc[empty_dense, "Title"]
                .fillna("")
                .astype(str)
                .map(sanitize_text)
            )
        else:
            title_fallback = pd.Series("", index=df_sem.loc[empty_dense].index)
        df_sem.loc[empty_dense, "dense_text"] = title_fallback
        still_empty = df_sem["dense_text"].fillna("").str.strip() == ""
        df_sem.loc[still_empty, "dense_text"] = "Document technique"

    # Warn if lex_text is empty after build — these rows will have zero BM25 recall.
    empty_lex = df_sem["lex_text"].fillna("").str.strip() == ""
    if empty_lex.any():
        logger.warning(
            "%d rows have empty lex_text after build_lex_text; BM25 recall will be zero for these.",
            int(empty_lex.sum()),
        )

    meta_cols = [
        "Reference", "Title", "Text", "Intent", "Category", "System", "Reason", "Action", "Rubric",
        "Header_Keyword", "Body_Keyword", "Side", "FAL", "feature_status", "has_match_code_header", "Creation_Date",
    ]
    meta_cols = [c for c in meta_cols if c in df_sem.columns]
    out_df = df_sem[meta_cols + ["dense_text", "lex_text"]].copy()
    out_df.to_parquet(out_path, index=False)
    logger.info("Saved vectorization parquet: %s | rows=%d cols=%d", out_path, len(out_df), len(out_df.columns))

    # Quality report
    dense_len = out_df["dense_text"].astype(str).str.len()
    lex_len = out_df["lex_text"].astype(str).str.len()
    report = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(out_df)),
        "empty_dense_text": int((out_df["dense_text"].fillna("").str.strip() == "").sum()),
        "empty_lex_text": int((out_df["lex_text"].fillna("").str.strip() == "").sum()),
        "dense_len_chars": {"p50": int(dense_len.median()), "p90": int(dense_len.quantile(0.9)), "max": int(dense_len.max())},
        "lex_len_chars": {"p50": int(lex_len.median()), "p90": int(lex_len.quantile(0.9)), "max": int(lex_len.max())},
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Text builder report saved: %s", report_path)

    return out_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3 - Build dense_text and lex_text for vectorization.")
    parser.add_argument("--artifacts_dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    run_prepare_texts(cfg, artifacts_dir=args.artifacts_dir or cfg.paths.artifacts_dir)
