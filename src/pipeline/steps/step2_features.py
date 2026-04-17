"""
Step 2 — Feature Extraction & Semantic/Lexical Split.

Reads the processed CSV, runs regex-based feature extraction, applies semantic
cleaning, then writes two parquet files: one for dense (semantic) search and
one for sparse (lexical) search.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from config import Config, load_config
from helpers.cleaning import clean_extracted_features
from helpers.du_engine import build_lex_id, build_lex_text_series
from helpers.patterns import compute_has_header, extract_features_df
from helpers.processing import detect_csv_separator

logger = logging.getLogger(__name__)

DEFAULT_STR = "Non spécifié"


# ---------------------------------------------------------------------------
# Internal helpers (step-specific)
# ---------------------------------------------------------------------------

def _compute_coverage(df: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in cols:
        if c not in df.columns:
            continue
        non_empty = (df[c].fillna("").astype(str).str.strip() != "") & (df[c] != DEFAULT_STR)
        out[c] = round(float(non_empty.mean() * 100.0), 2)
    return out


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_stage_features(
    cfg: Config,
    df: Optional[pd.DataFrame] = None,
    artifacts_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts structured features from DU text, applies semantic cleaning, and
    splits the data into semantic and lexical parquet datasets.

    Args:
        cfg:           Global configuration object.
        df:            Pre-loaded processed DataFrame. If None, reads from disk.
        artifacts_dir: Overrides cfg.paths.artifacts_dir for output paths.

    Returns:
        (df_semantic, df_lexical) — both as DataFrames.

    Side effects:
        Writes:
          - <artifacts_dir>/datasets/final_DU_semantic.parquet
          - <artifacts_dir>/datasets/final_DU_lexical.parquet
          - <artifacts_dir>/stats/feature_frequencies.json
          - <artifacts_dir>/stats/feature_coverage.json
          - <transformed_dir>/transformed_DU.csv
    """
    if artifacts_dir is None:
        artifacts_dir = cfg.paths.artifacts_dir

    datasets_dir = os.path.join(artifacts_dir, "datasets")
    stats_dir = os.path.join(artifacts_dir, "stats")
    transformed_dir = cfg.paths.transformed_dir
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    os.makedirs(transformed_dir, exist_ok=True)

    # Load from disk if not passed in
    if df is None:
        input_path = os.path.join(cfg.paths.processed_dir, "processed_DU.csv")
        logger.info("Step 2 — loading from: %s", input_path)
        _sep = cfg.processing.csv_separator or detect_csv_separator(input_path)
        df = pd.read_csv(input_path, sep=_sep, low_memory=False)
    else:
        logger.info("Step 2 — using in-memory DataFrame (%d rows)", len(df))

    ref_col, title_col, text_col = "Reference", "Title", "Text"
    has_header_col = "has_match_code_header"

    for c in [ref_col, title_col, text_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Available: {list(df.columns)}")

    if has_header_col not in df.columns:
        df[has_header_col] = df[text_col].fillna("").astype(str).apply(compute_has_header)

    df[ref_col] = df[ref_col].astype(str).str.strip()

    logger.info("Extracting features from %d rows...", len(df))
    df = extract_features_df(df, text_col, has_header_col)

    logger.info("Cleaning features...")
    df = clean_extracted_features(df)

    for col in ["Intent", "Category", "System", "Reason_Code", "Action_Description", "Rubric"]:
        if col in df.columns:
            df[col] = df[col].fillna(DEFAULT_STR).replace("", DEFAULT_STR)

    # Build semantic parquet
    semantic_priority = [
        "Reference", "Title", "Narrative",
        "Intent", "Reason_Code", "Action_Description", "System", "Category",
        "Header_Keyword", "Body_Keyword", "Side",
        "Rubric", "Plug_Action", "Cabling_Reason", "Circuit", "Tank_Drained",
        "Equip_Downloadable", "FAL",
        "feature_status", "has_match_code_header", "Creation_Date",
        "Doc_Type", "Doc_Ref", "Header_PN", "Header_FIN", "Body_PN", "Body_FIN", "Cable_Number", "Log_Code",
    ]
    sem_cols = [c for c in semantic_priority if c in df.columns]
    df_sem = df[sem_cols].copy().rename(columns={"Narrative": "Text", "Reason_Code": "Reason", "Action_Description": "Action"})

    sem_path = os.path.join(datasets_dir, "final_DU_semantic.parquet")
    df_sem.to_parquet(sem_path, index=False)
    logger.info("Saved semantic parquet: %s | rows=%d cols=%d", sem_path, len(df_sem), len(df_sem.columns))

    # Build lexical parquet
    candidate_id_cols = ["Doc_Type", "Doc_Ref", "Header_PN", "Header_FIN", "Body_PN", "Body_FIN", "Cable_Number"]
    id_cols = [c for c in candidate_id_cols if c in df.columns]
    lexical_base_cols = ["Title", "Narrative", "Header_Keyword", "Body_Keyword", "System", "Category", "Intent", "Reason_Code", "Action_Description", "Rubric", "FAL"]
    base_cols = [c for c in lexical_base_cols if c in df.columns]

    df_lex = pd.DataFrame()
    df_lex["Reference"] = df[ref_col].astype(str)
    # Use explicit Series for the empty case to avoid relying on scalar broadcasting.
    df_lex["Lex_ID"] = (
        df.apply(lambda r: build_lex_id(r, id_cols=id_cols, max_tokens=15), axis=1)
        if id_cols
        else pd.Series("", index=df.index)
    )
    df_lex["Lex_Text"] = build_lex_text_series(df, id_cols=id_cols, base_cols=base_cols)
    for c in id_cols:
        df_lex[c] = df[c].fillna("").astype(str)

    lex_path = os.path.join(datasets_dir, "final_DU_lexical.parquet")
    df_lex.to_parquet(lex_path, index=False)
    logger.info("Saved lexical parquet: %s | rows=%d cols=%d", lex_path, len(df_lex), len(df_lex.columns))

    # Stats
    cols_to_stat = ["System", "Category", "Intent", "Reason_Code", "Action_Description", "Rubric"]
    stats_freq: Dict[str, Any] = {}
    for col in cols_to_stat:
        if col not in df.columns:
            continue
        vc = df[col].fillna(DEFAULT_STR).value_counts()
        stats_freq[col] = {"unique_values": int(len(vc)), "doc_count": int(len(df)), "frequencies": vc.to_dict()}

    with open(os.path.join(stats_dir, "feature_frequencies.json"), "w", encoding="utf-8") as f:
        json.dump(stats_freq, f, indent=2, ensure_ascii=False)

    coverage = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "feature_status_counts": df["feature_status"].value_counts(dropna=False).to_dict() if "feature_status" in df.columns else {},
        "coverage_pct": _compute_coverage(df, ["System", "Category", "Intent", "Reason_Code", "Action_Description", "Rubric", "Header_Keyword", "Body_Keyword", "Narrative"]),
        "match_code_header_rate_pct": round(float(df[has_header_col].mean() * 100.0), 2) if has_header_col in df.columns else None,
    }
    with open(os.path.join(stats_dir, "feature_coverage.json"), "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)

    # Debug dump
    df.to_csv(os.path.join(transformed_dir, "transformed_DU.csv"), sep=cfg.processing.csv_separator or ";", index=False)

    return df_sem, df_lex


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 — Feature extraction and semantic/lexical split.")
    parser.add_argument("--artifacts_dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    run_stage_features(cfg, artifacts_dir=args.artifacts_dir or cfg.paths.artifacts_dir)
