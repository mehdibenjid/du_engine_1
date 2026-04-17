"""
Step 1 — Data Ingestion & Normalization.

Loads a raw DU CSV, filters by intervention type, runs language detection,
deduplicates by text content, and writes the processed dataset to disk.
"""
import argparse
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import Config, load_config
from helpers.processing import (
    add_match_code_flag,
    analyze_text_lengths,
    deduplicate_text_content,
    normalize_raw_columns,
    process_du_data,
    run_language_pipeline,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_process_data(
    cfg: Config,
    input_path: str,
    artifacts_dir: Optional[str] = None,
    match_code: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Ingests, normalizes, analyses, and deduplicates raw DU data.

    Args:
        cfg:           Global configuration object.
        input_path:    Path to the raw CSV file.
        artifacts_dir: Base artifacts directory (overrides cfg.paths.artifacts_dir).
        match_code:    If True, keep only rows with a regex header. Overrides config.

    Returns:
        Processed and deduplicated DataFrame.

    Side effects:
        Writes:
          - <processed_dir>/processed_DU.csv
          - <artifacts_dir>/datasets/DU_text_duplicates.csv
          - <artifacts_dir>/stats/corpus_manifest.json
    """
    if artifacts_dir is None:
        artifacts_dir = cfg.paths.artifacts_dir
    if match_code is None:
        match_code = cfg.processing.match_code

    processed_dir = cfg.paths.processed_dir
    audit_dir = os.path.join(artifacts_dir, "datasets")
    stats_dir = os.path.join(artifacts_dir, "stats")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    output_path = os.path.join(processed_dir, "processed_DU.csv")
    audit_path = os.path.join(audit_dir, "DU_text_duplicates.csv")
    manifest_path = os.path.join(stats_dir, "corpus_manifest.json")

    logger.info("Step 1 — processing raw data from: %s", os.path.abspath(input_path))

    manifest = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "input_path": os.path.abspath(input_path),
        "type_name": cfg.processing.type_name,
        "match_code_enabled": match_code,
        "counts": {},
    }

    df = process_du_data(input_path, type_name=cfg.processing.type_name, separator=cfg.processing.csv_separator)
    df = normalize_raw_columns(df)
    df = add_match_code_flag(df)
    manifest["counts"]["after_process_du_data"] = int(len(df))

    if match_code:
        logger.info("Match-code filter active.")
        before = len(df)
        df = df[df["has_match_code_header"]].copy()
        after = len(df)
        manifest["counts"]["after_match_code_filter"] = int(after)
        manifest["counts"]["dropped_by_match_code"] = int(before - after)
        logger.info("Match-code filter: %d → %d rows", before, after)
    else:
        manifest["counts"]["after_match_code_filter"] = int(len(df))
        manifest["counts"]["dropped_by_match_code"] = 0

    df = analyze_text_lengths(df, columns=["Title", "Text"])
    manifest["counts"]["after_length_analysis"] = int(len(df))

    df = run_language_pipeline(df, columns=["Title", "Text"])
    manifest["counts"]["after_language_pipeline"] = int(len(df))

    before_dedup = len(df)
    df, _ = deduplicate_text_content(df, text_col="Text", ref_col="Reference", output_audit_path=audit_path)
    manifest["counts"]["after_deduplication"] = int(len(df))
    manifest["counts"]["dropped_by_deduplication"] = int(before_dedup - len(df))

    # Write manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Manifest saved to: %s", manifest_path)

    # Write processed CSV
    df.to_csv(output_path, sep=cfg.processing.csv_separator or ";", index=False)
    logger.info("Processed data saved to: %s | rows=%d", output_path, len(df))

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1 — DU data ingestion and normalization.")
    parser.add_argument("--path", type=str, default=None, help="Path to the raw input CSV.")
    parser.add_argument("--artifacts_dir", type=str, default=None, help="Base artifacts directory.")
    parser.add_argument("--match_code", action="store_true", help="Keep only rows with a regex header.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    cfg = load_config()

    run_process_data(
        cfg,
        input_path=args.path or os.path.join(cfg.paths.raw_dir, "input.csv"),
        artifacts_dir=args.artifacts_dir or cfg.paths.artifacts_dir,
        match_code=args.match_code or None,
    )
