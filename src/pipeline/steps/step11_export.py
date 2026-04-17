"""
Step 11 — Consolidated Export.

Merges retrieval results, medoid-based classification, and monthly
batch clustering into a single output Parquet + CSV.
"""
import logging
import os
from typing import Optional

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def run_export(
    cfg: Config,
    artifacts_dir: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Merges all inference outputs into one consolidated file.

    Loads (where available):
        - incoming_topk_results.parquet   (step 6 — required)
        - incoming_classification.parquet (step 10 — optional)
        - incoming_clusters.parquet       (step 7  — optional)

    The join key is ``incoming_reference`` in all three datasets.
    Classification and clustering produce one row per query; topk produces
    ``top_k`` rows per query.  The join broadcasts the single-row columns
    across all topk rows for each query.

    Args:
        cfg:           Global config.
        artifacts_dir: Active version directory.
        output_path:   Override base output path (without extension).

    Returns:
        Consolidated DataFrame.

    Side effects:
        Writes:
          - <artifacts_dir>/datasets/consolidated_results.parquet
          - <artifacts_dir>/datasets/consolidated_results.csv
    """
    ds_dir = os.path.join(artifacts_dir, "datasets")

    # --- Topk (required) ---
    topk_path = os.path.join(ds_dir, "incoming_topk_results.parquet")
    if not os.path.exists(topk_path):
        raise FileNotFoundError(
            f"TopK results not found: {topk_path}. "
            "Ensure step 6 (retrieval) completed successfully."
        )
    df = pd.read_parquet(topk_path)
    df["incoming_reference"] = df["incoming_reference"].astype(str).str.strip()
    logger.info("Step 11 — loaded topk: %s | %d rows", topk_path, len(df))

    # --- Classification (optional) ---
    classify_path = os.path.join(ds_dir, "incoming_classification.parquet")
    if os.path.exists(classify_path):
        clf_df = pd.read_parquet(classify_path)
        clf_df["incoming_reference"] = clf_df["incoming_reference"].astype(str).str.strip()
        df = df.merge(clf_df, on="incoming_reference", how="left")
        logger.info("Merged classification: %d unique queries classified", len(clf_df))
    else:
        logger.info("No classification results found — skipping.")

    # --- Monthly clustering (optional) ---
    clusters_path = os.path.join(ds_dir, "incoming_clusters.parquet")
    if os.path.exists(clusters_path):
        cl_df = pd.read_parquet(clusters_path)
        # Identify the ID column — step 7 uses "incoming_reference"
        id_col = "incoming_reference"
        if id_col not in cl_df.columns:
            # Fallback: check for any column ending with "reference"
            ref_candidates = [c for c in cl_df.columns if "reference" in c.lower()]
            if ref_candidates:
                id_col = ref_candidates[0]
            else:
                logger.warning(
                    "Cannot find reference column in clusters parquet — skipping. "
                    "Columns: %s", list(cl_df.columns),
                )
                id_col = None

        if id_col is not None:
            cl_df[id_col] = cl_df[id_col].astype(str).str.strip()
            # Rename cluster_id -> monthly_cluster_id to avoid collision with
            # the existing_class_id from classification.
            rename_map = {"cluster_id": "monthly_cluster_id"}
            cl_df = cl_df.rename(columns=rename_map)

            merge_cols = [id_col, "monthly_cluster_id"]
            for opt_col in ["umap_x", "umap_y", "dist_to_medoid", "is_medoid"]:
                if opt_col in cl_df.columns:
                    merge_cols.append(opt_col)

            cl_df = cl_df[merge_cols].copy()
            if id_col != "incoming_reference":
                cl_df = cl_df.rename(columns={id_col: "incoming_reference"})

            df = df.merge(cl_df, on="incoming_reference", how="left")
            logger.info("Merged monthly clustering: %d unique queries clustered", len(cl_df))
    else:
        logger.info("No monthly clustering results found — skipping.")

    # --- Write outputs ---
    if output_path is None:
        base_path = os.path.join(ds_dir, "consolidated_results")
    else:
        base_path = output_path.rsplit(".", 1)[0] if "." in output_path else output_path

    parquet_out = base_path + ".parquet"
    csv_out = base_path + ".csv"

    df.to_parquet(parquet_out, index=False)
    df.to_csv(csv_out, index=False)
    logger.info("Consolidated export: %s | %d rows, %d cols", parquet_out, len(df), len(df.columns))
    logger.info("CSV mirror: %s", csv_out)

    return df
