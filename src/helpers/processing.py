"""
Core data ingestion, text analysis, language detection, and deduplication.
"""
import csv
import logging
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from lingua import Language, LanguageDetectorBuilder

logger = logging.getLogger(__name__)


def detect_csv_separator(file_path: str, fallback: str = ";") -> str:
    """
    Sniffs the CSV delimiter from the first 4 KB of the file.

    Args:
        file_path: Path to the CSV file.
        fallback:  Delimiter to return if sniffing fails.

    Returns:
        Detected delimiter character, or fallback on failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        logger.info("Detected CSV separator %r for: %s", dialect.delimiter, file_path)
        return dialect.delimiter
    except Exception:
        logger.warning("Could not detect separator for %s — using fallback %r.", file_path, fallback)
        return fallback


def process_du_data(file_path: str, type_name: str = "Intervention", separator: str = ";") -> pd.DataFrame:
    """
    Loads raw DU data from a CSV, filters by type, and returns a cleaned DataFrame.

    Args:
        file_path:  Path to the source CSV file.
        type_name:  Value to filter on (ANSWER_TYPE_NAME / TYPE_NAME column).
        separator:  CSV delimiter.

    Returns:
        Filtered and cleaned DataFrame.
    """
    abs_path = os.path.abspath(file_path)
    if not separator:
        separator = detect_csv_separator(abs_path)

    def _read_csv(path: str, sep: str) -> pd.DataFrame:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return pd.read_csv(f, sep=sep, dtype=str, keep_default_na=True)
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed for %s — retrying with Latin-1.", path)
            with open(path, "r", encoding="latin-1") as f:
                return pd.read_csv(f, sep=sep, dtype=str, keep_default_na=True)

    preferred_columns = ["REVISED_TYPE_NAME", "ANSWER_TYPE_NAME", "TYPE_NAME"]

    df = _read_csv(abs_path, separator)
    target_col = next((col for col in preferred_columns if col in df.columns), None)
    if target_col is None and separator != ",":
        logger.warning(
            "Expected columns not found with separator %r — retrying with ','.", separator
        )
        df = _read_csv(abs_path, ",")
        target_col = next((col for col in preferred_columns if col in df.columns), None)
    if target_col is None:
        raise KeyError(f"None of the expected columns {preferred_columns} exist in the DataFrame.")

    initial_count = len(df)
    stats = [{"Step": "Initial Load", "Count": initial_count, "Dropped": 0}]

    def update_stats(label: str, current_df: pd.DataFrame) -> None:
        dropped = stats[-1]["Count"] - len(current_df)
        stats.append({"Step": label, "Count": len(current_df), "Dropped": dropped})

    df = df[df[target_col] == type_name].copy()
    update_stats(f"Filter: {type_name} Only", df)

    # Keep only relevant columns
    cols_to_keep = ["REFERENCE", "ISSUE", "TITLE", "TEXT", "CREATION_DATE"]
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]

    mask_valid_text = df["TEXT"].notna() & (df["TEXT"].str.strip() != "")
    df = df[mask_valid_text].copy()
    update_stats("Filter: Non-empty TEXT", df)

    # Remove multi-issue DUs
    ref_counts = df["REFERENCE"].value_counts()
    single_issue_refs = ref_counts[ref_counts == 1].index
    df = df[df["REFERENCE"].isin(single_issue_refs)].copy()
    update_stats("Filter: Single-issue Refs", df)

    # Keep only ISSUE == "1"
    if "ISSUE" in df.columns:
        df = df[df["ISSUE"] == "1"].copy()
        df = df.drop(columns=["ISSUE"])
        update_stats("Filter: Issue 1 Only", df)

    df["TEXT"] = df["TEXT"].str.strip()
    df["TITLE"] = df["TITLE"].str.strip()

    stats_df = pd.DataFrame(stats)
    stats_df["Retention %"] = (stats_df["Count"] / initial_count * 100).round(2)
    logger.info("Processing summary (%s): %s", type_name, stats_df.to_dict(orient='records'))
    logger.info("Final retention rate: %s%%", stats_df.iloc[-1]["Retention %"])

    return df


def analyze_text_lengths(df: pd.DataFrame, columns: list = None, bins: int = 30, figsize: tuple = (8, 4)) -> pd.DataFrame:
    """
    Computes string lengths for specified columns and logs distribution statistics.

    Args:
        df:       Input DataFrame.
        columns:  Columns to analyse. Defaults to ["TITLE", "TEXT"].
        bins:     Histogram bins (unused in production, kept for notebook use).
        figsize:  Figure size (unused in production).

    Returns:
        Copy of the DataFrame with new '<COL>_LEN' columns added.
    """
    if columns is None:
        columns = ["TITLE", "TEXT"]

    df_processed = df.copy()
    for col in columns:
        if col in df_processed.columns:
            clean_series = df_processed[col].fillna("").astype(str)
            len_col_name = f"{col}_LEN"
            df_processed.loc[:, len_col_name] = clean_series.str.len()
            
            # Convert to dictionary for single-line logging
            stats = df_processed[len_col_name].describe().round(2).to_dict()
            
            # Removed the \n and passed the dictionary
            logger.info("%s length stats: %s", col, stats)
        else:
            logger.warning("Column '%s' not found in DataFrame.", col)

    return df_processed


class TextLanguageAnalyzer:
    """
    Detects text language using a fast heuristic followed by Lingua fallback.
    """

    def __init__(self, languages: list = None, max_chars: int = 800):
        if languages is None:
            languages = [Language.FRENCH, Language.ENGLISH]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()
        self.max_chars = max_chars
        self.fr_accents = set("éèàùâêîôûç")
        self.fr_words = {
            "le", "la", "les", "des", "du", "une", "un", "pour", "avec", "sur",
            "dans", "ne", "pas", "ce", "cet", "cette", "que", "qui", "au",
            "aux", "est", "étant", "ainsi",
        }
        self.en_words = {
            "the", "and", "for", "with", "from", "not", "this", "that", "these",
            "those", "is", "are", "as", "by", "on", "in", "at",
        }

    def _heuristic_lang(self, text: str):
        t = text.strip().lower()
        if len(t) < 10:
            return None
        if any(ch in self.fr_accents for ch in t):
            return "fr"
        tokens = set(t.split())
        fr_hits = len(tokens & self.fr_words)
        en_hits = len(tokens & self.en_words)
        if fr_hits == 0 and en_hits == 0:
            return None
        if fr_hits >= en_hits + 1:
            return "fr"
        if en_hits >= fr_hits + 1:
            return "en"
        return None

    def detect(self, text: str) -> str:
        if not isinstance(text, str) or len(text.strip()) < 10:
            return "unknown"
        h = self._heuristic_lang(text)
        if h:
            return h
        try:
            lang = self.detector.detect_language_of(text[: self.max_chars])
            if lang == Language.FRENCH:
                return "fr"
            if lang == Language.ENGLISH:
                return "en"
        except Exception:
            pass
        return "unknown"

def run_language_pipeline(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Applies language detection to specified columns and logs distribution.

    Args:
        df:      Input DataFrame.
        columns: Columns to analyse. Defaults to ["TITLE", "TEXT"].

    Returns:
        Copy of df with '<COL>_LANG' columns added.
    """
    if columns is None:
        columns = ["TITLE", "TEXT"]

    df = df.copy()
    analyzer = TextLanguageAnalyzer() # Assuming this is defined elsewhere

    for col in columns:
        if col in df.columns:
            lang_col = f"{col}_LANG"
            df[lang_col] = df[col].fillna("").astype(str).apply(analyzer.detect)
            vc = df[lang_col].value_counts(dropna=False)
            
            # Build a single dictionary with counts and percentages
            lang_stats = {}
            for lang, count in vc.items():
                pct = round((count / len(df) * 100), 2) if len(df) > 0 else 0.0
                lang_stats[str(lang).upper()] = f"{count} ({pct}%)"
                
            # Log the entire distribution on one line
            logger.info("Language distribution (%s): %s", col, lang_stats)

    if len(columns) >= 2 and all(f"{c}_LANG" in df.columns for c in columns):
        col1, col2 = columns[0], columns[1]
        mismatch_mask = df[f"{col1}_LANG"] != df[f"{col2}_LANG"]
        mismatch_count = int(mismatch_mask.sum())
        mismatch_rate = (mismatch_count / len(df)) * 100 if len(df) > 0 else 0.0
        logger.info(
            "Language consistency: %d/%d matched | mismatch rate=%.2f%%",
            len(df) - mismatch_count,
            len(df),
            mismatch_rate,
        )

    return df


def deduplicate_text_content(
    df: pd.DataFrame,
    text_col: str = "TEXT",
    ref_col: str = "REFERENCE",
    output_audit_path: str = "final_DU_text_duplicates.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes duplicate rows by text content and writes an audit CSV.

    Args:
        df:                Input DataFrame.
        text_col:          Column to deduplicate on.
        ref_col:           Reference column for audit trail.
        output_audit_path: Path to write the duplicate audit CSV.

    Returns:
        (deduplicated_df, audit_df)
    """
    df = df.copy()
    rows_before = len(df)

    for col in ["TITLE", "TEXT"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    active_text = df[df[text_col] != ""]
    text_groups = active_text.groupby(text_col)[ref_col].apply(list).to_dict()
    duplicate_map = {text: refs for text, refs in text_groups.items() if len(refs) > 1}

    audit_data = []
    for text, refs in duplicate_map.items():
        for ref in refs:
            clones = [r for r in refs if r != ref]
            audit_data.append(
                {
                    ref_col: ref,
                    "CLONE_REFERENCES": ", ".join(clones),
                    "NUMBER_OF_CLONES": len(clones),
                    "TEXT_PREVIEW": text[:150] + "…" if len(text) > 150 else text,
                }
            )

    df_audit = pd.DataFrame(audit_data)
    df_clean = df.drop_duplicates(subset=[text_col], keep="first").copy()
    rows_after = len(df_clean)
    rows_removed = rows_before - rows_after

    logger.info(
        "Deduplication: %d unique patterns | %d rows removed (%.2f%% reduction) | %d records remaining",
        len(duplicate_map),
        rows_removed,
        (rows_removed / rows_before) * 100 if rows_before > 0 else 0.0,
        rows_after,
    )

    if not df_audit.empty:
        abs_audit_path = os.path.abspath(output_audit_path)
        os.makedirs(os.path.dirname(abs_audit_path), exist_ok=True)
        df_audit.to_csv(abs_audit_path, index=False, sep=";")
        logger.info("Audit report saved to: %s", abs_audit_path)
    else:
        logger.info("No duplicates found — no audit report created.")

    return df_clean, df_audit


_MATCH_CODE_HEADER_RE = re.compile(r"^#\[[A-Z]{2}\]#[^#]+#[^#]+#[^#]+#[^#]+#")


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames uppercase raw CSV column names to title-cased equivalents.
    REFERENCE → Reference, TITLE → Title, TEXT → Text, CREATION_DATE → Creation_Date.
    """
    rename_map = {}
    for src, dst in [("REFERENCE", "Reference"), ("TITLE", "Title"), ("TEXT", "Text"), ("CREATION_DATE", "Creation_Date")]:
        if src in df.columns:
            rename_map[src] = dst
    return df.rename(columns=rename_map) if rename_map else df


def add_match_code_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a boolean 'has_match_code_header' column based on regex header detection
    (pattern: #[XX]#...#...#...#...#).
    """
    if "Text" not in df.columns:
        df["has_match_code_header"] = False
        return df
    df["has_match_code_header"] = (
        df["Text"].fillna("").astype(str).str.contains(_MATCH_CODE_HEADER_RE, regex=True, na=False)
    )
    return df
