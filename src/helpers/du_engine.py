import os
import re
import time
import pickle
import concurrent.futures
from typing import Dict, Optional, List, Tuple, Union
from collections import Counter
import math

import numpy as np
import pandas as pd
import faiss
import requests
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.util.retry import Retry
from tqdm.auto import tqdm

from helpers.auth import get_valid_token
from helpers.patterns import extract_features
import helpers.cleaning as hc

# Disable warnings for insecure requests (self-signed certs inside corporate VPNs)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
API_BASE_URL = "https://m2m-rest.api.modelops.ec.airbus-v.corp/models"
TARGET_MODEL_ID = "amazon.titan-embed-text-v2:0"
NAMESPACE = "ground-test-ec-cb"
DIMS = 1024  # Embedding dimension size

# Set of strings considered "empty" or invalid for feature extraction
_BAD = {
    "non spécifié", "non specifie", "non_specifie",
    "n/a", "na", "none", "null", "nan", "unknown",
    "---", "ras", "non specifie.", "non spécifié.", "nonspecifie"
}

# -----------------------------------------------------------------------------
# Basic Helper Functions
# -----------------------------------------------------------------------------

def norm(s) -> str:
    """Normalizes a string by stripping whitespace and collapsing internal spaces."""
    return re.sub(r"\s+", " ", str(s).strip())

def valid_feature(v) -> bool:
    """
    Checks if a value is a valid, non-empty feature.
    Returns False for None, NaN, empty strings, or strings in the _BAD set.
    """
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = norm(v)
    if not s:
        return False
    s2 = s.casefold()
    bad = {x.casefold() for x in _BAD}
    return s2 not in bad

def build_keywords(row) -> str:
    """Concatenates Header and Body keywords into a deduplicated string."""
    vals = []
    for k in ["Header_Keyword", "Body_Keyword"]:
        v = row.get(k, "")
        if valid_feature(v):
            vals.append(norm(v))
    seen: set = set()
    uniq = []
    for v in vals:
        key = v.lower()
        if key not in seen:
            uniq.append(v)
            seen.add(key)
    return " ".join(uniq).strip()

_BAD_TOKEN_RE = re.compile(r"(?i)\b(?:non\s*sp[ée]cifi[ée]|n/?a|none|null|nan|unknown|ras)\b")

STOP_WORDS = {
    # Français
    "le", "la", "les", "un", "une", "des", "du", "de", "des", "au", "aux",
    "et", "ou", "où", "en", "dans", "pour", "par", "sur", "sous", "vers",
    "avec", "sans", "ce", "cette", "ces", "mon", "ton", "son", "ma", "ta", "sa",
    "qui", "que", "quoi", "dont", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "est", "sont", "être", "avoir", "pas", "ne", "plus", "très", "bien", "fait", "faire",
    "qu", "une", "est", "sur", "les",
    # Anglais
    "the", "is", "at", "which", "on", "for", "with", "and", "or", "to", "in", "of"
}

def bm25_tokenize(text: str) -> list[str]:
    """
    Custom tokenizer for BM25.
    - Conserve les identifiants alphanumériques et les accents français.
    - Supprime les tokens de moins de 3 caractères.
    - Supprime les mots vides (stop words).
    """
    # Étape 1 : On extrait tout (incluant les accents)
    raw_tokens = re.findall(r"[^\W_][\w\-/_.]*", str(text))
    
    cleaned_tokens = []
    for tok in raw_tokens:
        # Étape 2 : On vérifie la longueur
        if len(tok) < 3 and tok.upper() not in {"NC", "WO", "PN", "FIN"}:
            continue
            
        # Étape 3 : On vérifie si c'est un mot vide (en comparant en minuscules)
        if tok.lower() in STOP_WORDS:
            continue
            
        cleaned_tokens.append(tok)
        
    return cleaned_tokens


def _strip_token_prefix(tok: str) -> str:
    """Turns tokens like 'HEADER_PN:13HH-A' into '13HH-A' for display."""
    if not isinstance(tok, str):
        tok = str(tok)
    if ":" in tok:
        pref, rest = tok.split(":", 1)
        # Only strip well-known prefixes (avoid nuking times like 'A:320')
        if pref.isupper() and len(pref) <= 18:
            return rest
    return tok


def bm25_token_contributions(
    bm25_obj,
    query_tokens: List[str],
    doc_i: int,
    *,
    top_n: int = 15,
    group_prefixes: bool = True,
) -> Dict[str, float]:
    """
    Compute per-token BM25 contributions for a given (query, document) pair.

    Returns:
        dict {token_for_display: contribution_score}
    Notes:
        - Best-effort: works with rank_bm25.BM25Okapi-like objects.
        - If internals are missing, returns {}.
    """
    if bm25_obj is None or not query_tokens:
        return {}

    # best-effort introspection of bm25 internals
    idf = getattr(bm25_obj, "idf", None)
    doc_len = getattr(bm25_obj, "doc_len", None)
    avgdl = getattr(bm25_obj, "avgdl", None)
    k1 = float(getattr(bm25_obj, "k1", 1.5))
    b = float(getattr(bm25_obj, "b", 0.75))

    if not isinstance(idf, dict):
        return {}

    # Get term frequencies for the doc
    tf_map = None
    doc_freqs = getattr(bm25_obj, "doc_freqs", None)  # rank_bm25
    if isinstance(doc_freqs, list) and 0 <= doc_i < len(doc_freqs) and isinstance(doc_freqs[doc_i], dict):
        tf_map = doc_freqs[doc_i]
        dl = int(doc_len[doc_i]) if isinstance(doc_len, list) and 0 <= doc_i < len(doc_len) else int(sum(tf_map.values()))
    else:
        # fallback: try corpus tokens
        corpus = getattr(bm25_obj, "corpus", None)
        if isinstance(corpus, list) and 0 <= doc_i < len(corpus):
            toks = corpus[doc_i]
            if isinstance(toks, (list, tuple)):
                tf_map = Counter(map(str, toks))
                dl = int(len(toks))
            else:
                return {}
        else:
            return {}

    if avgdl is None:
        if isinstance(doc_len, list) and doc_len:
            avgdl = float(sum(doc_len) / max(1, len(doc_len)))
        else:
            avgdl = float(dl) if dl > 0 else 1.0

    # Use unique query tokens (BM25 sums per term)
    uniq_q = list(dict.fromkeys(map(str, query_tokens)))

    contrib_raw: Dict[str, float] = {}
    for t in uniq_q:
        f = float(tf_map.get(t, 0.0)) if tf_map is not None else 0.0
        if f <= 0:
            continue
        idf_t = float(idf.get(t, 0.0))
        if idf_t == 0.0:
            continue

        denom = f + k1 * (1.0 - b + b * (float(dl) / float(avgdl)))
        if denom <= 0:
            continue
        score = idf_t * (f * (k1 + 1.0)) / denom
        if score > 0:
            contrib_raw[t] = float(score)

    if not contrib_raw:
        return {}

    # Optionally group prefixed/unprefixed tokens
    if group_prefixes:
        grouped: Dict[str, float] = {}
        for tok, sc in contrib_raw.items():
            key = _strip_token_prefix(tok)
            grouped[key] = grouped.get(key, 0.0) + float(sc)
        contrib = grouped
    else:
        contrib = contrib_raw

    # Keep top_n
    items = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:max(1, int(top_n))]
    return {k: round(float(v), 6) for k, v in items}

def sanitize_text(text: str) -> str:
    """Cleans text by removing 'bad' tokens, expanding common abbreviations, and normalizing whitespace."""
    if not valid_feature(text):
        return ""
    t = norm(text)
    t = _BAD_TOKEN_RE.sub(" ", t)
    t = re.sub(r"(?i)\bdep\s*/\s*rep\b", "dépose / repose", t)
    t = re.sub(r"(?i)\bd\s*/\s*r\b", "dépose / repose", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def strip_id_noise_for_dense(text: str, max_ids_to_keep: int = 6) -> str:
    """
    Preprocesses text for Dense Embedding to prevent lists of IDs from dominating the vector.

    Logic:
    1. Removes leading document markers (WO, NC, QSR).
    2. Scans tokens; if a token looks like a technical ID (alphanumeric), it counts it.
    3. Keeps only the first `max_ids_to_keep` IDs.
    4. Retains all natural language words.

    Args:
        text: The input text.
        max_ids_to_keep: Maximum number of ID-like tokens to preserve.

    Returns:
        The cleaned text string.
    """
    t = sanitize_text(text)
    if not t:
        return ""

    # remove leading doc markers (NC/WO/QSR + numbers) to focus on content
    t = re.sub(r"(?i)^\s*(wo|nc|qsr|ic|so\d?)\s*[,#:]?\s*\d[\w\-/.]*\s*", "", t).strip()
    t = re.sub(r"(?i)^\s*nc\s*\d+\s*[,#:]?\s*", "", t).strip()
    t = re.sub(r"^[\s,;:\-–—]+", "", t).strip()

    raw_tokens = re.split(r"(\s+|[,\;\|\(\)\[\]\{\}])", t)

    def looks_like_id(tok: str) -> bool:
        tok = tok.strip()
        if len(tok) < 3:
            return False
        if not re.search(r"\d", tok):
            return False
        return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-/_.]*", tok) is not None

    id_tokens = [tok for tok in raw_tokens if looks_like_id(tok)]
    
    # optimization: if few IDs, return whole string cleaned
    if len(id_tokens) <= max_ids_to_keep:
        return re.sub(r"\s+", " ", "".join(raw_tokens)).strip()

    kept, kept_ids = [], []
    for tok in raw_tokens:
        if looks_like_id(tok):
            if len(kept_ids) < max_ids_to_keep:
                kept.append(tok)
                kept_ids.append(tok)
            else:
                continue # Skip excess IDs
        else:
            kept.append(tok)

    return re.sub(r"\s+", " ", "".join(kept)).strip()

def clean_title_for_dense(title: str) -> str:
    """Strips structured codes, hash markers, and ID-heavy tokens from a title before embedding."""
    t = sanitize_text(title)
    if not t:
        return ""
    t = re.sub(r"\[[^\]]*\]", " ", t)
    t = re.sub(r"#(?:[^#]+)#", " ", t)
    t = re.sub(r"(?i)^\s*(nf|nb|nv|cw)\]?", " ", t)
    t = re.sub(r"^\s*\d{7,}\s+", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    id_like = sum(bool(re.search(r"\d", w)) for w in t.split())
    if t and id_like / max(1, len(t.split())) > 0.6:
        return ""
    return t


def build_dense_text(row: pd.Series, max_ids_to_keep: int = 6) -> str:
    """
    Constructs the string to be sent to the embedding model.

    Composition: Title + Narrative + Reason + Action + Rubric + System + Keywords.
    Intent/Category are included only when the Narrative is absent, to prevent
    category labels from skewing semantic matching when rich text exists.
    """
    parts = []

    title = clean_title_for_dense(row.get("Title", ""))
    if valid_feature(title):
        parts.append(f"Title: {title}")

    narrative = strip_id_noise_for_dense(row.get("Text", ""), max_ids_to_keep=max_ids_to_keep)
    has_narrative = bool(narrative)
    if narrative:
        parts.append(f"Narrative: {narrative}")

    for label, key in [("Reason", "Reason"), ("Action", "Action"), ("Rubric", "Rubric"), ("System", "System")]:
        v = sanitize_text(row.get(key, ""))
        if valid_feature(v):
            parts.append(f"{label}: {v}")

    if not has_narrative:
        for label, key in [("Intent", "Intent"), ("Category", "Category")]:
            v = sanitize_text(row.get(key, ""))
            if valid_feature(v):
                parts.append(f"{label}: {v}")

    kw = sanitize_text(build_keywords(row))
    if kw:
        parts.append(f"Keywords: {kw}")

    return " | ".join(p for p in parts if p.strip()).strip()


def build_lex_text(row_lex: pd.Series) -> str:
    """Builds the BM25 query string from Lex_ID (prefixed ID tokens) and Lex_Text."""
    lex_id = row_lex.get("Lex_ID", "")
    lex_txt = row_lex.get("Lex_Text", "")
    if not valid_feature(lex_id) and not valid_feature(lex_txt):
        return ""
    return norm(sanitize_text(f"{lex_id} {lex_txt}".strip()))


# -----------------------------------------------------------------------------
# Incoming Lexical Text Builder (for BM25 query construction)
# -----------------------------------------------------------------------------
_ID_ALLOWED = re.compile(r"[^A-Z0-9\-/_.]")

def _split_multi(v: str):
    return re.split(r"[|,;()\s]+", v)

def normalize_id_token(tok: str) -> str:
    """
    Standardizes ID tokens (e.g., stripping bad chars, uppercase)
    and validates length/content.
    """
    if tok is None:
        return ""
    tok = str(tok).strip().upper()
    tok = _ID_ALLOWED.sub("", tok)
    tok = tok.strip("-/")

    if len(tok) < 3 or len(tok) > 25:
        return ""
    if not any(c.isdigit() for c in tok):
        return ""
    return tok

def build_incoming_lex_text(row: pd.Series, max_tokens: int = 40) -> str:
    """
    BM25 query string:
    - prefixed technical ID tokens (high-signal)
    - + lexical context (title + narrative-like fields)
    """

    id_cols = [
        "Header_PN", "Header_FIN",
        "Body_PN", "Body_FIN",
        "Impacted_Equip_FINs", "FIN_Elements",
        "Cable_Number", "Panel_Number",
        "Disconnected_Plugs", "Removed_FINs",
        "Cabling_Ref_FIN",
        # Doc_Type/Doc_Ref are often UNIQUE → keep but don't let them dominate
        "Doc_Type", "Doc_Ref",
    ]

    # --- A) high-signal tokens (prefixed) ---
    tokens = []
    for col in id_cols:
        v = row.get(col, "")
        if not valid_feature(v):
            continue

        v = norm(v)

        # fix common Airbus token break: "13HH- A" -> "13HH-A"
        v = re.sub(r"-\s+", "-", v)

        for part in _split_multi(v):
            tok = normalize_id_token(part)
            if tok:
                tokens.append(f"{col.upper()}:{tok}")

                # Optional: add unprefixed version for robustness
                # (helps when corpus/query prefixes differ or missing)
                tokens.append(tok)

    # dedupe while preserving order + cap
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
        if len(uniq) >= max_tokens:
            break

    # --- B) lexical context (rich text) ---
    title = norm(row.get("Title", "")) if valid_feature(row.get("Title", "")) else ""
    text = norm(row.get("Text", "")) if valid_feature(row.get("Text", "")) else ""
    system = norm(row.get("System", "")) if valid_feature(row.get("System", "")) else ""
    rubric = norm(row.get("Rubric", "")) if valid_feature(row.get("Rubric", "")) else ""
    reason = norm(row.get("Reason", "")) if valid_feature(row.get("Reason", "")) else ""
    action = norm(row.get("Action", "")) if valid_feature(row.get("Action", "")) else ""
    kw = build_keywords(row)

    context = " ".join([title, text, system, rubric, reason, action, kw]).strip()

    out = " ".join(uniq + [context]).strip()
    return out if out else (title or "DU")


def build_lex_id(row: pd.Series, id_cols: List[str], max_tokens: int = 15) -> str:
    """
    Builds a space-separated string of prefixed ID tokens from the given columns.
    Used to populate the Lex_ID column in the lexical parquet (corpus side).
    Format: COL_NAME:TOKEN (e.g. HEADER_PN:13HH-A).
    """
    tokens = []
    for col in id_cols:
        v = row.get(col, "")
        if not valid_feature(v):
            continue
        for part in _split_multi(str(v)):
            tok = normalize_id_token(part)
            if tok:
                tokens.append(f"{col.upper()}:{tok}")
    seen: set = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return " ".join(uniq[:max_tokens])


def build_lex_text_series(df: pd.DataFrame, id_cols: List[str], base_cols: List[str]) -> pd.Series:
    """
    Builds a BM25-ready text Series by concatenating ID columns and base text columns.
    Used to populate the Lex_Text column in the lexical parquet (corpus side).
    """
    lex = pd.Series([""] * len(df), index=df.index, dtype="object")
    if id_cols:
        lex_ids = df[id_cols].fillna("").astype(str).agg(" ".join, axis=1)
        lex = lex + " " + lex_ids
    for col in base_cols:
        if col in df.columns:
            lex = lex + " " + df[col].fillna("").astype(str)
    return lex.str.replace(r"\s+", " ", regex=True).str.strip()


# -----------------------------------------------------------------------------
# Data Cleaning (Mapping to helper functions)
# -----------------------------------------------------------------------------
def clean_features_in_place(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies column-specific cleaning functions imported from helpers.cleaning.
    """
    cleaning_map = {
        "Category": hc.clean_categories,
        "Reason_Code": hc.clean_reason_code,
        "Rubric": hc.clean_rubric,
        "Plug_Action": hc.clean_plug_action,
        "Cabling_Reason": hc.clean_cabling_reason,
        "Action_Description": hc.clean_action_description,
        "FAL": hc.clean_fal,
        "Tank_Drained": hc.clean_tank_drained,
        "Circuit": hc.clean_circuit,
        "Side": hc.clean_side,
        "Equip_Downloadable": hc.clean_equip_downloadable,
        "Header_Keyword": hc.clean_keyword,
        "Body_Keyword": hc.clean_keyword,
        "System": hc.clean_system,
        "Narrative": hc.clean_narrative,
        "TITLE": hc.clean_title,
    }
    for col, func in cleaning_map.items():
        if col in df.columns:
            if col == "Category":
                # Category cleaning often requires the whole DF context
                df[col] = func(df, col)
            else:
                df[col] = df[col].apply(func)
    return df


# -----------------------------------------------------------------------------
# Network / Session Management
# -----------------------------------------------------------------------------
def build_session(max_workers: int) -> requests.Session:
    """Creates a Requests session with a connection pool and retry strategy sized for concurrency."""
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.8, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers, max_retries=retry)
    session.mount("https://", adapter)
    return session


# -----------------------------------------------------------------------------
# Embedding API (with Retries and Token Refresh)
# -----------------------------------------------------------------------------
def _embed_call(
    session: requests.Session,
    text: str,
    token: str,
    timeout: int = 30,
    *,
    api_url: str = None,
    model_id: str = None,
    namespace: str = None,
    dims: int = None,
) -> Tuple[np.ndarray, str]:
    """
    Low-level embedding API call.
    Handles 401 Unauthorized by refreshing the token and retrying once.
    API params default to module-level constants if not provided.
    """
    _api_url = api_url or API_BASE_URL
    _model_id = model_id or TARGET_MODEL_ID
    _namespace = namespace or NAMESPACE
    _dims = dims or DIMS

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "model_id": _model_id,
        "namespace": _namespace,
    }
    payload = {"inputText": text, "dimensions": _dims, "normalize": True}

    resp = session.post(_api_url, json=payload, headers=headers, verify=False, timeout=timeout)

    if resp.status_code == 401:
        token = get_valid_token()
        headers["Authorization"] = f"Bearer {token}"
        resp = session.post(_api_url, json=payload, headers=headers, verify=False, timeout=timeout)

    resp.raise_for_status()
    vec = resp.json().get("response_body", {}).get("embedding")
    return np.array(vec, dtype=np.float32), token


def embed_one_safe(
    session: requests.Session,
    text: str,
    token: str,
    timeout: int = 30,
    retries: int = 3,
    *,
    api_url: str = None,
    model_id: str = None,
    namespace: str = None,
    dims: int = None,
) -> Tuple[Optional[np.ndarray], str, Optional[str]]:
    """
    Safe wrapper for embedding with exponential backoff retries.

    API params (api_url, model_id, namespace, dims) default to module-level
    constants when not provided, allowing callers with config-sourced values
    to override them without breaking existing call sites.

    Returns:
        tuple: (vector, current_token, error_message)
        If successful, vector is np.ndarray and error_message is None.
    """
    _dims = dims or DIMS
    if not isinstance(text, str) or not text.strip():
        return None, token, "empty_input"

    last_err = None
    for attempt in range(retries):
        try:
            vec, token = _embed_call(
                session, text, token, timeout=timeout,
                api_url=api_url, model_id=model_id, namespace=namespace, dims=dims,
            )
            if vec is None or vec.shape[0] != _dims:
                return None, token, "bad_vector"
            return vec, token, None
        except Exception as e:
            last_err = f"exc_{type(e).__name__}"
            time.sleep(0.8 * (attempt + 1))
            continue

    return None, token, last_err or "failed"


def rrf_fuse_weighted(dense_idx, sparse_idx, rrf_k=60, w_dense=1.2, w_bm25=1.0):
    """
    Weighted Reciprocal Rank Fusion (RRF).
    
    Formula: Score = Sum( weight / (rrf_k + rank) )
    Merges rankings from Dense (Vector) and Sparse (BM25) searches.
    """
    scores = {}
    for rank, i in enumerate(sparse_idx):
        scores[int(i)] = scores.get(int(i), 0.0) + (w_bm25 / (rrf_k + rank + 1))
    for rank, i in enumerate(dense_idx):
        if int(i) == -1: # FAISS padding
            continue
        scores[int(i)] = scores.get(int(i), 0.0) + (w_dense / (rrf_k + rank + 1))
    return scores


# -----------------------------------------------------------------------------
# Main Retrieval Engine Class
# -----------------------------------------------------------------------------
class DUEngine:
    """
    Hybrid Retrieval Engine utilizing FAISS (Dense) and BM25 (Sparse).
    Handles artifacts loading, incoming data preparation, and hybrid search execution.
    """
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """
        Initializes the engine by loading pre-computed indexes and metadata.
        
        Args:
            artifacts_dir: Path to directory containing .index, .npy, .pkl, and .parquet files.
        """
        self.base = os.path.abspath(artifacts_dir)
        idx_dir = os.path.join(self.base, "indexes")

        # Define paths
        self.faiss_path = os.path.join(idx_dir, "faiss_C_1024.index")
        self.docids_path = os.path.join(idx_dir, "doc_ids_C_1024.npy")
        self.bm25_path = os.path.join(idx_dir, "bm25_C.pkl")
        self.meta_path = os.path.join(idx_dir, "meta_C.parquet")

        # Verification
        for p in [self.faiss_path, self.docids_path, self.bm25_path, self.meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing artifact: {p}")

        print("Loading FAISS...")
        self.index = faiss.read_index(self.faiss_path)
        self.doc_ids = np.load(self.docids_path, allow_pickle=True).tolist()

        print("Loading meta...")
        self.meta = pd.read_parquet(self.meta_path)
        self.meta["Reference"] = self.meta["Reference"].astype(str).str.strip()
        # Create quick lookup for metadata aggregation later
        self.meta_lookup = self.meta.set_index("Reference").to_dict(orient="index")

        print("Loading BM25...")
        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

        # Inverted index map for BM25 subset optimization
        self.inv_index_path = os.path.join(idx_dir, "inv_index_ids_C.pkl")
        with open(self.inv_index_path, "rb") as f:
            self.inv_index = pickle.load(f)
        
        self.vectors_path = os.path.join(idx_dir, "vectors_C_1024.npy")
        print("Loading vectors matrix for MMR...")
        self.doc_vectors = np.load(self.vectors_path) # Shape: (N, 1024)

        self.token = get_valid_token()
        self.session = build_session(max_workers=35)
        print("Engine ready")

    # -------- Incoming preparation (raw -> semantic-like row) --------
    def prepare_incoming(self, df_raw: pd.DataFrame, match_code: bool = True) -> pd.DataFrame:
        """
        Transforms raw CSV data into a format ready for retrieval.
        
        Steps:
        1. Validate columns.
        2. Filter for specific Answer Types.
        3. regex filter for specific header format (optional).
        4. Extract features using regex patterns (helper).
        5. Clean data features.
        6. Generate 'dense_text' (for embedding) and 'incoming_lex_text' (for BM25).
        """
        df = df_raw.copy()

        required = ["REFERENCE", "TITLE", "TEXT", "ANSWER_TYPE_NAME"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Incoming CSV missing {c}. Found: {list(df.columns)}")

        df = df[df["ANSWER_TYPE_NAME"] == "Intervention"].copy()

        if match_code:
            # Filter rows that look like technical logs based on a specific regex pattern
            regex_header = r"^#\[[A-Z]{2}\]#[^#]+#[^#]+#[^#]+#[^#]+#"
            df = df[df["TEXT"].astype(str).str.contains(regex_header, regex=True, na=False)].copy()

        # Extract structured features from the text blob
        feat = df["TEXT"].apply(lambda x: pd.Series(extract_features(x)))
        df = pd.concat([df, feat], axis=1)

        if "Narrative" not in df.columns:
            df["Narrative"] = df["TEXT"]

        df = clean_features_in_place(df)

        rename_map = {
            "REFERENCE": "Reference",
            "TITLE": "Title",
            "Narrative": "Text",
            "Reason_Code": "Reason",
            "Action_Description": "Action",
        }
        df = df.rename(columns=rename_map)
        df["Reference"] = df["Reference"].astype(str).str.strip()

        # Build the actual query strings
        df["dense_text"] = df.apply(build_dense_text, axis=1)
        df["incoming_lex_text"] = df.apply(build_incoming_lex_text, axis=1)
        return df

    def apply_lexicon_to_incoming(
        self,
        df_in: pd.DataFrame,
        lexicon_map: Dict[str, str],
        dense_col: str = "dense_text",
        lex_col: str = "incoming_lex_text",
        ref_col: str = "Reference",
    ) -> pd.DataFrame:
        """
        Enrichit les champs query (dense + BM25) pour les incoming en fonction d'un lexicon_map:
        {Reference -> lexicon_text}.
        IMPORTANT: ne touche PAS aux indexes corpus (FAISS/BM25) : ça ne change que la requête.
        """
        if not lexicon_map:
            return df_in

        df = df_in.copy()
        if ref_col not in df.columns:
            return df

        df[ref_col] = df[ref_col].astype(str).str.strip()

        def _enrich_row(row: pd.Series) -> pd.Series:
            ref = str(row.get(ref_col, "")).strip()
            lex = (lexicon_map.get(ref, "") or "").strip()
            if not lex:
                return row

            # BM25 query: Just append the raw keywords (space separated) to avoid TF-IDF noise
            base_lex = str(row.get(lex_col, "") or "").strip()
            row[lex_col] = f"{base_lex} {lex}" if base_lex else lex

            # Dense query: Keep the context marker so the embedding model understands the prompt
            base_dense = str(row.get(dense_col, "") or "").strip()
            row[dense_col] = f"{base_dense} | Lexique: {lex}" if base_dense else f"Lexique: {lex}"

            return row

        df = df.apply(_enrich_row, axis=1)
        return df

    @staticmethod
    def build_lexicon_map_from_csv(lexicon_csv_path: str) -> Dict[str, str]:
        """
        Lit le CSV append-only produit par Streamlit et retourne {incoming_reference -> lexicon_text}
        en prenant la dernière entrée (timestamp max) par incoming_reference.
        """
        if not lexicon_csv_path or not os.path.exists(lexicon_csv_path):
            return {}

        try:
            df = pd.read_csv(lexicon_csv_path, low_memory=False)
        except Exception:
            return {}

        if df.empty:
            return {}

        # colonnes attendues depuis ton Streamlit
        if "incoming_reference" not in df.columns or "lexicon_text" not in df.columns:
            return {}

        if "timestamp" in df.columns:
            try:
                df = df.sort_values("timestamp")
            except Exception:
                pass

        df_last = df.groupby("incoming_reference", as_index=False).tail(1)
        return dict(
            zip(
                df_last["incoming_reference"].astype(str),
                df_last["lexicon_text"].fillna("").astype(str),
            )
        )


    # -------- Retrieval Logic --------
    def retrieve_one(
        self,
        dense_text: str,
        qvec: Optional[np.ndarray] = None,
        lex_text: Optional[str] = None,
        top_k: int = 10,
        candidate_k: int = 200,
        bm25_candidate_k: int = 400,
        filters: Optional[Dict[str, str]] = None,
        use_hybrid: bool = True,
        exclude_ref: Optional[str] = None,
        incoming_meta: Optional[Dict[str, str]] = None,
        rrf_k: int = 60,
        w_dense: float = 1.2,
        w_bm25: float = 1.0,
        boost_cat: float = 0.05,
        boost_int: float = 0.03,
        use_mmr: bool = False,
        mmr_lambda: float = 0.6,
        debug_timing: bool = False,
        explain_bm25: bool = True,
        explain_bm25_topn: int = 15,
        explain_bm25_group: bool = True,
    ) -> Union[List[Dict], Tuple[List[Dict], Dict]]:
        """
        Performs hybrid retrieval for a single query.

        Algorithm:
        1. Embed Query (if qvec not provided).
        2. FAISS Search (Dense) -> Get Top N candidates.
        3. BM25 Search (Sparse) -> Get Top N candidates.
           - Optimization: Uses an inverted index to only score documents containing query tokens 
             if the subset is small, avoiding a full corpus scan.
        4. RRF Fusion (Weighted) -> Combine scores.
        5. Filtering & Boosting -> Apply metadata filters and boost scores if Categories/Intents match.
        
        Returns:
            List of result dictionaries. If debug_timing=True, returns (results, timing_dict).
        """
        t_all0 = time.perf_counter()

        # --- 1. Embed (optional) ---
        t0 = time.perf_counter()
        if qvec is None:
            vec, self.token, err = embed_one_safe(self.session, dense_text, self.token, timeout=30, retries=3)
            if err is not None or vec is None:
                # fail-safe: empty results but still return timing if needed
                timing = {"t_total": round(time.perf_counter() - t_all0, 4), "t_embed": round(time.perf_counter() - t0, 4), "embed_err": err or "failed"}
                return ([], timing) if debug_timing else []
            qvec = vec
        qvec = qvec.reshape(1, -1)
        t_embed = time.perf_counter() - t0

        # --- 2. FAISS Search (Dense) ---
        t0 = time.perf_counter()
        dense_scores, dense_idx = self.index.search(qvec, candidate_k)
        dense_idx = dense_idx[0]
        dense_scores = dense_scores[0]
        t_faiss = time.perf_counter() - t0

        # Map ID -> Rank/Score
        dense_rank = {int(i): (r + 1) for r, i in enumerate(dense_idx) if int(i) != -1}
        dense_score_map = {int(i): float(s) for i, s in zip(dense_idx, dense_scores) if int(i) != -1}

        # --- 3. BM25 Search (Sparse) ---
        t0 = time.perf_counter()
        if lex_text is None or not str(lex_text).strip():
            lex_text = dense_text
        tokens = bm25_tokenize(lex_text)

        # Optimization: collect candidate docs via inverted index first
        subset_idx = set()
        for tok in tokens:
            if tok in self.inv_index:
                subset_idx.update(self.inv_index[tok])
        subset_idx = list(subset_idx)

        # Fallback logic: 
        # If the subset of documents containing these tokens is very small (<20),
        # we might just do a full scan (or logic implies opposite here usually, but code 
        # treats <20 as "too small to trust subset only" -> scan all).
        if len(subset_idx) < 20:
            # too small -> fallback to full bm25 scoring across all docs
            bm25_scores = self.bm25.get_scores(tokens)
            top_sparse = np.argsort(bm25_scores)[::-1][:bm25_candidate_k]
            bm25_mode = "full"
        else:
            # score only the subset of documents (faster)
            bm25_scores = np.zeros(len(self.doc_ids))
            subset_scores = self.bm25.get_batch_scores(tokens, subset_idx)

            for doc_i, score in zip(subset_idx, subset_scores):
                bm25_scores[doc_i] = score

            top_sparse = np.argsort(bm25_scores)[::-1][:bm25_candidate_k]
            bm25_mode = "subset"

        t_bm25 = time.perf_counter() - t0

        bm25_rank = {int(i): (r + 1) for r, i in enumerate(top_sparse)}
        bm25_score_map = {int(i): float(bm25_scores[int(i)]) for i in top_sparse}

        # --- 4. Fusion + sort ---
        t0 = time.perf_counter()
        if not use_hybrid:
            fused = dense_score_map
            fused_kind = "dense"
        else:
            fused = rrf_fuse_weighted(dense_idx, top_sparse, rrf_k=rrf_k, w_dense=w_dense, w_bm25=w_bm25)
            fused_kind = "rrf"

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        t_fuse_sort = time.perf_counter() - t0

        # --- 5. Post-Processing: boost + filters + top_k selection ---
        t0 = time.perf_counter()

        ranked = ranked[:max(200, top_k * 20)]  # Performance cap before doing meta lookups

        tmp = []
        for idx, score in ranked:
            ref = self.doc_ids[int(idx)]

            if exclude_ref and ref == exclude_ref:
                continue

            meta = self.meta_lookup.get(ref, {})

            # Hard filtering (Exact match on metadata)
            if filters:
                ok = True
                for k, v in filters.items():
                    if str(meta.get(k, "")).lower() != str(v).lower():
                        ok = False
                        break
                if not ok:
                    continue

            # Soft Boosting (Category/Intent match)
            boosted = float(score)
            if use_hybrid and incoming_meta:
                qcat = incoming_meta.get("Category")
                qint = incoming_meta.get("Intent")
                if valid_feature(qcat) and meta.get("Category") == qcat:
                    boosted += float(boost_cat)
                if valid_feature(qint) and meta.get("Intent") == qint:
                    boosted += float(boost_int)

            tmp.append((idx, score, boosted, ref, meta))

        # Re-sort by final boosted score
        tmp.sort(key=lambda x: x[2], reverse=True)

        # ==========================================
        # MMR (Maximal Marginal Relevance) Loop
        # ==========================================
        if use_mmr and len(tmp) > 0:
            candidate_indices = [int(x[0]) for x in tmp]
            # Normalize relevance scores to [0, 1] so they are on the same scale as Cosine Similarity
            raw_scores = np.array([float(x[2]) for x in tmp])
            max_s, min_s = np.max(raw_scores), np.min(raw_scores)
            if max_s > min_s:
                norm_scores = (raw_scores - min_s) / (max_s - min_s)
            else:
                norm_scores = np.ones_like(raw_scores)

            # Instantaneous slice of the 1024D vectors for all candidates
            candidate_vectors = self.doc_vectors[candidate_indices]

            selected_idx = []
            unselected_idx = list(range(len(tmp)))
            k_to_pick = min(top_k, len(tmp))

            for _ in range(k_to_pick):
                if not selected_idx:
                    # Slot 1: Just pick the highest relevance score
                    best_local_idx = unselected_idx[np.argmax(norm_scores[unselected_idx])]
                else:
                    # Vectorized penalty calculation
                    unsel_vecs = candidate_vectors[unselected_idx]  # Shape: (U, 1024)
                    sel_vecs = candidate_vectors[selected_idx]      # Shape: (S, 1024)
                    
                    # Dot product matrix (Cosine Similarity since normalized)
                    sim_matrix = np.dot(unsel_vecs, sel_vecs.T)     # Shape: (U, S)
                    
                    # Get the max similarity to ANY already-selected document
                    max_sim_to_selected = np.max(sim_matrix, axis=1) # Shape: (U,)
                    
                    # MMR Formula: (Lambda * Relevance) - ((1 - Lambda) * Penalty)
                    mmr_scores = (mmr_lambda * norm_scores[unselected_idx]) - ((1.0 - mmr_lambda) * max_sim_to_selected)
                    
                    # Pick the best adjusted score
                    best_local_idx = unselected_idx[np.argmax(mmr_scores)]
                
                selected_idx.append(best_local_idx)
                unselected_idx.remove(best_local_idx)

            # Re-order the `tmp` list based on the MMR selections
            tmp = [tmp[i] for i in selected_idx]

        out = []
        for idx, raw_score, final_score, ref, meta in tmp:
            if len(out) >= top_k:
                break
            dcos = dense_score_map.get(int(idx))
            bm25score = bm25_score_map.get(int(idx))

            bm25_breakdown = {}
            if explain_bm25 and lex_text:
                try:
                    bm25_breakdown = bm25_token_contributions(
                        self.bm25,
                        tokens,
                        int(idx),
                        top_n=explain_bm25_topn,
                        group_prefixes=explain_bm25_group,
                    )
                except Exception:
                    bm25_breakdown = {}


            out.append({
                "reference": ref,
                "rrf_score": round(float(raw_score), 6) if fused_kind == "rrf" else None,
                "dense_cosine": None if dcos is None else round(float(dcos), 6),
                "bm25_score": None if bm25score is None else round(float(bm25score), 6),
                "dense_rank": dense_rank.get(int(idx)),
                "bm25_rank": bm25_rank.get(int(idx)),
                "score_raw_rrf": round(float(raw_score), 6) if fused_kind == "rrf" else None,
                "score_final": round(float(final_score), 6),
                "title": meta.get("Title", ""),
                "text": meta.get("Text", ""),
                "dense_input": meta.get("dense_text", ""),
                "bm25_input": meta.get("lex_text", ""),
                "bm25_token_breakdown": str(bm25_breakdown) if bm25_breakdown else "{}",
                "intent": meta.get("Intent", ""),
                "category": meta.get("Category", ""),
                "reason": meta.get("Reason", ""),
                "action": meta.get("Action", ""),
                "system": meta.get("System", ""),
                "rubric": meta.get("Rubric", ""),
                "FAL": meta.get("FAL", ""),
                "match_code_present": meta.get("has_match_code_header", ""),
            })

        t_post = time.perf_counter() - t0
        t_total = time.perf_counter() - t_all0

        timing = {
            "t_total": round(t_total, 4),
            "t_embed": round(t_embed, 4),
            "t_faiss": round(t_faiss, 4),
            "t_bm25": round(t_bm25, 4),
            "t_fuse_sort": round(t_fuse_sort, 4),
            "t_post": round(t_post, 4),
            "candidate_k": int(candidate_k),
            "bm25_candidate_k": int(bm25_candidate_k),
            "bm25_mode": bm25_mode,
            "bm25_subset_size": len(subset_idx),
            "fused_kind": str(fused_kind),
        }

        if debug_timing:
            return out, timing
        return out

    # -------- Batch Retrieval --------
    def retrieve_batch(
        self,
        df_incoming_prepared: pd.DataFrame,
        top_k: int = 10,
        candidate_k: int = 200,
        bm25_candidate_k: int = 400,
        filters: Optional[Dict[str, str]] = None,
        use_hybrid: bool = True,
        exclude_self: bool = True,
        rrf_k: int = 60,
        w_dense: float = 1.2,
        w_bm25: float = 1.0,
        boost_cat: float = 0.05,
        boost_int: float = 0.03,
        use_mmr: bool = False,
        mmr_lambda: float = 0.6,
        lexicon_map: Optional[Dict[str, str]] = None,
        debug_timing: bool = False,
        explain_bm25: bool = True,
        explain_bm25_topn: int = 15,
        explain_bm25_group: bool = True,
        # embedding knobs
        max_workers: int = 35,
        timeout: int = 30,
        retries: int = 3,
        show_progress: bool = True,
        return_query_embeddings: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Processes a dataframe of queries in batch.
        
        Optimizations:
        - Deduplicates 'dense_text' before embedding to avoid redundant API calls.
        - Uses ThreadPoolExecutor for parallel embedding.
        
        Args:
            df_incoming_prepared: DataFrame processed by `prepare_incoming`.
            max_workers: Concurrency limit for embedding API calls.
            exclude_self: If True, removes the query document itself from results.
            
        Returns:
            DataFrame containing exploded results (1 row per result per query).
        """

        # -----------------------------
        # 0) Optional: enrich incoming queries with lexicon (affects BOTH BM25 + dense)
        # -----------------------------
        if lexicon_map:
            df_incoming_prepared = self.apply_lexicon_to_incoming(df_incoming_prepared, lexicon_map)

        # -----------------------------
        # 1) Batch embedding unique dense_text
        # -----------------------------
        dense_series = df_incoming_prepared.get("dense_text", pd.Series([], dtype=str)).fillna("").astype(str)
        uniq_dense = dense_series.unique().tolist()
        uniq_dense = [t for t in uniq_dense if t.strip()]

        workers = min(int(max_workers), max(1, len(uniq_dense)))

        t0 = time.perf_counter()
        text_to_vec: Dict[str, np.ndarray] = {}
        text_to_err: Dict[str, str] = {}

        def _job(text: str):
            # Each thread embeds one text
            vec, tok, err = embed_one_safe(self.session, text, self.token, timeout=timeout, retries=retries)
            return text, vec, tok, err

        if show_progress:
            pbar = tqdm(total=len(uniq_dense), desc=f"Embedding incoming (unique={len(uniq_dense)} workers={workers})")
        else:
            pbar = None

        # NOTE: We don't aggressively share-token across threads to avoid race weirdness.
        # Each call can refresh on 401 anyway. After batch, we refresh token once.
        success = 0
        fail = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_job, text) for text in uniq_dense]
            for fut in concurrent.futures.as_completed(futs):
                text, vec, tok, err = fut.result()

                if err is None and vec is not None and vec.shape[0] == DIMS:
                    text_to_vec[text] = vec
                    success += 1
                else:
                    text_to_err[text] = err or "failed"
                    fail += 1

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        # refresh token once after the heavy lifting
        self.token = get_valid_token()

        t_embed_total = time.perf_counter() - t0
        total = success + fail
        success_rate = (success / total * 100.0) if total else 0.0
        avg_per = (t_embed_total / total) if total else 0.0

        print("\nEmbedding phase summary")
        print("----------------------")
        print(f"Unique texts        : {total}")
        print(f"Success             : {success}")
        print(f"Failed              : {fail}")
        print(f"Success rate        : {success_rate:.2f}%")
        print(f"Total embed time    : {t_embed_total:.2f}s")
        print(f"Avg per embedding   : {avg_per:.3f}s")
        print(f"Workers used        : {workers}")

        if success_rate < 98.0:
            print(f"\nWARNING: embedding success_rate={success_rate:.2f}% < 98%")
            if fail:
                sample = list(text_to_err.items())[:5]
                print("   Sample failures (first 5):")
                for txt, err in sample:
                    print(f"   - err={err} | text_preview={txt[:120]!r}")
            print()

        # -----------------------------
        # 2) Retrieval loop (Using pre-computed embeddings)
        # -----------------------------
        rows = []
        timings = []

        it = df_incoming_prepared.iterrows()
        if show_progress:
            it = tqdm(it, total=len(df_incoming_prepared), desc="Retrieving (incoming rows)")
        
        emb_rows = []  # for caching incoming query embeddings
        for _, r in it:
            incoming_ref = r.get("Reference", "")
            exclude_ref = incoming_ref if exclude_self else None

            dense_text = str(r.get("dense_text", "") or "")
            lex_text = str(r.get("incoming_lex_text", "") or "")

            # Look up vector from batch result
            qvec = text_to_vec.get(dense_text)
            
            if return_query_embeddings:
                emb_rows.append({
                    "incoming_reference": str(incoming_ref),
                    "dense_text": dense_text,
                    "incoming_lex_text": lex_text,
                    "qvec": None if qvec is None else qvec.astype(np.float32).tolist(),
                    "embed_missing": bool(qvec is None),
                })

            incoming_meta = {"Category": r.get("Category", ""), "Intent": r.get("Intent", "")}

            if debug_timing:
                hits, timing = self.retrieve_one(
                    dense_text=dense_text,
                    qvec=qvec,
                    lex_text=lex_text,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    bm25_candidate_k=bm25_candidate_k,
                    filters=filters,
                    use_hybrid=use_hybrid,
                    exclude_ref=exclude_ref,
                    incoming_meta=incoming_meta,
                    rrf_k=rrf_k,
                    w_dense=w_dense,
                    w_bm25=w_bm25,
                    boost_cat=boost_cat,
                    boost_int=boost_int,
                    use_mmr=use_mmr,
                    mmr_lambda=mmr_lambda,
                    debug_timing=True,

                )
                # ensure flat primitives only
                timing.update({
                    "incoming_reference": str(incoming_ref),
                    "incoming_dense_len": int(len(dense_text)),
                    "incoming_lex_len": int(len(lex_text)),
                    "embed_missing": bool(qvec is None),
                })
                timings.append(timing)
            else:
                hits = self.retrieve_one(
                    dense_text=dense_text,
                    qvec=qvec,
                    lex_text=lex_text,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    bm25_candidate_k=bm25_candidate_k,
                    filters=filters,
                    use_hybrid=use_hybrid,
                    exclude_ref=exclude_ref,
                    incoming_meta=incoming_meta,
                    rrf_k=rrf_k,
                    w_dense=w_dense,
                    w_bm25=w_bm25,
                    boost_cat=boost_cat,
                    boost_int=boost_int,
                    use_mmr=use_mmr,
                    mmr_lambda=mmr_lambda,
                    debug_timing=False,
                    explain_bm25=explain_bm25,
                    explain_bm25_topn=explain_bm25_topn,
                    explain_bm25_group=explain_bm25_group
                )

            for rank, h in enumerate(hits, start=1):
                rows.append({
                    "incoming_reference": incoming_ref,
                    "incoming_title": r.get("Title", ""),
                    "incoming_text": r.get("Text", ""),
                    "incoming_intent": r.get("Intent", ""),
                    "incoming_category": r.get("Category", ""),
                    "rank": rank,
                    "incoming_dense_input": r.get("dense_text", ""),
                    "incoming_intent": r.get("Intent", ""),
                    "incoming_category": r.get("Category", ""),
                    "incoming_reason": r.get("Reason", ""),
                    "incoming_action": r.get("Action", ""),
                    "incoming_system": r.get("System", ""),
                    "incoming_rubric": r.get("Rubric", ""),
                    "incoming_FAL": r.get("FAL", ""),
                    "incoming_match_code_present": r.get("has_match_code_header", ""),
                    #debug cols
                    "incoming_lex_input": r.get("incoming_lex_text", ""),
                    "incoming_lex_tokens": " ".join(bm25_tokenize(r.get("incoming_lex_text", "") or ""))[:2000],
                    **h,
                })

        res_df = pd.DataFrame(rows)

        emb_df = None
        if return_query_embeddings:
            emb_df = pd.DataFrame(emb_rows)

        if debug_timing:
            timing_df = pd.DataFrame(timings)
            if return_query_embeddings:
                return res_df, timing_df, emb_df
            return res_df, timing_df

        if return_query_embeddings:
            return res_df, emb_df

        return res_df
