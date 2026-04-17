import re
from typing import Any, Dict, List, Optional

import pandas as pd

# ==========================================
# CONSTANTS
# ==========================================
EMPTY = {"", "N/A", "---", "nil", "RAS", "***", "&", "*", "//", "NA", "..."}
PHONE_RE = re.compile(r"\b0[1-9](?:[.\s-]?\d{2}){4}\b", re.I)
PREFIX_RE = re.compile(r"^#\[(?P<code>[A-Z]{2})\]", re.I)
LIST_DETECT_RE = re.compile(r"[\/,;&+]|\b(?:ET|AND)\b", re.I)

# ==========================================
# HELPERS
# ==========================================
def clean_value(val: str) -> str:
    val = (val or "").strip()
    val = re.sub(r"\[[^\]]+\]", "", val)
    val = re.sub(r"\s+", " ", val).strip()
    return val

def is_empty(val: Optional[str]) -> bool:
    if val is None: return True
    return clean_value(val) in EMPTY

def is_complex_list(val: str) -> bool:
    if not val: return False
    val_check = re.sub(r"P/N", "PN", val, flags=re.I)
    return bool(LIST_DETECT_RE.search(val_check))

def norm_id(val: str) -> str:
    val = re.sub(r"[^A-Z0-9\-]", "", (val or "").upper())
    return val.strip("-")

def norm_code_list(val: str) -> str:
    if not val: return ""
    s = val.upper()
    s = re.sub(r"\bP/N\b", "PN", s)
    raw_tokens = re.split(r"[\/,;&+]|\s+(?:ET|AND)\s+", s)
    out = []
    seen = set()
    for t in raw_tokens:
        t_clean = norm_id(t)
        if len(t_clean) >= 2 and t_clean not in seen:
            if any(ch.isdigit() for ch in t_clean) or len(t_clean) > 2:
                seen.add(t_clean)
                out.append(t_clean)
    return " | ".join(out)

def norm_cable(val: str) -> str:
    return re.sub(r"[^A-Z0-9\-]", "", (val or "").upper()).strip("-")

# ==========================================
# MAIN EXTRACTION LOGIC
# ==========================================
def extract_features(text: str) -> Dict[str, Any]:
    row_data = {}
    if not isinstance(text, str): return row_data

    m_prefix = PREFIX_RE.match(text)
    if not m_prefix: return row_data

    code = m_prefix.group("code").upper()
    row_data["Log_Code"] = code
    raw = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    # 1. INTENT MAPPING
    INTENT_MAP = {
        "CA": "Signalement maintenance: Attestation GTI ou RTI incorrecte.",
        "CD": "Action technique: Mise à jour de la conception (Design Update).",
        "CI": "Action maintenance: Procédure d'inspection standard.",
        "CO": "Logistique: Procédure de stockage ou déstockage.",
        "CR": "Documentation technique: Mise à jour du GTR.",
        "CT": "Action technique: Dépannage et recherche de panne.",
        "CW": "Statut opérationnel: OSW (travaux restants) ou FOT reporté.",
        "NB": "Défaut physique: Scellé de sécurité rompu ou brisé.",
        "ND": "Problème système: Problème de Delta P (différentiel de pression).",
        "NF": "Défaut système: Défaut fonctionnel détecté.",
        "NI": "Erreur montage: Mauvaise installation d'un composant.",
        "NL": "Défaut physique: Fuite de liquide ou de gaz détectée.",
        "NM": "Matériel: Pièce manquante dans l'assemblage.",
        "NO": "Défaut critique: FOD (Dommage/Débris corps étranger).",
        "NP": "Défaut connectique: Broche ou pin tordue.",
        "NQ": "Qualité: Problème de tampon ou de couple de serrage incorrect.",
        "NS": "Dommage structurel: Impact visible sur le fuselage.",
        "NU": "Système: Mise à jour logicielle requise.",
        "NV": "Inspection: Défaut visuel ou cosmétique constaté.",
        "NW": "Défaut électrique: Fil ou câblage endommagé."
    }
    row_data["Intent"] = INTENT_MAP.get(code, "CHANGE_REQUEST")

    # 2. CATEGORY
    cat_val = None
    m_cat_a = re.search(r"Choisissez\s+d'abord\s+la\s+cat\S*gorie\s+d[ue]\s+d\S*faut\s*:\s*([^:]+?)(?=\s+Type|\s+Mot-?cl\S*|\s+Rubriques|\s+FAL|\s+Texte|\s+FIN|\s+Côté|\s+Dépose|\s+Système|$)", raw, flags=re.I)
    m_cat_b = re.search(r"Type\s+d[ue]\s+d\S*faut(?:\s*\(.*?\))?\s*[:\-]\s*(.*?)(?=\s+Type\s+d[ue]\s+d\S*faut)", raw, flags=re.I)
    
    if m_cat_a: cat_val = clean_value(m_cat_a.group(1))
    elif m_cat_b: cat_val = clean_value(m_cat_b.group(1))

    if cat_val and cat_val not in EMPTY:
        cat_val = cat_val.replace("¨¦", "é").replace("¨¤", "à").replace("¡ú", "->")
        row_data["Category"] = cat_val
    else:
        row_data["Category"] = "Non spécifié"

    # 3. HEADER
    m = re.match(r"^#\[(?P<code>[A-Z]{2})\]#([^#]+)#([^#]+)#([^#]+)#([^#]+)#", raw, flags=re.I)
    if m:
        raw_doc_type = clean_value(m.group(2))
        raw_doc_ref = clean_value(m.group(3))
        split_match = re.match(r"^([A-Z/]+)(\d+)$", raw_doc_type)

        if split_match:
            row_data["Doc_Type"] = split_match.group(1)
            doc_ref_val = split_match.group(2)
        else:
            row_data["Doc_Type"] = raw_doc_type
            doc_ref_val = raw_doc_ref

        if not is_empty(doc_ref_val):
            row_data["Doc_Ref"] = norm_code_list(doc_ref_val) if is_complex_list(doc_ref_val) else norm_id(doc_ref_val)
        else:
            row_data["Doc_Ref"] = "Non spécifié"

        ctx_type = clean_value(m.group(4)).upper()
        ctx_val = clean_value(m.group(5))
        if not is_empty(ctx_val):
            if ctx_type == "PN": row_data["Header_PN"] = norm_id(ctx_val)
            elif ctx_type == "FIN": row_data["Header_FIN"] = norm_code_list(ctx_val) if is_complex_list(ctx_val) else norm_id(ctx_val)
            elif ctx_type == "FD": row_data["Header_Keyword"] = ctx_val

    # 4. BODY
    LABELS = [
        (r"GTI/RTI", "gti_rti"), (r"Work\s+order\s+Number", "wo"),
        (r"Type\s+de\s+défaut\s*\(2\s*letter\s*code\)", "reason"), (r"Type\s+de\s+défaut", "reason"),
        (r"Mot-?clé.*?\(FD\)", "keyword_body"), (r"FAL", "fal"),
        (r"Part\s+Number", "part_number"), (r"Rubriques", "rubriques"),
        (r"Nom\s+ou\s+raison\s+du\s+chantier", "chantier"), (r"Prises\s+déconnectées", "disconnected_plugs"),
        (r"Equipements\s+déposés\s*\(FIN\)", "removed_fins"), (r"Eléments\s+mécaniques\s+déposés\s*\(P/N\)", "_stop_mech"),
        (r"Panneaux\s+déposés", "_stop_panels"), (r"Eléments\s+cabine\s+déposés.*?\)", "_stop_cabin"),
        (r"Autres", "_stop_autres"), (r"Dépose\s*/\s*Pose\s+ou\s+Remplacement", "action"),
        (r"Système", "system"), (r"Circuit", "circuit"), (r"Côté", "side"),
        (r"FIN\s+Number", "fin_number"), (r"Identification\s+ligne", "line_identification"),
        (r"Tenant\s+ou\s+aboutissant\s+de\s+la\s+ligne\s*\(FIN\s*\+\s*port\s+hydraulique\)", "line_endpoints"),
        (r"FIN\s+des\s+éléments", "fin_elements"), (r"Numéro\s+de\s+câble.*?\)", "cable_number"),
        (r"Numéro\s+de\s+cable.*?\)", "cable_number"), (r"Numéro\s+de\s+câble", "cable_number"),
        (r"Numéro\s+de\s+cable", "cable_number"), (r"Précisez\s+la\s+référence\s*\(P/N\)", "pn_ref"),
        (r"Repère\s+mécanique.*?(?:P/N\)|FIN)", "repere"), (r"Zone\s*\(.*?\)", "zone"),
        (r"Précisez\s+la\s+zone", "zone_detail"), (r"Précisez\s+N(?:°|º|o)\s*de\s+panneau", "panel_number"),
        (r"Panneaux\s+ou\s+portes\s+ouvertes.*?élément", "access"), (r"Prises\s*$", "prises_block"),
        (r"Prises", "plug_action"), (r"Liste\s+des\s+FINs\s+impactés.*?\(Prises\)", "impacted_fins"),
        (r"La\s+bâche\s+hydraulique.*?vidée\s*\?", "tank_drained"), (r"Raison\s+Câblage", "cabling_reason"),
        (r"FIN\s*\(.*?\)\s*ou\s*Câble\s*\(.*?\)", "cabling_ref"),
        (r"FIN\s*\(préciser.*?\)\s*ou\s*Câble\s*\(préciser.*?\)", "cabling_ref"),
        (r"Equipements\s+rackés", "racked_equipments"), (r"Equipements\s+téléchargeable\s*\?\s*", "equip_downloadable"),
        (r"Equipements\s+telechargeable\s*\?\s*", "equip_downloadable"),
        (r"FIN\s+des\s+équipements\s+impactés", "impacted_equipment_fins"),
        (r"Précisez\s+si\s+des\s+éléments\s+ont\s+été\s+déposés\s+pour\s+accéder\s+à\s+l'intervention", "access_removed"),
        (r"Texte\s+libre", "narrative"), (r"Numéro\s+de\s+téléphone", "phone"),
    ]

    occurrences = []
    for pat, key in LABELS:
        for mm in re.finditer(rf"({pat})\s*:", raw, flags=re.I):
            occurrences.append((mm.start(), mm.end(), key))
    occurrences.sort(key=lambda x: x[0])

    extracted = {}
    for i, (start, end, key) in enumerate(occurrences):
        next_start = occurrences[i + 1][0] if i + 1 < len(occurrences) else len(raw)
        val = clean_value(raw[end:next_start].strip(" ,;"))
        if key not in extracted and not is_empty(val):
            extracted[key] = val

    # 5. ASSIGN
    if "reason" in extracted: row_data["Reason_Code"] = extracted["reason"]
    if "keyword_body" in extracted: row_data["Body_Keyword"] = extracted["keyword_body"]
    if "fal" in extracted: row_data["FAL"] = extracted["fal"]
    
    raw_pn = extracted.get("part_number") or extracted.get("pn_ref")
    if raw_pn: row_data["Body_PN"] = norm_id(raw_pn)

    if "rubriques" in extracted:
        r = extracted["rubriques"]
        row_data["Rubric"] = clean_value(r.split(":")[-1]) if ":" in r else r

    if "system" in extracted: row_data["System"] = extracted["system"]
    if "action" in extracted: row_data["Action_Description"] = extracted["action"]
    if "narrative" in extracted:
        narrative = extracted["narrative"].strip()
        narrative = re.split(r"Cordialement", narrative, flags=re.I)[0]
        row_data["Narrative"] = PHONE_RE.sub("[CONTACT_INFO]", narrative)

    return row_data


# ==========================================
# FEATURE EXTRACTION UTILITIES
# ==========================================

_HEADER_RE = re.compile(r"^#\[[A-Z]{2}\]#[^#]+#[^#]+#[^#]+#[^#]+#", re.I)
_EMPTY_VALUES = {"non spécifié", "n/a", "na", "none", "null", "unknown"}


def is_empty_value(v) -> bool:
    """
    Returns True if v is None, NaN, empty string, or a known placeholder
    (non spécifié, n/a, none, null, unknown).
    """
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return not str(v).strip() or str(v).strip().lower() in _EMPTY_VALUES


def compute_has_header(text: str) -> bool:
    """Returns True if text begins with a valid #[XX]# structured header."""
    return isinstance(text, str) and bool(_HEADER_RE.match(text.strip()))


def compute_feature_status(has_header: bool, feat: Dict[str, Any]) -> str:
    """
    Returns a quality label for the extraction result:
      no_header | failed | partial | ok
    """
    if not has_header:
        return "no_header"
    if not feat:
        return "failed"
    keys = set(feat.keys())
    has_core = ("Intent" in keys) and (
        ("System" in keys) or ("Reason_Code" in keys) or ("Action_Description" in keys) or ("Category" in keys)
    )
    return "ok" if has_core else "partial"


def extract_features_df(df: pd.DataFrame, text_col: str, has_header_col: str) -> pd.DataFrame:
    """
    Applies extract_features row-wise and appends the resulting columns to df.

    Args:
        df:             Input DataFrame.
        text_col:       Column containing raw text with optional #[XX]# header.
        has_header_col: Boolean column indicating whether the header regex matched.

    Returns:
        DataFrame with extracted feature columns appended.
    """
    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        txt = str(row.get(text_col, "") or "").strip()
        has_header = bool(row.get(has_header_col, False))
        feat = extract_features(txt) if (txt and has_header) else {}
        feat["feature_status"] = compute_feature_status(has_header, feat)
        out_rows.append(feat)
    feat_df = pd.DataFrame(out_rows)
    return pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
