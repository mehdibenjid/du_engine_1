import logging

import pandas as pd
import re

logger = logging.getLogger(__name__)

def clean_categories(df, column_name='Category'):
    replacements = {
        "FOD": "FOD (Dommage/Débris corps étranger)",
        "FOD (Dommage ou débris par corps étranger)": "FOD (Dommage/Débris corps étranger)",
        "Choisissez d'abord la catégorie par défaut: FOD": "FOD (Dommage/Débris corps étranger)",
        "Impact": "Impact (Structure/Fuselage)",
        "Impact sur fuselage": "Impact (Structure/Fuselage)",
        "Choisissez d'abord la catégorie du défaut:": "Non spécifié",
        "Choisissez d'abord la catégorie par défaut: ": "Non spécifié",
        "Installation Electrique": "Installation Électrique",
        "Installation électrique": "Installation Électrique",
        ": Installation Électrique": "Installation Électrique",
        "Électrique": "Installation Électrique",
        "connector": "Installation Électrique",
        "DEU-B": "Installation Électrique",
        "PSU": "Installation Électrique",
        "Installation Mecanique": "Installation Mécanique",
        "Installation incorrecte": "Installation Mécanique",
        "roue 3": "Installation Mécanique",
        "bouteille EXT": "Installation Mécanique",
        "Fonctionnelle Stamp violé carte LP du 70VU": "Fonctionnelle",
    }
    # Gestion sécurisée si la colonne contient des NaN
    clean_col = df[column_name].fillna('Non spécifié').astype(str).str.strip()
    clean_col = clean_col.replace(replacements)
    clean_col = clean_col.replace(['nan', 'None', '', 'X'], 'Non spécifié')
    return clean_col

def clean_reason_code(text):
    if pd.isna(text) or str(text).strip() == "": return "Non spécifié"
    text = str(text)
    # Nettoyage Technique
    patterns = [r'(PN|P/N|PART|FIN|S/N|SN|CMS|QSR)\s?[:/]?\s?[A-Z0-9-]{4,}', r'\b(?=\w*\d)[A-Z0-9-]{7,}\b', r'[+"]']
    for p in patterns: text = re.sub(p, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    if text.upper() == "X" or text == "": return "Non spécifié"
    text = re.sub(r'\s*[:.,]\s*$', '', text).replace('()', '').strip()
    
    # Regroupement Sémantique
    if "FOD" in text: return "FOD (Dommage/Débris corps étranger)"
    if "Design update" in text: return "Mise à jour de la conception"
    if "FOT" in text: return "FAL Operational Test Meeting"
    if "Software update" in text or "Mise à jour du logiciel" in text: return "Mise à jour du logiciel"
    if "Troubleshooting" in text: return "Dépannage / Recherche de panne"
    if "OSW" in text: return "OSW (travaux restants)"
    if text.startswith("Installation incorrecte"): return "Installation incorrecte"
    if "Défauts visuels sur les pièces" in text: return "Défauts visuels sur les pièces"
    if "rayé, impacté, cassé" in text: return "Élément rayé, impacté, cassé"
    if "Pin" in text and ("tordue" in text or "reculée" in text or "mauvais" in text): return "Défaut connectique (Pin tordue/reculée)"
    if "Stamp" in text or "Torquage" in text: return "Problème de Stamp ou Torquage"
    return text

def clean_rubric(val):
    val = str(val).strip()
    if "VN/VG/VT" in val: val = "Intervention Systèmes Avioniques & Structure (VN/VG/VT)"
    if "Autres élément électrique" in val: val = "Éléments électriques (diodes/capteurs ou autres)"
    if val == "Autres": return "Non spécifié"
    top_values = ["Prises", "Éléments électriques (diodes/capteurs ou autres)", "Tuyauteries", "Equipements rackés", "Panneaux", "Chantier / Elément complet", "Ajout, Remplacement, câbles coupés", "Ajout, Remplacement, Réparation de harnais/gaines", "Intervention Systèmes Avioniques & Structure (VN/VG/VT)"]
    return val if val in top_values else "Non spécifié"

def clean_plug_action(val):
    if pd.isna(val) or str(val).strip() == "": return "Non spécifié"
    val = str(val).strip()
    if re.search(r'(débranch|rebranch|connect|déconnect)', val, re.IGNORECASE): return "Déconnexion / Reconnexion (Prise/Harnais)"
    if "dépinouillée" in val or "depinouillée" in val: return "Prise dépinouillée"
    if "remplace" in val.lower() or "REMPLACEMENT" in val or "changé" in val: return "Remplacement (Prise/Équipement)"
    if re.search(r'(cassée|impact|rayée|polluée|abimée)', val, re.IGNORECASE): return "Prise endommagée (Cassée/Impactée/Polluée)"
    if "Stamp" in val: return "Problème de Stamp (Déchiré/Manquant)"
    return "Non spécifié"

def clean_cabling_reason(val):
    if pd.isna(val) or str(val).strip() == "": return "Non spécifié"
    val = str(val).strip()
    if "Ajout" in val: return "Ajout de câble / harnais"
    if re.search(r'(harnais|gaine|manchon)', val, re.IGNORECASE): return "Harnais : Coupé / Remplacé / Réparé"
    if re.search(r'c[âa]ble', val, re.IGNORECASE): return "Câble : Coupé / Remplacé / Réparé"
    if "remplacé" in val.lower(): return "Câble : Coupé / Remplacé / Réparé"
    return "Non spécifié"

def clean_action_description(val):
    if pd.isna(val): return "Non spécifié"
    val = str(val).strip()
    val_lower = val.lower()
    if val in [".", "/", "-", "--", "----", "------", "NON", "NEANT"]: return "Non spécifié"
    if "pas de" in val_lower or "n/a" in val_lower or "aucun" in val_lower: return "Aucune action / Non applicable"
    if "remplace" in val_lower or "changé" in val_lower or "maj" in val_lower: return "Remplacement"
    if "dépose" in val_lower or "pose" in val_lower or "rentrée/sortie" in val_lower: return "Dépose / Pose"
    if "torquage" in val_lower or "serrage" in val_lower: return "Serrage / Torquage"
    if "réglage" in val_lower or "mise à niveau" in val_lower or "rework" in val_lower: return "Réglage / Ajustement"
    if "fuite" in val_lower or "test" in val_lower or "contrôle" in val_lower or "dépannage" in val_lower or "visite" in val_lower: return "Inspection / Diagnostic"
    return "Non spécifié"

def clean_fal(val):
    if pd.isna(val) or str(val).strip() == "": return "Non spécifié"
    val = str(val).strip()
    if re.search(r'A320.*Martin', val, re.IGNORECASE): return "A320 (St Martin)"
    if re.search(r'A321.*Lagard', val, re.IGNORECASE): return "A321 (Lagardère)"
    if re.search(r'A320.*Lagard', val, re.IGNORECASE): return "A320 (Lagardère)"
    return "Non spécifié"

def clean_tank_drained(val):
    if pd.isna(val) or str(val).strip() == "": return "Non spécifié"
    val = str(val).strip().upper()
    if val in ["YES", "Y", "OUI", "TRUE", "1"]: return "Oui"
    if val in ["NO", "N", "NON", "FALSE", "0"]: return "Non"
    return "Non spécifié"

def clean_circuit(val):
    if pd.isna(val) or str(val).strip() in ["", ".", "------", "/"]: return "Non spécifié"
    val = str(val).lower()
    if "tous" in val: return "Tous (Vert + Jaune + Bleu)"
    has_vert = "vert" in val
    has_jaune = "jaune" in val or "yellow" in val
    has_bleu = "bleu" in val or "blue" in val
    count = sum([has_vert, has_jaune, has_bleu])
    if count == 3: return "Tous (Vert + Jaune + Bleu)"
    elif count == 2:
        if has_vert and has_jaune: return "Vert + Jaune"
        if has_vert and has_bleu:  return "Vert + Bleu"
        if has_jaune and has_bleu: return "Jaune + Bleu"
    elif count == 1:
        if has_vert: return "Vert"
        if has_jaune: return "Jaune"
        if has_bleu: return "Bleu"
    return "Non spécifié"

def clean_side(val):
    if pd.isna(val) or str(val).strip() == "": return "Non spécifié"
    val = str(val).upper().strip()
    if "BOTH" in val or ("LHS" in val and "RHS" in val): return "Les deux (Both)"
    if re.search(r'(AXE|AXIS|CENTR|CTR|Y0)', val): return "Centre (Axe)"
    if "LHS" in val or "LEFT" in val or "MOTEUR 1" in val: return "Gauche (LHS)"
    if "RHS" in val or "RIGHT" in val or "MOTEUR 2" in val: return "Droite (RHS)"
    return "Non spécifié"

def clean_equip_downloadable(val):
    if pd.isna(val) or str(val).strip() == "": return "Non spécifié"
    val = str(val).strip().upper()
    if val in ["YES", "Y", "OUI", "TRUE", "1"]: return "Oui"
    if val in ["NO", "N", "NON", "FALSE", "0"]: return "Non"
    return "Non spécifié"

def clean_keyword(val):
    if pd.isna(val) or str(val).strip() in ["", "X", ".", "-"]: return "Non spécifié"
    val = str(val).strip().lower()
    # Process
    if "fot" in val: return "Processus : FOT (FAL Operational Test Meeting)"
    if "storage" in val: return "Processus : Stockage Avion"
    # Cabine
    if "hatrack" in val or "coffre" in val: return "Cabine : Coffre à bagages (Hatrack)"
    if "ceiling" in val or "plafond" in val: return "Cabine : Panneau Plafond"
    if "lining" in val or "linning" in val or "habillage" in val or "scuff" in val: return "Cabine : Habillage / Lining"
    if "floor" in val or "plancher" in val or "carpet" in val or "moquette" in val: return "Cabine : Plancher / Moquette"
    if "sidewall" in val or "partition" in val or "cloison" in val: return "Cabine : Cloison / Panneau Latéral"
    if "galley" in val or "cuisine" in val: return "Cabine : Galley (Cuisine)"
    if "lavatory" in val or "toilettes" in val or "wc" in val: return "Cabine : Toilettes (Lavatory)"
    if "seat" in val or "siège" in val or "psu" in val: return "Cabine : Siège / PSU"
    if "stowage" in val or "rangement" in val: return "Cabine : Meuble de rangement (Stowage)"
    # Systèmes
    if "pressu" in val: return "Système : Pressurisation"
    if "oxygen" in val or "o²" in val or "masque" in val: return "Système : Oxygène"
    if "apu" in val: return "Système : APU"
    if "rat" in val: return "Système : RAT"
    if "bonding" in val or "masse" in val or "metallisation" in val: return "Système : Métallisation / Tresse de masse"
    if "hydrau" in val or "leak" in val or "fuite" in val: return "Système : Hydraulique / Fuite"
    # Structure
    if "door" in val or "porte" in val or "trappe" in val or "nlg" in val or "mlg" in val: return "Structure : Porte / Trappe"
    if "dent" in val or "skin" in val or "impact" in val or "fuselage" in val or "extrados" in val: return "Structure : Impact Fuselage / Peau"
    if "slat" in val or "bec" in val or "volet" in val or "flap" in val or "spoiler" in val or "wing" in val or "aile" in val or "fairing" in val: return "Structure : Aile / Voilure (Slat/Flap)"
    # Clean simple
    val = re.sub(r'\b[a-z0-9]{5,}\b', '', val) 
    final_val = val.title().strip()

    if final_val == "":
        return "Non spécifié"
        
    return final_val

def clean_system(val):
    if pd.isna(val) or str(val).strip() in ["", ".", "Autres Cote: Les deux", "Autres Cote: Gauche"]: return "Non spécifié"
    val = str(val).strip()
    val_lower = val.lower()
    if "hydr" in val_lower or "hydro" in val_lower or "frein" in val_lower or "brake" in val_lower or "landing gear" in val_lower or "mlg" in val_lower: return "Hydraulique"
    if "air" in val_lower or "bleed" in val_lower or "oxy" in val_lower or "dégivrage" in val_lower: return "Air"
    if "fuel" in val_lower: return "Fuel"
    if "eau" in val_lower: return "Eau"
    if "structure" in val_lower: return "Structure"
    if any(x in val_lower for x in ['apu', 'eng', 'reverse', 'sidestick', 'cdve', 'atc', 'emer']): return "Systèmes Avioniques / Moteur"
    return "Non spécifié"


def clean_narrative(text):
    """
    Nettoyage V11 (Ultimate) :
    - Gère les tournures passives ("il a été constaté", "ont été relevés").
    - Gère les verbes de constat actifs ("nous avons remarqué").
    - Gère Dates, Politesse, Admin, Encodage.
    """
    if pd.isna(text) or str(text).strip() in ["", ".", "nan", '""', "null"]:
        return "" 

    text = str(text).strip()

    # --- ÉTAPE 0 : Réparation Encodage & Normalisation ---
    replacements = {
        "¨¦": "é", "¨¤": "à", "O²": "O2", "//": ",", " - ": ", ", 
        "``": "", "`": "", '"': ""
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    
    text = re.sub(r'(?<=[a-zA-Z0-9])\.(?=[a-zA-Z])', '. ', text)

    # --- ÉTAPE 1 : Suppression du Wrapper ---
    text = re.sub(r"^\(?(?:compl.{1,2}ment\s+d['’\s]?informations?)\)?\s*[:\.-]?\s*", "", text, flags=re.IGNORECASE)

    # --- ÉTAPE 2 : Suppression RADICALE des Contacts ---
    text = re.sub(r"(Num[ée]ro(\s+de\s+téléphone)?(\s+pro)?|Tél|Tel|Contact)\s*[:\.]?.*", "", text, flags=re.IGNORECASE)

    # --- ÉTAPE 3 : Suppression GLOBALE de la Politesse ---
    politesse_global = [
        r"\b(Bonjours?|Bonsoirs?|kéBonjour|Salut|Adieu)\b",
        r"\b(Bonne journ[ée]e?|Bonne apr[èe]s-midi|Bonne soir[ée]e?)\b",
        r"\b(Cordialement|Cdt|Cdlt|Salutations|Bien à vous|Svp|Merci)\b",
        r"\b(D'avance merci|Merci d'avance|En vous remerciant| En vous remerciant, à vous)\b"
    ]
    for p in politesse_global:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # --- ÉTAPE 4 : Suppression des "Tics de Langage" & Constats (MISE A JOUR) ---
    fillers = [
        # Formules de politesse/reporting
        r"\bnous\s+vous\s+(?:signalons|informons|prions|demandons|indiquons)\b",
        r"\bnous\s+(?:signalons|informons|notons)\b",
        r"\bnous\s+vous\b",
        r"\bsignalons\s+(?:que)?\b",
        r"\bnotons\s+(?:que)?\b",
        r"\bil\s+est\s+à\s+noter\s+(?:que)?\b",
        
        # Verbes de constatation (ACTIF) : "(Nous) avons constaté"
        r"\b(?:nous\s+)?(?:avons|avions)\s+(?:constat[ée]s?|remarqu[ée]s?|observ[ée]s?|trouv[ée]s?|détect[ée]s?|not[ée]s?|relev[ée]s?|vu)\b",
        r"\b(?:nous\s+)?(?:constatons|remarquons|observons|trouvons|détectons|notons|relevons)\b",
        r"\bavons\s+(?:constat[ée]s?|remarqu[ée]s?|observ[ée]s?|trouv[ée]s?|détect[ée]s?|not[ée]s?|relev[ée]s?|vu)\b",

        # Verbes de constatation (PASSIF) : "Il a été constaté", "Ont été relevés"
        # Regex : (Il)? (a/ont) été (verbe)
        r"\b(?:il\s+)?(?:a|ont)\s+été\s+(?:constat[ée]s?|remarqu[ée]s?|observ[ée]s?|trouv[ée]s?|détect[ée]s?|not[ée]s?|relev[ée]s?|vu)\b"
    ]
    for f in fillers:
        text = re.sub(f, "", text, flags=re.IGNORECASE)

    # Nettoyage du "que" résiduel (ex: "Il a été constaté que..." -> "que..." -> "...")
    text = re.sub(r"\bque\s+", "", text, flags=re.IGNORECASE)

    # --- ÉTAPE 5 : Suppression du "Chat Admin" ---
    patterns_end = [
        r"\b(Demandons|Demande|Veuillez|Prière de|Pouvez-vous|Pourriez-vous)\s+.*?(conduite|reprise|marche|démarche|suite|procédure|RTI|essai|action|consigne|génération).*",
        r"\bde (nous|me) (fournir|donner|rédiger|indiquer|créer|transmettre|communiquer).*?(reprise|RTI|essai|conduite|marche).*",
        r"pour votre prise en compte.*",
        r"Important\s*:.*"
    ]
    for p in patterns_end:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    # --- ÉTAPE 6 : LA BOUCLE DE NETTOYAGE ---
    start_patterns = [
        r"^\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?(?:\s*(?:à|at|@)?\s*\d{1,2}[:hH]\d{2})?[\s\.,-]*", # Dates
        r"^FOT planifi[ée]e?.*?(?:\d{1,2}[hH]\d{2})?\.?\s*", # FOT
        r"^(?:Bonjour|Bonsoir|Salut|Merci|Cordialement|Cdt|Cdlt|Svp|Salutations|Bien à vous)\s*[,.]?\s*", # Politesse
        r"^(?:Suite|Faisant suite|Relatif|Concernant|Lors|En vue|Pour|Dans le cadre|Au vu|A la demande|Suivant|Selon)\s+(?:.{1,4}\s+)?", # Contexte
        r"^(?:Le|La|Les|Un|Une|Des|Du|Au|L'|D')\s+", # Déterminants
        r"^[\W_]+\s*" # Symboles
    ]
    
    combined_start_regex = "|".join(start_patterns)
    
    cleaned = text
    while True:
        prev = cleaned
        cleaned = re.sub(combined_start_regex, "", prev, flags=re.IGNORECASE)
        if cleaned == prev:
            break
    text = cleaned

    # --- ÉTAPE 7 : Standardisation Technique ---
    text = re.sub(r"\b(NC|IC|GTI|OF|SO|WO|DU|FI|QLB)\s*[-_]?\s*(\d+)", r"\1\2", text, flags=re.IGNORECASE)
    
    # --- ÉTAPE 8 : Lissage Final ---
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*([,.-])\s*', r'\1 ', text)
    text = re.sub(r'[,.-]\s*[,.-]', ', ', text)
    text = re.sub(r'^[,.-]\s*', '', text)
    text = re.sub(r'[,.-]\s*$', '', text)
    
    # --- ÉTAPE 9 : Filtre "Texte Vide" ---
    if len(text) < 3 and not re.search(r'\d', text):
        return ""

    return text.strip()

def clean_title(text):
    """
    Nettoyage RADICAL pour TITLE.
    - Supprime les blocs de tags (Match Codes) où qu'ils soient (début, milieu, fin).
    - Supprime les préfixes administratifs.
    """
    if pd.isna(text) or str(text).strip() in ["", ".", "nan", '""', "null"]:
        return ""
    
    text = str(text).strip()

    start_patterns = [
        r"^\d{1,2}[/.-]\d{1,2}(?:[/.-]\d{2,4})?(?:\s*(?:à|at|@)?\s*\d{1,2}[:hH]\d{2})?[\s\.,-]*", # Dates
        r"^FOT planifi[ée]e?.*?(?:\d{1,2}[hH]\d{2})?\.?\s*", # FOT
        r"^(?:Bonjour|Bonsoir|Salut|Merci|Cordialement|Cdt|Cdlt|Svp|Salutations|Bien à vous)\s*[,.]?\s*", # Politesse
        r"^(?:Suite|Faisant suite|Relatif|Concernant|Lors|En vue|Pour|Dans le cadre|Au vu|A la demande|Suivant|Selon)\s+(?:.{1,4}\s+)?", # Contexte
        r"^(?:Le|La|Les|Un|Une|Des|Du|Au|L'|D')\s+", # Déterminants
        r"^[\W_]+\s*" # Symboles
    ]
    
    combined_start_regex = "|".join(start_patterns)
    cleaned = text
    while True:
        prev = cleaned
        cleaned = re.sub(combined_start_regex, "", prev, flags=re.IGNORECASE)
        if cleaned == prev:
            break
    text = cleaned

    # 1. Encodage
    replacements = {"¨¦": "é", "¨¤": "à", "O²": "O2", "//": ",", " - ": ", ", "``": "", "`": "", '"': ""}
    for bad, good in replacements.items(): text = text.replace(bad, good)

    # 2. SUPPRESSION CHIRURGICALE DES TAGS (Match Codes)
    # Regex : #[... followed by any #content# sequences until the last #
    # Ex: "Texte #[nf]#nc#123# fin" -> "Texte  fin"
    text = re.sub(r"\#\[.*?\](?:#[^#]*)*#", "", text, flags=re.IGNORECASE)

    # 3. Préfixes & Actions
    text = re.sub(r"^(?:WO|OF|NC|FI|QSR|Man|ALB|QLB)[_\s\.]*[\d-]+\s*(?:[-_:\.]\s*)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Dep/Rep|Depose/Repose|Depose / Repose|Dép/rep)\b", "Dépose/Repose", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Dec/Rec|Deconnexion/Reconnexion|Deconnexion / Reconnexion)\b", "Déconnexion/Reconnexion", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:Suite (?:à|au|aux)|Concernant|Objet\s*:)\s+", "", text, flags=re.IGNORECASE)
    
    # 4. Nettoyage final (Espaces multiples créés par la suppression)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^[,.-]\s*', '', text)
    
    return text.capitalize()


def clean_extracted_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies column-specific cleaning functions to a DataFrame of extracted features.
    Operates on the standard post-extraction column set (Title, Narrative, Category, etc.).
    """
    cleaning_map = {
        "Category": clean_categories,
        "Reason_Code": clean_reason_code,
        "Rubric": clean_rubric,
        "Plug_Action": clean_plug_action,
        "Cabling_Reason": clean_cabling_reason,
        "Action_Description": clean_action_description,
        "FAL": clean_fal,
        "Tank_Drained": clean_tank_drained,
        "Circuit": clean_circuit,
        "Side": clean_side,
        "Equip_Downloadable": clean_equip_downloadable,
        "Header_Keyword": clean_keyword,
        "Body_Keyword": clean_keyword,
        "System": clean_system,
        "Narrative": clean_narrative,
        "Title": clean_title,
    }
    for col, func in cleaning_map.items():
        if col not in df.columns:
            continue
        logger.debug("Cleaning column: %s", col)
        df[col] = func(df, col) if col == "Category" else df[col].apply(func)
    return df
