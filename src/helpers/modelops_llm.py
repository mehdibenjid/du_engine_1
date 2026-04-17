"""
Nova Pro LLM wrapper for the Airbus ModelOps endpoint.

API URL and namespace are read from config/api.toml. All call parameters
(model_id, tokens, temperature, …) remain as function arguments so callers
can override them per-request.
"""
import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3

from config import load_config
from helpers.auth import get_valid_token

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

_cfg = load_config()
_API_URL  = _cfg.api.base_url
_NAMESPACE = _cfg.api.namespace


def _extract_text_from_nova_response(data: Any) -> str:
    """
    Best-effort extraction of generated text from a Nova Pro response.
    Handles the Bedrock-like schema and common fallback structures.
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data

    # Bedrock-like: {"output":{"message":{"content":[{"text":"..."}]}}}
    try:
        out = data.get("output", {})
        msg = out.get("message", {})
        content = msg.get("content", [])
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
    except Exception:
        pass

    # Fallback: {"messages":[{"role":"assistant","content":[{"text":"..."}]}]}
    try:
        msgs = data.get("messages", [])
        if isinstance(msgs, list):
            for m in reversed(msgs):
                if isinstance(m, dict) and m.get("role") in ("assistant", "model"):
                    cont = m.get("content", [])
                    if isinstance(cont, list) and cont:
                        c0 = cont[0]
                        if isinstance(c0, dict) and "text" in c0:
                            return str(c0["text"])
    except Exception:
        pass

    # Last resort: recursive search for any "text" key
    def _walk(obj):
        if isinstance(obj, dict):
            if "text" in obj and isinstance(obj["text"], str):
                return obj["text"]
            for v in obj.values():
                r = _walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = _walk(v)
                if r:
                    return r
        return ""

    return _walk(data) or ""


def call_nova_pro(
    user_text: str,
    system_text: Optional[str] = None,
    model_id: str = None,
    max_new_tokens: int = 700,
    temperature: float = 0.1,
    timeout: int = 60,
    retries: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """
    Calls Nova Pro via the ModelOps endpoint with exponential back-off.

    Args:
        user_text:      The user turn of the prompt.
        system_text:    Optional system prompt (sent as top-level key).
        model_id:       Model identifier; defaults to config value.
        max_new_tokens: Max tokens to generate.
        temperature:    Sampling temperature.
        timeout:        Per-request timeout in seconds.
        retries:        Number of retry attempts after failure.

    Returns:
        (generated_text, raw_response_json)
    """
    if model_id is None:
        model_id = _cfg.llm.model_id

    token = get_valid_token()

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "model_id": model_id,
        "namespace": _NAMESPACE,
        "converse_mode": "False",
    }

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"text": user_text}]}
    ]

    payload: Dict[str, Any] = {
        "messages": messages,
        "inferenceConfig": {
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
        },
    }

    if system_text:
        payload["system"] = [{"text": system_text}]

    session = requests.Session()
    last_err = None

    for attempt in range(retries + 1):
        try:
            resp = session.post(_API_URL, headers=headers, json=payload, verify=False, timeout=timeout)

            if resp.status_code == 401:
                token = get_valid_token()
                headers["Authorization"] = f"Bearer {token}"
                resp = session.post(_API_URL, headers=headers, json=payload, verify=False, timeout=timeout)

            if resp.status_code >= 400:
                last_err = f"HTTP_{resp.status_code}: {resp.text[:500]}"
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning("API error %s. Retrying in %.2fs…", resp.status_code, sleep_time)
                time.sleep(sleep_time)
                continue

            data = resp.json()
            txt = _extract_text_from_nova_response(data)
            return txt, data

        except Exception as e:
            last_err = f"exc_{type(e).__name__}"
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning("Network exception %s. Retrying in %.2fs…", e, sleep_time)
            time.sleep(sleep_time)
            continue

    raise RuntimeError(f"Nova call failed after {retries} retries. Last error: {last_err}")


def safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Attempts to parse a JSON object from LLM response text.
    Falls back to searching for the first { ... } substring if direct parsing fails.
    Returns an empty dict on failure.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except Exception:
                pass
    return {}


_LANG_INSTRUCTION = {
    "fr": "Réponds UNIQUEMENT en français. Tous les champs texte (label, summary, keywords, etc.) doivent être rédigés en français.",
    "en": "Respond in English only. All text fields (label, summary, keywords, etc.) must be written in English.",
}


def build_cluster_prompt(
    cluster_id: int,
    cluster_size: int,
    coherence: float,
    meta_dist: Dict[str, Any],
    reps: List[tuple],
    language: str = "fr",
) -> tuple:
    """
    Builds the (system_prompt, user_prompt_json) tuple for Nova Pro cluster summarization.

    Args:
        cluster_id:   Numeric cluster identifier.
        cluster_size: Number of documents in the cluster.
        coherence:    Mean cosine similarity to medoid (quality signal).
        meta_dist:    Top-N metadata distributions per field.
        reps:         List of (reference, text) tuples for representative documents.
        language:     Output language for the LLM response ("fr" or "en").

    Returns:
        (system_str, user_json_str)
    """
    lang_instr = _LANG_INSTRUCTION.get(language.lower(), _LANG_INSTRUCTION["fr"])
    system = (
        f"You are a technical analyst. Return VALID JSON ONLY. No markdown, no backticks, no extra text. "
        f"{lang_instr}"
    )
    user = {
        "task": "Summarize a cluster of maintenance requests (DUs).",
        "requirements": {
            "output_json_schema": {
                "label": "short label (max 8 words)",
                "summary": "5-8 lines, concrete, no fluff",
                "keywords": "list of 5-10 keywords",
                "common_systems": "list",
                "common_actions": "list",
                "routing_suggestion": "short suggestion for categorization or routing",
                "llm_confidence": "number from 0 to 1",
                "outliers_or_notes": "short string (optional)",
            },
            "rules": [
                "Use metadata distribution as factual proportions; do NOT invent percentages.",
                "Base the summary primarily on representatives texts.",
                "If the cluster is heterogeneous or low coherence, say so in outliers_or_notes and lower llm_confidence.",
            ],
        },
        "cluster_context": {
            "cluster_id": int(cluster_id),
            "cluster_size": int(cluster_size),
            "coherence_mean_cosine_to_medoid": float(coherence) if coherence == coherence else None,
            "metadata_distribution_top": meta_dist,
        },
        "representatives": [{"reference": ref, "text": txt} for ref, txt in reps],
    }
    return system, json.dumps(user, ensure_ascii=False)
