"""
OAuth2 mTLS authentication against the Airbus ModelOps SSO.

All connection parameters are read from config/api.toml via load_config().
The valid token is cached in memory and refreshed automatically before expiry.
"""
import logging
import os
import time

import jwt
import requests

from config import load_config

logger = logging.getLogger(__name__)

# In-memory token cache (module-level singleton).
_TOKEN_CACHE: dict = {
    "access_token": None,
    "token_exp": 0,
}

# Lazy-initialized config cache — avoids reading TOML files at import time,
# which would break test environments that lack a real config directory.
_cfg = None


def _get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def _cert_file() -> str:
    cfg = _get_cfg()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, cfg.auth.cert_file)


def _key_file() -> str:
    cfg = _get_cfg()
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, cfg.auth.key_file)


def get_access_token() -> str | None:
    """Authenticates via mTLS and returns a raw OAuth2 access token."""
    cfg = _get_cfg()
    sso_url = cfg.auth.sso_url
    cert_file = _cert_file()
    key_file = _key_file()

    logger.debug("Initiating mTLS request to %s", sso_url)

    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        logger.error("Certificate files missing. Expected cert=%s  key=%s", cert_file, key_file)
        return None

    payload = {
        "grant_type": "client_credentials",
        "client_id": cfg.auth.client_id,
        "scope": cfg.auth.scope,
    }

    try:
        response = requests.post(
            sso_url,
            data=payload,
            cert=(cert_file, key_file),
            verify=True,  # Always verify the server certificate.
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logger.error("Authentication failed: %s", e)
        return None


def get_valid_token() -> str | None:
    """
    Returns a valid access token, refreshing from the SSO if the cached one
    is missing or within SAFETY_BUFFER seconds of expiry.
    """
    cfg = _get_cfg()
    safety_buffer = cfg.auth.safety_buffer
    current_time = time.time()

    if (
        _TOKEN_CACHE["access_token"]
        and _TOKEN_CACHE["token_exp"] > (current_time + safety_buffer)
    ):
        return _TOKEN_CACHE["access_token"]

    logger.info("Token missing or expiring soon — refreshing.")
    new_token = get_access_token()

    if new_token:
        try:
            decoded = jwt.decode(new_token, options={"verify_signature": False})
            exp_timestamp = decoded.get("exp", current_time + 3600)
        except jwt.DecodeError:
            # Token is valid (SSO issued it) but not a parseable JWT.
            # Cache it with a conservative 1-hour TTL rather than discarding it.
            logger.warning("Token is not a parseable JWT — caching with default 1h TTL.")
            exp_timestamp = current_time + 3600

        _TOKEN_CACHE["access_token"] = new_token
        _TOKEN_CACHE["token_exp"] = exp_timestamp
        logger.debug("Token refreshed. Expires at %s", exp_timestamp)
        return new_token

    return None
