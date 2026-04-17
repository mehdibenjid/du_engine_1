import logging
import os
import time
import requests
import jwt  # Requires 'PyJWT' package

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SSO_URL = "https://ssobroker-val.airbus.com:10443/as/token.oauth2"
CLIENT_ID = "CLI_CD8C_MODELOPS-M2M-REST-EC-V"
SCOPE = "SCO_CD8C_MODELOPS-V"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_FILE = os.path.join(BASE_DIR, "cert.pem")
KEY_FILE = os.path.join(BASE_DIR, "private.key")

# Global variable to cache the token in memory while the script runs
_TOKEN_CACHE = {
    "access_token": None,
    "token_exp": 0
}

def get_access_token():
    """Authenticates via mTLS to get an OAuth2 token."""
    logger.debug(f"Initiating mTLS request to {SSO_URL}")

    if not os.path.exists(CERT_FILE) or not os.path.exists(KEY_FILE):
        logger.error(f"Certificate files missing. Expected at: {CERT_FILE}")
        return None

    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "scope": SCOPE
    }

    try:
        response = requests.post(
            SSO_URL,
            data=payload,
            cert=(CERT_FILE, KEY_FILE),
            verify=False
        )
        response.raise_for_status()
        return response.json().get("access_token")

    except requests.exceptions.RequestException as e:
        logger.error(f"Authentication failed: {e}")
        return None

def get_valid_token():
    """Returns a valid access token, refreshing if necessary."""
    SAFETY_BUFFER = 120 # Seconds
    current_time = time.time()

    # 1. Check Cache
    if _TOKEN_CACHE["access_token"] and _TOKEN_CACHE["token_exp"] > (current_time + SAFETY_BUFFER):
        return _TOKEN_CACHE["access_token"]

    # 2. Refresh Token
    logger.info("Token missing or expiring. Refreshing...")
    new_token = get_access_token()

    if new_token:
        try:
            # Decode without verifying signature (API handles verification)
            decoded = jwt.decode(new_token, options={"verify_signature": False})
            exp_timestamp = decoded.get("exp", current_time + 3600)

            # Update Cache
            _TOKEN_CACHE["access_token"] = new_token
            _TOKEN_CACHE["token_exp"] = exp_timestamp
            
            logger.debug(f"Token refreshed. Expires at {exp_timestamp}")
            return new_token
        except jwt.DecodeError:
            logger.error("Failed to decode JWT response.")
            return None
    
    return None
