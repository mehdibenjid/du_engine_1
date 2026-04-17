"""
Central configuration loader.

Reads the four TOML files under config/ and exposes a single typed Config
object. Call load_config() once at application startup (main.py or runner.py)
and pass the result down to every step function.
"""
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Project root is two levels up from this file (src/config.py -> src/ -> DU_Engine/)
_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AuthConfig:
    sso_url: str
    client_id: str
    scope: str
    cert_file: str
    key_file: str
    safety_buffer: int


@dataclass
class ApiConfig:
    base_url: str
    namespace: str
    embedding_model_id: str
    embedding_dims: int


@dataclass
class PathsConfig:
    data_dir: str
    raw_dir: str
    processed_dir: str
    transformed_dir: str
    artifacts_dir: str


@dataclass
class ProcessingConfig:
    type_name: str
    match_code: bool
    max_chars: int
    max_ids_to_keep: int
    embedding_strategy: str
    csv_separator: str = ""  # Set to "" to auto-detect from input files


@dataclass
class RetrievalConfig:
    top_k: int
    candidate_k: int
    bm25_candidate_k: int
    rrf_k: int
    w_dense: float
    w_bm25: float
    boost_cat: float
    boost_int: float
    use_mmr: bool
    mmr_lambda: float
    max_workers: int
    timeout: int
    retries: int


@dataclass
class ClusteringConfig:
    use_umap: bool
    umap_n_components: int
    umap_n_neighbors: int
    umap_min_dist: float
    hdbscan_min_cluster_size: int
    hdbscan_min_samples: int
    random_state: int
    rep_closest_n: int
    rep_diverse_n: int
    rep_mmr_lambda: float
    max_text_chars_for_llm: int
    classification_threshold: float = 0.75


@dataclass
class NovaProConfig:
    model_id: str
    max_new_tokens: int
    temperature: float
    timeout: int
    retries: int
    max_workers: int
    max_reps_sent: int
    truncate_chars: int
    language: str = "fr"


@dataclass
class Config:
    auth: AuthConfig
    api: ApiConfig
    paths: PathsConfig
    processing: ProcessingConfig
    retrieval: RetrievalConfig
    clustering: ClusteringConfig
    llm: NovaProConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_dir: Optional[Path] = None) -> Config:
    """
    Loads and merges all four TOML config files into a single Config object.

    Args:
        config_dir: Path to the config directory. Defaults to <project_root>/config.

    Returns:
        A fully populated Config instance.
    """
    if config_dir is None:
        config_dir = _CONFIG_DIR

    def _read(name: str) -> dict:
        path = config_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}. "
                f"Ensure all required TOML files exist under {config_dir}."
            )
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"Failed to parse config file '{path}': {exc}") from exc

    api_raw      = _read("api.toml")
    pipeline_raw = _read("pipeline.toml")
    search_raw   = _read("search.toml")
    llm_raw      = _read("llm.toml")

    return Config(
        auth=AuthConfig(**api_raw["auth"]),
        api=ApiConfig(**api_raw["api"]),
        paths=PathsConfig(**pipeline_raw["paths"]),
        processing=ProcessingConfig(**pipeline_raw["processing"]),
        retrieval=RetrievalConfig(**search_raw["retrieval"]),
        clustering=ClusteringConfig(**search_raw["clustering"]),
        llm=NovaProConfig(**llm_raw["nova_pro"]),
    )
