"""Global configuration using Pydantic Settings.

Loads environment variables (including from a .env file) and provides
centralized access for components and utilities.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application-wide settings.

    Attributes:
        embedding_model_name: Local model folder name under ./models where
            model.safetensors and tokenizer files live.
        device: Optional device string for transformers (e.g., "cuda:0", "cpu").

        Chunker configuration (can be overridden via .env):
        - CHUNK_THRESHOLD: float/int/"auto" (string); default "auto".
        - CHUNK_SIZE: int; default 512.
        - CHUNK_SIMILARITY_WINDOW: int; default 3.
        - CHUNK_MIN_SENTENCES: int; default 2.
        - CHUNK_MIN_CHARACTERS_PER_SENTENCE: int; default 24.
        - CHUNK_DELIM: JSON array string or plain string; default JSON list of common delimiters.
        - CHUNK_INCLUDE_DELIM: "prev" | "next" | "none"; default "prev".
        - CHUNK_SKIP_WINDOW: int; default 0.
        - CHUNK_INCLUDE_PARAMS: bool; default True.
        - CHUNK_INCLUDE_CAPTION_INDICES: bool; default True.
        - CHUNK_FAIL_FAST: bool; default True.

        Language detection configuration:
        - ENGLISH_ASCII_THRESHOLD: float; default 0.95 (95% ASCII required for English detection).
        - ASCII_CHAR_UPPER_LIMIT: int; default 128 (standard ASCII character limit).

        Time conversion configuration:
        - SECONDS_TO_MILLISECONDS_FACTOR: int; default 1000 (conversion factor from seconds to milliseconds).

        Chunker threshold configuration:
        - MIN_PERCENTILE_THRESHOLD: int; default 1 (minimum valid percentile).
        - MAX_PERCENTILE_THRESHOLD: int; default 100 (maximum valid percentile).
        - PERCENTILE_TO_DECIMAL_DIVISOR: float; default 100.0 (convert percentile to decimal).

        Embedding normalization configuration:
        - L2_NORM_P_VALUE: int; default 2 (p-value for L2 normalization).
        - EMBEDDING_NORMALIZATION_DIM: int; default 1 (dimension for normalization).
        - EMBEDDING_DIMENSION_FALLBACK: int; default 1024 (fallback when EMBEDDING_DIMENSION not set).

        Qdrant configuration:
        - QDRANT_URL: string; default "http://localhost:6300".
        - QDRANT_COLLECTION: string; default "memoirr".
        - QDRANT_RECREATE_INDEX: bool; default True.
        - QDRANT_RETURN_EMBEDDING: bool; default True.
        - QDRANT_WAIT_RESULT: bool; default True.
    """

    embedding_model_name: str = Field(
        default="qwen3-embedding-0.6B", alias="EMBEDDING_MODEL_NAME"
    )
    device: str | None = Field(default=None, alias="EMBEDDING_DEVICE")
    embedding_dimension: int | None = Field(default=None, alias="EMBEDDING_DIMENSION")

    # Language detection constants
    english_ascii_threshold: float = Field(default=0.95, alias="ENGLISH_ASCII_THRESHOLD")
    ascii_char_upper_limit: int = Field(default=128, alias="ASCII_CHAR_UPPER_LIMIT")

    # Time conversion constants
    seconds_to_milliseconds_factor: int = Field(default=1000, alias="SECONDS_TO_MILLISECONDS_FACTOR")

    # Chunker threshold constants
    min_percentile_threshold: int = Field(default=1, alias="MIN_PERCENTILE_THRESHOLD")
    max_percentile_threshold: int = Field(default=100, alias="MAX_PERCENTILE_THRESHOLD") 
    percentile_to_decimal_divisor: float = Field(default=100.0, alias="PERCENTILE_TO_DECIMAL_DIVISOR")

    # Embedding normalization constants
    l2_norm_p_value: int = Field(default=2, alias="L2_NORM_P_VALUE")
    embedding_normalization_dim: int = Field(default=1, alias="EMBEDDING_NORMALIZATION_DIM")

    # Embedding fallback (should be avoided - prefer explicit EMBEDDING_DIMENSION)
    embedding_dimension_fallback: int = Field(default=1024, alias="EMBEDDING_DIMENSION_FALLBACK")

    # Chunker settings
    chunk_threshold: str = Field(default="auto", alias="CHUNK_THRESHOLD")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_similarity_window: int = Field(default=3, alias="CHUNK_SIMILARITY_WINDOW")
    chunk_min_sentences: int = Field(default=2, alias="CHUNK_MIN_SENTENCES")
    chunk_min_characters_per_sentence: int = Field(default=24, alias="CHUNK_MIN_CHARACTERS_PER_SENTENCE")
    chunk_delim: str = Field(default='[". ", "! ", "? ", "\\n"]', alias="CHUNK_DELIM")
    chunk_include_delim: str | None = Field(default="prev", alias="CHUNK_INCLUDE_DELIM")
    chunk_skip_window: int = Field(default=0, alias="CHUNK_SKIP_WINDOW")
    chunk_include_params: bool = Field(default=True, alias="CHUNK_INCLUDE_PARAMS")
    chunk_include_caption_indices: bool = Field(default=True, alias="CHUNK_INCLUDE_CAPTION_INDICES")
    chunk_fail_fast: bool = Field(default=True, alias="CHUNK_FAIL_FAST")

    # Preprocessor settings
    pre_min_len: int = Field(default=1, alias="PRE_MIN_LEN")
    pre_dedupe_window_ms: int = Field(default=1000, alias="PRE_DEDUPE_WINDOW_MS")

    # Qdrant settings
    qdrant_url: str = Field(default="http://localhost:6300", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="memoirr", alias="QDRANT_COLLECTION")
    qdrant_recreate_index: bool = Field(default=True, alias="QDRANT_RECREATE_INDEX")
    qdrant_return_embedding: bool = Field(default=True, alias="QDRANT_RETURN_EMBEDDING")
    qdrant_wait_result: bool = Field(default=True, alias="QDRANT_WAIT_RESULT")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()  # type: ignore[call-arg]
