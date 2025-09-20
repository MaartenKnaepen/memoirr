"""Global configuration using Pydantic Settings.

Loads environment variables (including from a .env file) and provides
centralized access for components and utilities.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


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
        - USE_LANGDETECT: bool; default True (use langdetect library for proper language detection).
        - LANGDETECT_CONFIDENCE_THRESHOLD: float; default 0.7 (minimum confidence for langdetect classification).
        - LANGDETECT_FALLBACK_TO_ASCII: bool; default True (fallback to ASCII heuristic if langdetect fails).

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
    use_langdetect: bool = Field(default=True, alias="USE_LANGDETECT")
    langdetect_confidence_threshold: float = Field(default=0.7, alias="LANGDETECT_CONFIDENCE_THRESHOLD")
    langdetect_fallback_to_ascii: bool = Field(default=True, alias="LANGDETECT_FALLBACK_TO_ASCII")

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

    # Logging configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")  # 'json' or 'console'
    log_file: str | None = Field(default=None, alias="LOG_FILE")
    service_name: str = Field(default="memoirr", alias="SERVICE_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")

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

    @field_validator('embedding_dimension')
    @classmethod
    def validate_embedding_dimension(cls, v):
        if v is not None and v <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be positive")
        return v

    @field_validator('chunk_size')
    @classmethod
    def validate_chunk_size(cls, v):
        if v <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        return v

    @field_validator('chunk_similarity_window')
    @classmethod
    def validate_similarity_window(cls, v):
        if v < 1:
            raise ValueError("CHUNK_SIMILARITY_WINDOW must be at least 1")
        return v

    @field_validator('chunk_min_sentences')
    @classmethod
    def validate_min_sentences(cls, v):
        if v < 1:
            raise ValueError("CHUNK_MIN_SENTENCES must be at least 1")
        return v

    @field_validator('chunk_min_characters_per_sentence')
    @classmethod
    def validate_min_chars_per_sentence(cls, v):
        if v < 1:
            raise ValueError("CHUNK_MIN_CHARACTERS_PER_SENTENCE must be at least 1")
        return v

    @field_validator('english_ascii_threshold')
    @classmethod
    def validate_ascii_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("ENGLISH_ASCII_THRESHOLD must be between 0.0 and 1.0")
        return v

    @field_validator('ascii_char_upper_limit')
    @classmethod
    def validate_ascii_limit(cls, v):
        if not 1 <= v <= 1114111:  # Valid Unicode range
            raise ValueError("ASCII_CHAR_UPPER_LIMIT must be between 1 and 1114111")
        return v

    @field_validator('langdetect_confidence_threshold')
    @classmethod
    def validate_langdetect_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("LANGDETECT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        return v

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of: {', '.join(valid_levels)}")
        return v.upper()

    @field_validator('log_format')
    @classmethod
    def validate_log_format(cls, v):
        valid_formats = ['json', 'console']
        if v.lower() not in valid_formats:
            raise ValueError(f"LOG_FORMAT must be one of: {', '.join(valid_formats)}")
        return v.lower()

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of: {', '.join(valid_envs)}")
        return v.lower()


class MemoirrConfigError(Exception):
    """Configuration error with helpful suggestions."""
    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


def validate_model_accessibility(settings: Settings) -> List[str]:
    """Validate that the embedding model can be accessed."""
    issues = []
    
    try:
        from src.core.model_utils import resolve_model_path, find_model_candidates, validate_model_directory
        
        # Try to resolve the model path
        try:
            model_path = resolve_model_path(settings.embedding_model_name)
            
            # Validate the model directory structure
            if not validate_model_directory(model_path):
                issues.append(f"Model directory '{model_path}' exists but appears invalid (missing required files)")
                
        except FileNotFoundError:
            # Model not found - provide helpful suggestions
            candidates = find_model_candidates(settings.embedding_model_name)
            if candidates:
                suggestions = [f"  - {candidate}" for candidate in candidates[:3]]
                issues.append(f"Model '{settings.embedding_model_name}' not found. Did you mean:\n" + "\n".join(suggestions))
            else:
                # Check if models directory exists
                models_dir = Path("models")
                if not models_dir.exists():
                    issues.append(f"Model '{settings.embedding_model_name}' not found. Models directory 'models/' does not exist")
                else:
                    # List available models
                    available = [p.name for p in models_dir.iterdir() if p.is_dir()]
                    if available:
                        suggestions = [f"  - {model}" for model in available[:5]]
                        issues.append(f"Model '{settings.embedding_model_name}' not found. Available models:\n" + "\n".join(suggestions))
                    else:
                        issues.append(f"Model '{settings.embedding_model_name}' not found. No models found in 'models/' directory")
                        
    except ImportError:
        # model_utils not available during early initialization
        pass
        
    return issues


def validate_qdrant_config(settings: Settings) -> List[str]:
    """Validate Qdrant configuration."""
    issues = []
    
    # Basic URL validation
    if not settings.qdrant_url:
        issues.append("QDRANT_URL cannot be empty")
    elif settings.qdrant_url not in [":memory:", "http://localhost:6300"] and not settings.qdrant_url.startswith(("http://", "https://")):
        issues.append("QDRANT_URL must be ':memory:', start with 'http://' or 'https://'")
    
    # Collection name validation
    if not settings.qdrant_collection:
        issues.append("QDRANT_COLLECTION cannot be empty")
    elif "/" in settings.qdrant_collection or " " in settings.qdrant_collection or "@" in settings.qdrant_collection:
        issues.append("QDRANT_COLLECTION should only contain letters, numbers, hyphens, and underscores")
    
    return issues


def validate_chunk_delimiters(settings: Settings) -> List[str]:
    """Validate chunk delimiter configuration."""
    issues = []
    
    try:
        import json
        # Try to parse as JSON first
        try:
            delimiters = json.loads(settings.chunk_delim)
            if not isinstance(delimiters, list):
                issues.append("CHUNK_DELIM JSON must be a list of strings")
            elif not all(isinstance(d, str) for d in delimiters):
                issues.append("CHUNK_DELIM list must contain only strings")
            elif len(delimiters) == 0:
                issues.append("CHUNK_DELIM list cannot be empty")
        except json.JSONDecodeError:
            # Not JSON, should be a string - this is valid
            pass
                
    except ImportError:
        pass
        
    return issues


def validate_threshold_config(settings: Settings) -> List[str]:
    """Validate threshold configuration."""
    issues = []
    
    threshold = settings.chunk_threshold
    
    if isinstance(threshold, str):
        if threshold.lower() not in ["auto"]:
            try:
                # Try to parse as float
                float_val = float(threshold)
                if float_val >= 1.0 or float_val < 0.0:
                    issues.append("CHUNK_THRESHOLD as float must be between 0.0 and 1.0 (exclusive of 1.0)")
            except ValueError:
                issues.append("CHUNK_THRESHOLD must be 'auto', a float (0.0-1.0), or an integer (1-100)")
    
    return issues


def validate_settings_comprehensive(settings: Settings) -> Dict[str, Any]:
    """Comprehensive settings validation with detailed feedback."""
    validation_result = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "suggestions": []
    }
    
    # Collect all validation issues
    all_issues = []
    all_warnings = []
    
    # Model validation
    model_issues = validate_model_accessibility(settings)
    all_issues.extend(model_issues)
    
    # Qdrant validation
    qdrant_issues = validate_qdrant_config(settings)
    all_issues.extend(qdrant_issues)
    
    # Delimiter validation
    delimiter_issues = validate_chunk_delimiters(settings)
    all_issues.extend(delimiter_issues)
    
    # Threshold validation
    threshold_issues = validate_threshold_config(settings)
    all_issues.extend(threshold_issues)
    
    # Configuration warnings (non-blocking)
    if settings.embedding_dimension is None:
        all_warnings.append("EMBEDDING_DIMENSION not set - will use fallback. Consider setting explicit dimension for better performance")
    
    if settings.device is None:
        all_warnings.append("EMBEDDING_DEVICE not set - will auto-detect. Consider setting 'cuda:0' or 'cpu' for consistent behavior")
    
    if settings.log_format == "console" and settings.environment == "production":
        all_warnings.append("Using console logging in production environment - consider LOG_FORMAT=json for better log aggregation")
    
    if settings.chunk_size > 2048:
        all_warnings.append("CHUNK_SIZE is very large (>2048) - may impact embedding quality and processing speed")
    
    if settings.chunk_similarity_window > 10:
        all_warnings.append("CHUNK_SIMILARITY_WINDOW is very large (>10) - may slow down processing significantly")
    
    # Set final validation state
    validation_result["is_valid"] = len(all_issues) == 0
    validation_result["issues"] = all_issues
    validation_result["warnings"] = all_warnings
    
    # Add helpful suggestions
    if not validation_result["is_valid"]:
        validation_result["suggestions"] = [
            "Check your .env file for typos",
            "Ensure all required models are downloaded and placed in the models/ directory",
            "Verify Qdrant is running if using external instance",
            "Run 'python -m src.tools.validate_config' for detailed diagnostics"
        ]
    
    return validation_result


@lru_cache(maxsize=1) 
def get_settings(validate: bool = False) -> Settings:
    """Return a cached Settings instance with optional validation.
    
    Args:
        validate: Whether to run comprehensive validation (default: True)
        
    Returns:
        Settings instance
        
    Raises:
        MemoirrConfigError: If validation fails and validate=True
    """
    try:
        settings = Settings()  # type: ignore[call-arg]
        
        if validate:
            validation = validate_settings_comprehensive(settings)
            
            # Log validation results
            try:
                from src.core.logging_config import get_logger
                logger = get_logger(__name__)
                
                if not validation["is_valid"]:
                    logger.error(
                        "Configuration validation failed",
                        issues=validation["issues"],
                        component="config_validation"
                    )
                    
                if validation["warnings"]:
                    logger.warning(
                        "Configuration warnings detected",
                        warnings=validation["warnings"],
                        component="config_validation"
                    )
                    
                if validation["is_valid"]:
                    logger.info(
                        "Configuration validation passed",
                        warnings_count=len(validation["warnings"]),
                        component="config_validation"
                    )
                    
            except ImportError:
                # Logging not available during early initialization
                pass
            
            # Raise error if validation failed
            if not validation["is_valid"]:
                error_msg = "Configuration validation failed:\n"
                for issue in validation["issues"]:
                    error_msg += f"  ❌ {issue}\n"
                
                if validation["suggestions"]:
                    error_msg += "\nSuggestions:\n"
                    for suggestion in validation["suggestions"]:
                        error_msg += f"  💡 {suggestion}\n"
                
                raise MemoirrConfigError(error_msg, validation["suggestions"])
        
        return settings
        
    except Exception as e:
        if isinstance(e, MemoirrConfigError):
            raise
        else:
            raise MemoirrConfigError(f"Failed to load configuration: {str(e)}")
