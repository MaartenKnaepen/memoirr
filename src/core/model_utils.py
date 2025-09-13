"""Utilities for model path resolution and management.

This module provides common functionality for resolving local model paths
across components and preventing code duplication.
"""

from pathlib import Path
from typing import Optional
from src.core.logging_config import get_logger


def resolve_model_path(model_name: str, base_dir: str = "models") -> Path:
    """Resolve the path to a local model directory.
    
    This function implements the standard Memoirr model resolution logic:
    1. Try direct path: models/<model_name>
    2. Fallback: search recursively for folders with matching terminal name (case-insensitive)
    
    Args:
        model_name: Name of the model to resolve (e.g., "qwen3-embedding-0.6B")
        base_dir: Base directory to search in (default: "models")
        
    Returns:
        Path to the resolved model directory
        
    Raises:
        FileNotFoundError: If no matching model directory is found
    """
    logger = get_logger(__name__)
    root = Path(base_dir)
    
    logger.debug(
        "Starting model path resolution",
        model_name=model_name,
        base_dir=base_dir,
        base_dir_exists=root.exists(),
        component="model_utils"
    )
    
    # Try direct path first
    model_dir = root / model_name
    if model_dir.exists() and model_dir.is_dir():
        logger.info(
            "Model resolved via direct path",
            model_name=model_name,
            resolved_path=str(model_dir),
            method="direct",
            component="model_utils"
        )
        return model_dir
    
    logger.debug(
        "Direct path not found, trying recursive search",
        direct_path=str(model_dir),
        component="model_utils"
    )
    
    # Fallback: search recursively by terminal folder name (case-insensitive)
    target = model_name.split("/")[-1].lower()
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == target]
    
    logger.debug(
        "Recursive search completed",
        target_name=target,
        candidates_found=len(candidates),
        component="model_utils"
    )
    
    if candidates:
        resolved_path = candidates[0]
        logger.info(
            "Model resolved via recursive search",
            model_name=model_name,
            resolved_path=str(resolved_path),
            method="recursive",
            total_candidates=len(candidates),
            component="model_utils"
        )
        return resolved_path
    
    logger.error(
        "Model not found",
        model_name=model_name,
        base_dir=base_dir,
        target_name=target,
        base_dir_exists=root.exists(),
        component="model_utils"
    )
    
    raise FileNotFoundError(f"Model '{model_name}' not found in {base_dir}/")


def find_model_candidates(model_name: str, base_dir: str = "models") -> list[Path]:
    """Find all potential model directory candidates.
    
    Useful for debugging and providing user feedback about available models.
    
    Args:
        model_name: Name of the model to search for
        base_dir: Base directory to search in (default: "models")
        
    Returns:
        List of Path objects representing potential model directories
    """
    logger = get_logger(__name__)
    root = Path(base_dir)
    
    logger.debug(
        "Searching for model candidates",
        model_name=model_name,
        base_dir=base_dir,
        base_dir_exists=root.exists(),
        component="model_utils"
    )
    
    if not root.exists():
        logger.warning(
            "Base directory does not exist",
            base_dir=base_dir,
            component="model_utils"
        )
        return []
    
    target = model_name.split("/")[-1].lower()
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == target]
    
    logger.info(
        "Model candidate search completed",
        model_name=model_name,
        target_name=target,
        candidates_found=len(candidates),
        candidate_paths=[str(p) for p in candidates],
        component="model_utils"
    )
    
    return candidates


def validate_model_directory(model_path: Path) -> bool:
    """Validate that a directory contains the expected model files.
    
    Checks for the presence of key model files that indicate a properly
    structured sentence-transformers model directory.
    
    Args:
        model_path: Path to the model directory to validate
        
    Returns:
        True if the directory appears to contain a valid model
    """
    logger = get_logger(__name__)
    
    logger.debug(
        "Starting model directory validation",
        model_path=str(model_path),
        path_exists=model_path.exists(),
        is_directory=model_path.is_dir() if model_path.exists() else False,
        component="model_utils"
    )
    
    if not model_path.exists() or not model_path.is_dir():
        logger.warning(
            "Model directory validation failed - path does not exist or is not a directory",
            model_path=str(model_path),
            exists=model_path.exists(),
            is_dir=model_path.is_dir() if model_path.exists() else False,
            component="model_utils"
        )
        return False
    
    # Check for common sentence-transformers files
    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    tokenizer_files = ["tokenizer.json", "vocab.txt"]
    
    # Check which files exist
    existing_files = [f.name for f in model_path.iterdir() if f.is_file()]
    
    # Must have config.json
    has_config = any((model_path / f).exists() for f in required_files)
    if not has_config:
        logger.warning(
            "Model directory validation failed - missing required config files",
            model_path=str(model_path),
            required_files=required_files,
            existing_files=existing_files,
            component="model_utils"
        )
        return False
    
    # Must have at least one model file
    has_model = any((model_path / f).exists() for f in model_files)
    if not has_model:
        logger.warning(
            "Model directory validation failed - missing model files",
            model_path=str(model_path),
            model_files=model_files,
            existing_files=existing_files,
            component="model_utils"
        )
        return False
    
    # Check tokenizer files (informational)
    has_tokenizer = any((model_path / f).exists() for f in tokenizer_files)
    
    logger.info(
        "Model directory validation completed",
        model_path=str(model_path),
        is_valid=True,
        has_config=has_config,
        has_model=has_model,
        has_tokenizer=has_tokenizer,
        total_files=len(existing_files),
        component="model_utils"
    )
    
    return True