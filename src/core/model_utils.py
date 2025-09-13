"""Utilities for model path resolution and management.

This module provides common functionality for resolving local model paths
across components and preventing code duplication.
"""

from pathlib import Path
from typing import Optional


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
    root = Path(base_dir)
    
    # Try direct path first
    model_dir = root / model_name
    if model_dir.exists() and model_dir.is_dir():
        return model_dir
    
    # Fallback: search recursively by terminal folder name (case-insensitive)
    target = model_name.split("/")[-1].lower()
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == target]
    
    if candidates:
        return candidates[0]
    
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
    root = Path(base_dir)
    
    if not root.exists():
        return []
    
    target = model_name.split("/")[-1].lower()
    return [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == target]


def validate_model_directory(model_path: Path) -> bool:
    """Validate that a directory contains the expected model files.
    
    Checks for the presence of key model files that indicate a properly
    structured sentence-transformers model directory.
    
    Args:
        model_path: Path to the model directory to validate
        
    Returns:
        True if the directory appears to contain a valid model
    """
    if not model_path.exists() or not model_path.is_dir():
        return False
    
    # Check for common sentence-transformers files
    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    tokenizer_files = ["tokenizer.json", "vocab.txt"]
    
    # Must have config.json
    if not any((model_path / f).exists() for f in required_files):
        return False
    
    # Must have at least one model file
    if not any((model_path / f).exists() for f in model_files):
        return False
    
    # Should have some tokenizer files (not strictly required for all models)
    return True