"""Run chonkie.SemanticChunker with a self-hosted, local embedding model.

This module provides a small adapter to load a sentence-transformers-compatible
model from a local folder path (e.g., ./models/<MODEL_NAME>/model.safetensors)
so we can run completely offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from chonkie import SemanticChunker as ChonkieSemanticChunker
try:
    from chonkie.embeddings import BaseEmbeddings
except Exception:  # pragma: no cover - best-effort import for different versions
    from chonkie.embeddings.base import BaseEmbeddings  # type: ignore
from sentence_transformers import SentenceTransformer

from src.components.chunker.utilities.semantic_chunker.types import ChunkSpan, ChunkerParams


class _LocalEmbeddings(BaseEmbeddings):
    """Embeddings adapter that loads a sentence-transformers pipeline or falls back to direct safetensors.

    If a full sentence-transformers model folder is present, we delegate to SentenceTransformer.
    Otherwise, we load the safetensors weights and create a minimal mean pooling encoder to avoid
    automatic model creation mismatches.
    """

    def __init__(self, model_dir: str | Path, *, device: str | None = None) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.model_dir = str(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        try:
            # Try a proper sentence-transformers load first
            self.st_model = SentenceTransformer(self.model_dir, device=self.device)
            self._mode = "st"
        except Exception:
            # Fall back: build a basic transformer + mean pooling from safetensors
            self.st_model = None
            self._mode = "hf"
            # Load config/tokenizer if available
            try:
                self.config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
            except Exception:
                self.config = None
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, use_fast=True)
            except Exception:
                self.tokenizer = None
            # Validate presence of required files for HF load
            from pathlib import Path as _Path
            _dir = _Path(self.model_dir)
            has_cfg = (_dir / "config.json").exists()
            has_tok = any((_dir / name).exists() for name in ["tokenizer.json", "vocab.txt", "merges.txt", "tokenizer.model"])
            if not (has_cfg and has_tok):
                raise RuntimeError(
                    "Incomplete local model directory for offline load. Expected config/tokenizer files next to model.safetensors. "
                    f"Looked in: {_dir}. Please place at least 'config.json' and tokenizer files (e.g., 'tokenizer.json' or 'vocab.txt'/'merges.txt')."
                )
            # Load raw model weights
            self.model = AutoModel.from_pretrained(
                self.model_dir,
                local_files_only=True,
                trust_remote_code=True,
            ).to(self.device)

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:  # type: ignore[override]
        import torch
        from torch.nn.functional import normalize

        texts_list = list(texts)
        if self._mode == "st":
            return self.st_model.encode(
                texts_list,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        # HF mode: basic mean pooling over last hidden state
        if self.tokenizer is None:
            # Create a simple tokenizer via AutoTokenizer from model_dir name
            from transformers import AutoTokenizer as _AutoTok
            self.tokenizer = _AutoTok.from_pretrained(self.model_dir, local_files_only=True, use_fast=True)

        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**toks)
                # Mean pool over tokens that are not padding
                attn = toks["attention_mask"].unsqueeze(-1)
                summed = (out.last_hidden_state * attn).sum(dim=1)
                counts = attn.sum(dim=1).clamp(min=1)
                emb = summed / counts
                from src.core.config import get_settings
                settings = get_settings()
                emb = normalize(emb, p=settings.l2_norm_p_value, dim=settings.embedding_normalization_dim)
            all_vecs.append(emb.cpu().numpy())
        return np.concatenate(all_vecs, axis=0)


def _resolve_model_dir(model_name: str) -> str:
    """Resolve a model identifier for chonkie.

    - If a matching local folder exists under ./models or ./models/chunker (direct or nested),
      return that full path.
    - Otherwise, return the original string unchanged to let chonkie load a remote model id
      like "minishlab/potion-base-8M".

    Supported names can be raw folder names like "qwen3-embedding-0.6B" or simple aliases like
    "qwen3" / "qwen3-embedding" that map to the canonical local folder.
    """
    # Normalize common aliases to canonical folder names for local search
    terminal = model_name.split("/")[-1]
    key = terminal.lower()
    alias_map = {
        "qwen3": "qwen3-embedding-0.6B",
        "qwen3-embedding": "qwen3-embedding-0.6B",
        # Add more aliases here if needed
    }
    canonical = alias_map.get(key, terminal)

    # Check common local roots
    candidates = [Path("models") / canonical, Path("models/chunker") / canonical]
    for base in candidates:
        if base.exists():
            return str(base)

    # Fallback: search by terminal name (case-insensitive) anywhere under models/
    target = canonical.lower()
    models_root = Path("models")
    if models_root.exists():
        matches = [p for p in models_root.rglob("*") if p.is_dir() and p.name.lower() == target]
        if len(matches) >= 1:
            # Prefer shallowest path to avoid arbitrary deep matches
            matches.sort(key=lambda p: len(p.parts))
            return str(matches[0])

    # Not found locally: return the original input so chonkie can fetch remote id
    return model_name


def _coerce_threshold(value) -> float:
    """Accept float/int/str and return a valid (0,1) float.

    - "auto" or empty -> default 0.75
    - numeric strings allowed
    - integers in (1,100] are treated as percentages
    - values are validated to be 0 < x < 1
    """
    default = 0.75
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto" or v == "":
            return default
        try:
            f = float(v)
        except ValueError as e:
            raise ValueError(f"Invalid threshold string: {value}") from e
    elif isinstance(value, (int, float)):
        f = float(value)
    else:
        raise ValueError(f"Unsupported threshold type: {type(value)}")

    # Interpret percentages
    from src.core.config import get_settings
    settings = get_settings()
    
    if f > settings.min_percentile_threshold and f <= settings.max_percentile_threshold:
        f = f / settings.percentile_to_decimal_divisor
    if not (0 < f < 1):
        raise ValueError(f"threshold must be between 0 and 1, got {value}")
    return f


def run_semantic_chunker(
    text: str,
    params: ChunkerParams,
    *,
    model_name: str,
    device: str | None = None,
) -> List[ChunkSpan]:
    """Chunk a concatenated text string using chonkie SemanticChunker.

    Args:
        text: The full concatenated text built from captions.
        params: Chunking configuration.
        model_name: Local model folder name under ./models.
        device: Optional device string (e.g., "cuda:0" or "cpu").

    Returns:
        List of ChunkSpan produced by the chunker.
    """
    model_dir = _resolve_model_dir(model_name)

    thr = _coerce_threshold(params.threshold)

    # Pass a string path to chonkie; it will load the model internally.
    # If EMBEDDING_DIMENSION is set, forward it so chonkie/SentenceTransformer can build Pooling.
    from src.core.config import get_settings
    dim = get_settings().embedding_dimension

    # Build kwargs to be compatible across Chonkie versions
    common_kwargs = {
        "embedding_model": model_dir,
        "chunk_size": params.chunk_size,
        "similarity_window": params.similarity_window,
        "min_sentences": params.min_sentences,
        "min_characters_per_sentence": params.min_characters_per_sentence,
        "delim": params.delim,
        "include_delim": params.include_delim,
    }
    # Prefer modern "threshold" parameter, but fall back to "similarity_threshold" if needed.
    try:
        chunker = ChonkieSemanticChunker(threshold=thr, **common_kwargs)
    except TypeError:
        # Older versions may use similarity_threshold
        chunker = ChonkieSemanticChunker(similarity_threshold=thr, **common_kwargs)

    # chonkie may expose __call__ or .chunk depending on version
    if hasattr(chunker, "chunk"):
        chunks = chunker.chunk(text)
    else:
        chunks = chunker(text)

    out: List[ChunkSpan] = []
    for ch in chunks:
        out.append(
            ChunkSpan(
                text=ch.text,
                start_index=int(getattr(ch, "start_index", 0)),
                end_index=int(getattr(ch, "end_index", 0)),
                token_count=int(getattr(ch, "token_count", 0)),
            )
        )
    return out
