"""Custom exceptions for SRT preprocessing."""
from __future__ import annotations


class SRTParseError(Exception):
    """Raised when an SRT file or text cannot be parsed properly."""


class LanguageFilterError(Exception):
    """Raised when language filtering encounters an unrecoverable error."""
