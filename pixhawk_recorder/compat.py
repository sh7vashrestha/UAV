"""Compatibility helpers for legacy/custom checkpoints."""

from __future__ import annotations

import __main__


def forbid_accumulation_hook(*args, **kwargs):
    return None


def ensure_checkpoint_compat() -> None:
    """Expose legacy symbols expected by older pickled checkpoints."""
    __main__.forbid_accumulation_hook = forbid_accumulation_hook
