"""
Repo-relative path helpers for model_config.json.

``train_path`` used to be stored as an absolute drive path, which broke
auto-detection of test data as soon as a model directory was copied to another
machine. It is now saved relative to the repository root and resolved against the
root on read; old configs with absolute values keep working unchanged.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def to_repo_relative(path):
    """Return ``path`` relative to the repo root with '/' separators.

    Falls back to the absolute path when no relative form exists (other drive).
    Empty / None values are returned untouched.
    """
    if not path:
        return path
    absolute = os.path.abspath(str(path))
    try:
        rel = os.path.relpath(absolute, REPO_ROOT)
    except ValueError:
        return absolute
    return rel.replace(os.sep, '/')


def resolve_repo_path(path):
    """Resolve a stored path: absolute values pass through, relative ones are
    interpreted against the repo root (never against the current directory)."""
    if not path:
        return path
    if os.path.isabs(str(path)):
        return str(path)
    return os.path.normpath(os.path.join(str(REPO_ROOT), str(path)))
