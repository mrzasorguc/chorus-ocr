"""Backward-compatible command-line entry point for Chorus."""

from chorus.cli import main
from chorus.pipeline import read

__all__ = ["read"]


if __name__ == "__main__":
    main()
