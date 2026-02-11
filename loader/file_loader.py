#!/usr/bin/env python3
"""
FileLoader - Unified File Loading Layer for pg-agent

Responsibilities:
- Load user-provided files safely
- Detect file type
- Read content correctly
- Normalize output for analysis pipeline

This module is intentionally I/O only.
NO parsing. NO analysis. NO business logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any


# ---------------------------------------------------------------------
# File Loader
# ---------------------------------------------------------------------

class FileLoader:
    """
    Loads files from disk and returns normalized content.
    """

    SUPPORTED_EXTENSIONS = {
        ".log": "log",
        ".txt": "text",
        ".json": "json",
        ".sql": "sql",
        ".html": "html",
        ".htm": "html",
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a file from disk and normalize its content.

        Returns:
        {
            "path": str,
            "type": str,
            "content": Any,
            "size_bytes": int
        }
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        file_type = self._detect_type(path)
        size_bytes = path.stat().st_size

        self.logger.info(
            "Loading file: %s (type=%s, size=%d bytes)",
            path, file_type, size_bytes
        )

        content = self._read_file(path, file_type)

        return {
            "path": str(path),
            "type": file_type,
            "content": content,
            "size_bytes": size_bytes,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_type(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        return self.SUPPORTED_EXTENSIONS[ext]

    def _read_file(self, path: Path, file_type: str) -> Any:
        """
        Read file content based on type.
        """
        try:
            if file_type == "json":
                return self._read_json(path)

            # Everything else treated as text
            return self._read_text(path)

        except Exception as exc:
            self.logger.error("Failed to read file %s: %s", path, exc)
            raise RuntimeError(f"Failed to read file: {path}") from exc

    def _read_text(self, path: Path) -> str:
        """
        Read text-based files safely.
        """
        return path.read_text(encoding="utf-8", errors="replace")

    def _read_json(self, path: Path) -> Any:
        """
        Read and parse JSON files.
        """
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)