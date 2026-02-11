#!/usr/bin/env python3
"""
AgentContext - Runtime Memory for pg-agent

This module provides a structured, in-memory context store
for the terminal agent.

Purpose:
- Maintain state across commands
- Store loaded files, parsed data, signals, evidence
- Enable follow-up questions without reprocessing
- Act as the "working memory" of the agent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------

@dataclass
class AgentContext:
    """
    Holds the full runtime context of the agent session.
    """

    # File information
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    raw_content: Optional[str] = None

    # Parsed / processed data
    parsed_data: Optional[Any] = None

    # Analysis pipeline outputs
    signals: Optional[Any] = None
    evidence: Optional[Any] = None
    recommendations: Optional[Any] = None

    # Metadata
    loaded_at: Optional[str] = None
    last_analyzed_at: Optional[str] = None

    # Free-form notes (useful for LLM grounding)
    notes: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Context Manager
# ---------------------------------------------------------------------

class ContextManager:
    """
    Manages the AgentContext lifecycle.

    This class is intentionally simple and fast:
    - In-memory only
    - No disk writes
    - No persistence side effects
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._context = AgentContext()

    # --------------------------------------------------------------
    # File lifecycle
    # --------------------------------------------------------------

    def set_file(
        self,
        path: str,
        file_type: str,
        content: str,
    ) -> None:
        """
        Register a newly loaded file into context.
        """
        self.logger.info("File loaded into context: %s", path)

        self._context.file_path = path
        self._context.file_type = file_type
        self._context.raw_content = content
        self._context.loaded_at = datetime.utcnow().isoformat()

        # Reset downstream data
        self._context.parsed_data = None
        self._context.signals = None
        self._context.evidence = None
        self._context.recommendations = None

    # --------------------------------------------------------------
    # Pipeline stages
    # --------------------------------------------------------------

    def set_parsed_data(self, data: Any) -> None:
        self._context.parsed_data = data

    def set_signals(self, signals: Any) -> None:
        self._context.signals = signals
        self._context.last_analyzed_at = datetime.utcnow().isoformat()

    def set_evidence(self, evidence: Any) -> None:
        self._context.evidence = evidence

    def set_recommendations(self, recommendations: Any) -> None:
        self._context.recommendations = recommendations

    # --------------------------------------------------------------
    # Accessors
    # --------------------------------------------------------------

    def get_context(self) -> AgentContext:
        """
        Return the full context object.
        """
        return self._context

    def has_file_loaded(self) -> bool:
        return self._context.file_path is not None

    def has_analysis(self) -> bool:
        return self._context.signals is not None

    # --------------------------------------------------------------
    # Utility
    # --------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset the entire session context.
        """
        self.logger.info("Resetting agent context")
        self._context = AgentContext()

    def summary(self) -> Dict[str, Any]:
        """
        Lightweight summary for status/debug commands.
        """
        return {
            "file_loaded": self._context.file_path,
            "file_type": self._context.file_type,
            "loaded_at": self._context.loaded_at,
            "analyzed_at": self._context.last_analyzed_at,
            "has_signals": self._context.signals is not None,
            "has_evidence": self._context.evidence is not None,
            "has_recommendations": self._context.recommendations is not None,
        }