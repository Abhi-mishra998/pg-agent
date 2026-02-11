#!/usr/bin/env python3
"""
OutputValidator - LLM Output Validation (Production Grade)

Purpose:
- Validate LLM outputs deterministically
- Detect low-quality / hallucinated responses
- Provide confidence scoring & remediation hints
- Optionally use LLM for deep semantic checks

Design:
- Heuristic-first
- LLM-assisted (optional, non-blocking)
- Fail-safe by default

Note: This validator works fully without LLM - it uses deterministic
heuristics when Ollama is not available.
"""

import logging
import json
import math
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from llm.llama_client import LlamaClient


# -------------------------------------------------------------------
# Output Validator
# -------------------------------------------------------------------

class OutputValidator:
    """
    Validates LLM outputs using:
    1. Deterministic heuristics (always available)
    2. Optional LLM semantic evaluation (when Ollama available)
    """

    def __init__(
        self,
        llm_client: Optional[LlamaClient] = None,
        strict_mode: bool = False,
        enable_llm_validation: bool = False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # LLM client is optional - don't fail if unavailable
        self._llm_client = llm_client
        if self._llm_client is None:
            try:
                self._llm_client = LlamaClient(silent=True)
            except Exception:
                self._llm_client = None
        
        self.strict_mode = strict_mode
        # Only enable LLM validation if we have a working client
        self.enable_llm_validation = enable_llm_validation and self._llm_client is not None

    # ------------------------------------------------------------------
    # Public API (USED BY TERMINAL AGENT)
    # ------------------------------------------------------------------

    def validate(self, text: str) -> Dict[str, Any]:
        """
        Validate text using deterministic logic.
        This method MUST NEVER crash the agent.
        """

        checks = {}
        issues: List[str] = []
        suggestions: List[str] = []

        # ---------------------------
        # Heuristic Checks
        # ---------------------------

        checks["not_empty"] = self._check_not_empty(text)
        checks["min_length"] = self._check_min_length(text)
        checks["structure"] = self._check_structure(text)
        checks["coherence"] = self._check_coherence(text)
        checks["grammar"] = self._check_grammar(text)
        checks["safety"] = self._check_safety(text)

        for name, passed in checks.items():
            if not passed:
                issues.append(name)

        # ---------------------------
        # Confidence Scoring
        # ---------------------------

        confidence = self._compute_confidence(checks)

        valid = confidence >= (0.85 if self.strict_mode else 0.6)

        if not valid:
            suggestions.extend(self._generate_suggestions(issues))

        result = {
            "valid": valid,
            "confidence": round(confidence, 2),
            "issues": issues,
            "suggestions": suggestions,
            "checks": checks,
        }

        # ---------------------------
        # Optional LLM Validation
        # ---------------------------

        if self.enable_llm_validation and self._llm_client:
            llm_result = self._validate_with_llm(text)
            result["llm_validation"] = llm_result

            # Blend confidence safely
            if "confidence" in llm_result:
                result["confidence"] = round(
                    (result["confidence"] * 0.7) + (llm_result["confidence"] * 0.3),
                    2,
                )

        return result

    # ------------------------------------------------------------------
    # LLM-Based Validation (Optional)
    # ------------------------------------------------------------------

    def _validate_with_llm(self, text: str) -> Dict[str, Any]:
        """
        LLM-based semantic validation.
        NEVER trusted blindly.
        """
        if not self._llm_client:
            return {
                "valid": True,
                "confidence": 0.5,
                "issues": ["llm_validation_unavailable"],
                "summary": "LLM validation unavailable",
            }

        prompt = f"""
You are a senior incident-response reviewer.

Evaluate the following output for:
- factual grounding
- coherence
- hallucination risk
- operational usefulness

TEXT:
{text}

Respond in STRICT JSON ONLY:
{{
  "valid": true | false,
  "confidence": 0.0-1.0,
  "issues": ["..."],
  "summary": "short justification"
}}
"""

        try:
            response = self._llm_client.generate(
                prompt=prompt,
                system="You are a strict validator. Output JSON only.",
                options={"temperature": 0.2},
            )

            parsed = json.loads(response.text)

            return {
                "valid": bool(parsed.get("valid", False)),
                "confidence": float(parsed.get("confidence", 0.5)),
                "issues": parsed.get("issues", []),
                "summary": parsed.get("summary", ""),
            }

        except Exception as e:
            self.logger.warning("LLM validation failed, fallback used: %s", e)
            return {
                "valid": True,
                "confidence": 0.5,
                "issues": ["llm_validation_failed"],
                "summary": "LLM validation unavailable",
            }

    # ------------------------------------------------------------------
    # Confidence Logic
    # ------------------------------------------------------------------

    def _compute_confidence(self, checks: Dict[str, bool]) -> float:
        """
        Weighted confidence score.
        Penalizes structural & safety issues harder.
        """

        weights = {
            "not_empty": 0.25,
            "min_length": 0.15,
            "structure": 0.2,
            "coherence": 0.2,
            "grammar": 0.1,
            "safety": 0.1,
        }

        score = 0.0
        for check, passed in checks.items():
            weight = weights.get(check, 0)
            score += weight if passed else 0

        # Normalize
        return min(max(score, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        mapping = {
            "not_empty": "Provide a non-empty response.",
            "min_length": "Expand the response with more detail.",
            "structure": "Organize the response into clear sections.",
            "coherence": "Ensure logical flow and clear reasoning.",
            "grammar": "Fix grammatical or sentence issues.",
            "safety": "Avoid unsafe or speculative claims.",
        }
        return [mapping.get(i, f"Fix issue: {i}") for i in issues]

    # ------------------------------------------------------------------
    # Heuristic Checks
    # ------------------------------------------------------------------

    def _check_not_empty(self, text: str) -> bool:
        return bool(text and text.strip())

    def _check_min_length(self, text: str, min_len: int = 40) -> bool:
        return len(text.strip()) >= min_len

    def _check_structure(self, text: str) -> bool:
        keywords = ["root", "cause", "impact", "recommend", "confidence"]
        hits = sum(1 for k in keywords if k in text.lower())
        return hits >= 2

    def _check_coherence(self, text: str) -> bool:
        sentences = [s for s in text.split(".") if len(s.strip()) > 5]
        return len(sentences) >= 2

    def _check_grammar(self, text: str) -> bool:
        words = text.split()
        return len(words) >= 5 and text[0].isupper()

    def _check_safety(self, text: str) -> bool:
        # Informational security terms are allowed
        banned = ["rm -rf /", "drop database prod"]
        return not any(b in text.lower() for b in banned)


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def create_output_validator(
    llm_client: Optional[LlamaClient] = None,
    strict_mode: bool = False,
    enable_llm_validation: bool = False,
) -> OutputValidator:
    return OutputValidator(
        llm_client=llm_client,
        strict_mode=strict_mode,
        enable_llm_validation=enable_llm_validation,
    )