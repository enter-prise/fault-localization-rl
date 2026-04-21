# utils/llm.py
from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """
    Standardized response object for all backends.
    """
    text: str
    raw: Optional[Any] = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class LLMWrapper:
    """
    A lightweight LLM wrapper for the fault localization system.

    Design goals:
    1. Keep the system runnable even when no real LLM is configured.
    2. Provide a unified interface for reasoner / verifier / other agents.
    3. Make future replacement with OpenAI / local models easy.

    Supported modes:
    - mode="mock"   : deterministic fallback, no network required
    - mode="openai" : OpenAI-compatible usage if package + key are available

    Typical usage:
        llm = LLMWrapper()
        text = llm.generate("Explain why this function may be faulty.")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: int = 60,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.mode = mode or os.getenv("LLM_MODE", "mock").lower()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt or (
            "You are a careful software debugging assistant. "
            "You must produce concise, evidence-based analysis."
        )

        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        logger.info(
            "Initialized LLMWrapper with mode=%s, model=%s",
            self.mode,
            self.model,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Main plain-text generation method used by agents.

        Returns:
            str: model output text
        """
        resp = self.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.text

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Structured completion call.
        """
        system_prompt = system_prompt or self.system_prompt
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.mode == "openai":
            try:
                return self._complete_openai(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                logger.exception("OpenAI backend failed. Falling back to mock. Error: %s", e)
                return self._complete_mock(prompt, system_prompt)

        return self._complete_mock(prompt, system_prompt)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Chat-style interface.
        messages format:
            [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        resp = self.chat_complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.text

    def chat_complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Structured chat call.
        """
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self.mode == "openai":
            try:
                return self._chat_complete_openai(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                logger.exception("OpenAI chat backend failed. Falling back to mock. Error: %s", e)
                return self._chat_complete_mock(messages)

        return self._chat_complete_mock(messages)

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        default: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ask model for JSON and parse it safely.

        If parsing fails, returns `default` or a fallback dict.
        """
        default = default or {}
        text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = self._safe_parse_json(text)
        if parsed is None:
            return default
        return parsed

    # ------------------------------------------------------------------
    # Backend: OpenAI
    # ------------------------------------------------------------------

    def _complete_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """
        OpenAI-compatible backend.
        Supports the modern `openai` Python package if installed.

        Environment variables:
            OPENAI_API_KEY
            OPENAI_BASE_URL (optional)
            LLM_MODEL
        """
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI package not installed. Please `pip install openai` "
                "or use LLM_MODE=mock."
            ) from e

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)

        start = time.time()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        elapsed = time.time() - start
        logger.info("OpenAI completion finished in %.2fs", elapsed)

        text = resp.choices[0].message.content if resp.choices else ""
        usage = None
        if hasattr(resp, "usage") and resp.usage is not None:
            try:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            except Exception:
                usage = None

        return LLMResponse(
            text=text or "",
            raw=resp,
            model=self.model,
            usage=usage,
        )

    def _chat_complete_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI package not installed. Please `pip install openai` "
                "or use LLM_MODE=mock."
            ) from e

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)

        start = time.time()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )
        elapsed = time.time() - start
        logger.info("OpenAI chat completion finished in %.2fs", elapsed)

        text = resp.choices[0].message.content if resp.choices else ""
        usage = None
        if hasattr(resp, "usage") and resp.usage is not None:
            try:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            except Exception:
                usage = None

        return LLMResponse(
            text=text or "",
            raw=resp,
            model=self.model,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Backend: Mock / Fallback
    # ------------------------------------------------------------------

    def _complete_mock(self, prompt: str, system_prompt: str) -> LLMResponse:
        """
        Deterministic fallback output.
        This makes the whole system runnable even without a real LLM.
        """
        text = self._mock_reasoning(prompt, system_prompt)
        return LLMResponse(
            text=text,
            raw={"mode": "mock"},
            model="mock",
            usage=None,
        )

    def _chat_complete_mock(self, messages: List[Dict[str, str]]) -> LLMResponse:
        merged = "\n".join(
            f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}"
            for m in messages
        )
        text = self._mock_reasoning(merged, self.system_prompt)
        return LLMResponse(
            text=text,
            raw={"mode": "mock"},
            model="mock",
            usage=None,
        )

    # ------------------------------------------------------------------
    # Mock behavior
    # ------------------------------------------------------------------

    def _mock_reasoning(self, prompt: str, system_prompt: str) -> str:
        """
        Heuristic fallback logic.

        It does not pretend to be smart.
        It just produces structured, stable output so the pipeline can run.
        """
        lower_prompt = prompt.lower()

        # Case 1: JSON request
        if "json" in lower_prompt or "return a json" in lower_prompt:
            return json.dumps(
                {
                    "summary": "Mock response generated because no real LLM backend is configured.",
                    "suspicion": "The most suspicious code is the entity most semantically aligned with the issue text.",
                    "confidence": 0.35,
                    "evidence": [
                        "Keyword overlap with issue description",
                        "Potentially relevant API/function names",
                        "Fallback mock reasoning only"
                    ],
                    "verdict": "uncertain"
                },
                ensure_ascii=False,
                indent=2,
            )

        # Case 2: verifier / debate style
        verifier_keywords = ["verify", "verification", "debate", "counter", "refute", "genuine fault"]
        if any(k in lower_prompt for k in verifier_keywords):
            return (
                "Verification result: uncertain but plausible.\n"
                "Supporting evidence:\n"
                "1. The candidate appears semantically related to the reported issue.\n"
                "2. No strong contradictory evidence was provided in the prompt.\n"
                "3. This is a mock fallback decision and should not be treated as final proof.\n"
                "Final verdict: needs_manual_review."
            )

        # Case 3: ranking / reasoning style
        reasoner_keywords = ["rank", "rerank", "reason", "fault", "bug", "suspicious"]
        if any(k in lower_prompt for k in reasoner_keywords):
            return (
                "Reasoning summary:\n"
                "- Prefer entities whose names and local context align with the issue description.\n"
                "- Prefer functions directly involved in the failing behavior.\n"
                "- Prefer nodes connected to recent suspicious call paths or error-relevant modules.\n"
                "Conclusion:\n"
                "Select the top semantically aligned candidate as the current best hypothesis."
            )

        # Generic fallback
        return (
            "Mock LLM response.\n"
            "No external model backend is configured, so this output is generated by a deterministic fallback.\n"
            "The pipeline can continue running, but result quality will be limited."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse JSON robustly from raw text.
        """
        text = text.strip()
        if not text:
            return None

        # direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            return {"data": obj}
        except Exception:
            pass

        # try to extract json block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            chunk = text[start:end + 1]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
                return {"data": obj}
            except Exception:
                return None

        return None