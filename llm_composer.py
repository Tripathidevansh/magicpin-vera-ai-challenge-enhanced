#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — LLM Composer (Phase 3)
===================================================
Coordinates LLM message composition with automatic provider fallback,
strict JSON schema extraction, grounding validation, and deterministic
fallback safety.

Pipeline:
DecisionEngine -> VerifiedContextPacket -> LLM Completion -> JSON Parse
-> Schema & CTA Validation -> GroundingValidator -> Final Action
(On any failure -> DeterministicComposer fallback)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from context_packet import VerifiedContextPacket
from decision_engine import CandidateTrigger, DeterministicComposer, GroundingValidator
from llm_provider import LLMManager, TIMEOUT_LLM_CALL

# Configuration flags
ENABLE_LLM_COMPOSER = os.environ.get("ENABLE_LLM_COMPOSER", "true").lower() in ("true", "1", "yes")
LLM_MODE = os.environ.get("LLM_MODE", "hybrid").lower()  # "deterministic", "hybrid", "llm"


class LLMComposer:
    """
    Composes WhatsApp messages via LLM while enforcing that the deterministic
    DecisionEngine remains the sole authority for decisions and suppression.
    """

    def __init__(self, llm_manager: Optional[LLMManager] = None) -> None:
        self.llm_manager = llm_manager or LLMManager()

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON object from LLM response text."""
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()
        # Strip markdown code blocks
        if "```" in text:
            # Match json codeblock
            match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
            else:
                match = re.search(r"(\{[\s\S]*\})", text)
                if match:
                    text = match.group(1).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            # Fallback regex search for { ... }
            match = re.search(r"(\{[\s\S]*\})", text)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

        return None

    @classmethod
    def _validate_and_return_fallback(
        cls,
        fallback_action: Optional[Dict[str, Any]],
        candidate: CandidateTrigger,
        prefix: str = "[deterministic_fallback] ",
    ) -> Optional[Dict[str, Any]]:
        """Validate and return deterministic fallback action, tagging rationale."""
        if not fallback_action:
            return None
        is_valid, _ = GroundingValidator.validate(
            body=fallback_action.get("body", ""),
            category=candidate.category,
            merchant=candidate.merchant,
            trigger=candidate.trigger,
            customer=candidate.customer,
            cta=fallback_action.get("cta"),
        )
        if not is_valid:
            return None

        # Annotate rationale so observers see fallback occurred
        rat = fallback_action.get("rationale", "")
        if not rat.startswith("["):
            fallback_action["rationale"] = f"{prefix}{rat}"
        return fallback_action

    async def compose(
        self, candidate: CandidateTrigger
    ) -> Optional[Dict[str, Any]]:
        """
        Attempts LLM composition with full guardrail validation.
        Seamlessly falls back to DeterministicComposer if LLM fails,
        times out, or violates grounding rules.
        """
        enable_llm = os.environ.get("ENABLE_LLM_COMPOSER", "true").lower() in ("true", "1", "yes")
        llm_mode = os.environ.get("LLM_MODE", "hybrid").lower()

        # 1. Routing check: if deterministic mode or disabled, use DeterministicComposer
        if not enable_llm or llm_mode == "deterministic":
            fallback = DeterministicComposer.compose(candidate)
            return self._validate_and_return_fallback(fallback, candidate, "[mode:deterministic] ")

        # In hybrid mode without configured providers, fall back cleanly
        if llm_mode == "hybrid" and not self.llm_manager.has_any_provider():
            fallback = DeterministicComposer.compose(candidate)
            return self._validate_and_return_fallback(fallback, candidate, "[fallback:no_llm_provider] ")

        from prompt_builder import PromptBuilder

        # 2. Extract Verified Context Packet
        packet = VerifiedContextPacket.from_candidate(candidate)
        sys_prompt = PromptBuilder.build_system_prompt()
        user_prompt = PromptBuilder.build_user_prompt(packet)

        # 3. Call LLM Manager (Primary -> Fallback)
        raw_text, provider_used = await self.llm_manager.complete_with_fallback(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            timeout=TIMEOUT_LLM_CALL,
        )

        if not raw_text:
            # Provider failed or timed out -> Fallback to deterministic composer
            fallback = DeterministicComposer.compose(candidate)
            return self._validate_and_return_fallback(
                fallback, candidate, f"[fallback:{provider_used}] "
            )

        # 4. Parse JSON Output
        parsed = self._extract_json(raw_text)
        if not parsed:
            fallback = DeterministicComposer.compose(candidate)
            return self._validate_and_return_fallback(
                fallback, candidate, "[fallback:malformed_json] "
            )

        body = str(parsed.get("body", "")).strip()
        cta = str(parsed.get("cta", "open_ended")).strip()
        template_name = str(parsed.get("template_name") or f"vera_{candidate.trigger_kind}_v1")
        template_params = parsed.get("template_params", [])
        if not isinstance(template_params, list):
            template_params = []
        rationale = str(parsed.get("rationale") or candidate.why_now)

        if not body:
            fallback = DeterministicComposer.compose(candidate)
            return self._validate_and_return_fallback(
                fallback, candidate, "[fallback:empty_body] "
            )

        # Normalize CTA to allowed values
        valid_ctas = {"binary_yes_no", "open_ended", "multi_choice_slot", "none"}
        if cta not in valid_ctas:
            cta = "open_ended"

        # 5. Strict Grounding Validation
        is_valid, reason = GroundingValidator.validate(
            body=body,
            category=candidate.category,
            merchant=candidate.merchant,
            trigger=candidate.trigger,
            customer=candidate.customer,
            cta=cta,
        )

        if not is_valid:
            # LLM output failed grounding -> Fallback to deterministic composer
            fallback = DeterministicComposer.compose(candidate)
            return self._validate_and_return_fallback(
                fallback, candidate, f"[fallback:grounding_violation:{reason}] "
            )

        # 6. Build Accepted Action
        date_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        conv_id = f"conv_{candidate.merchant_id}_{candidate.trigger_kind}_{date_stamp}"

        return {
            "conversation_id": conv_id,
            "merchant_id": candidate.merchant_id,
            "customer_id": candidate.customer_id,
            "send_as": packet.send_as,
            "trigger_id": candidate.trigger_id,
            "template_name": template_name,
            "template_params": [str(p) for p in template_params],
            "body": body,
            "cta": cta,
            "suppression_key": candidate.suppression_key,
            "rationale": f"[{provider_used}] {rationale}",
        }
