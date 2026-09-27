#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — LLM Composition Layer Test Suite (Phase 3)
======================================================================
Tests requirements A through K with MockProvider (zero external API calls,
no API keys required):

A. Valid JSON accepted
B. Malformed JSON -> fallback to DeterministicComposer
C. Provider timeout -> fallback to DeterministicComposer
D. Gemini failure -> Groq fallback succeeds
E. Both providers fail -> fallback to DeterministicComposer
F. Fabricated ₹99999 -> GroundingValidator rejects -> fallback
G. URL in body -> GroundingValidator rejects -> fallback
H. Taboo medical claim -> GroundingValidator rejects -> fallback
I. Internal jargon -> GroundingValidator rejects -> fallback
J. Multiple CTAs -> GroundingValidator rejects -> fallback
K. Valid grounded response -> accepted
"""

import asyncio
import json
import os
import unittest
from typing import Any, Dict

from decision_engine import CandidateTrigger, DeterministicComposer, GroundingValidator
from llm_composer import LLMComposer
from llm_provider import LLMManager, MockProvider


def make_candidate_trigger() -> CandidateTrigger:
    """Creates a realistic, isolated CandidateTrigger for testing."""
    category = {
        "slug": "dentists",
        "display_name": "Dental Clinics",
        "voice": {
            "tone": "clinical_peer",
            "register": "peer",
            "code_mix": "english_primary",
            "vocab_allowed": ["clinical", "trial", "abstract", "patient"],
            "vocab_taboo": ["cure guaranteed", "painless guaranteed", "cheap", "100% cure"],
        },
        "digest": [
            {
                "id": "dig_01",
                "title": "Pediatric caries remineralization study",
                "source": "Journal of Clinical Pediatric Dentistry",
                "trial_n": 450,
                "patient_segment": "pediatric_patients",
            }
        ],
    }

    merchant = {
        "identity": {
            "id": "m_dentist_01",
            "name": "Apex Dental Studio",
            "owner_first_name": "Pooja",
            "locality": "Koramangala",
            "city": "Bengaluru",
            "languages": ["en"],
        },
        "category_slug": "dentists",
        "offers": [
            {"id": "off_01", "title": "Preventive checkup ₹500", "value": 500, "status": "active"}
        ],
        "signals": ["pediatric_patient_interest"],
    }

    trigger = {
        "id": "trg_test_01",
        "kind": "research_digest",
        "urgency": 4,
        "expires_at": "2026-10-01T00:00:00Z",
        "suppression_key": "supp:test:01",
        "payload": {
            "top_item_id": "dig_01",
            "delta_pct": 25,
        },
    }

    return CandidateTrigger(
        trigger_id="trg_test_01",
        merchant_id="m_dentist_01",
        customer_id=None,
        category_slug="dentists",
        trigger_kind="research_digest",
        urgency=4,
        suppression_key="supp:test:01",
        expires_at_iso="2026-10-01T00:00:00Z",
        why_now="Pediatric caries study published in Journal of Clinical Pediatric Dentistry",
        relevant_facts=trigger["payload"],
        category=category,
        merchant=merchant,
        trigger=trigger,
        customer=None,
    )


class TestLLMLayer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Enable LLM mode for these tests
        os.environ["ENABLE_LLM_COMPOSER"] = "true"
        os.environ["LLM_MODE"] = "llm"
        os.environ["LLM_CALL_TIMEOUT"] = "0.05"

    # A. Valid JSON
    async def test_a_valid_json(self):
        valid_response = json.dumps({
            "template_name": "vera_research_digest_v1",
            "template_params": ["Pooja", "Pediatric study"],
            "body": "Dr. Pooja, Journal of Clinical Pediatric Dentistry published the 450-patient caries trial. Would you like me to pull the abstract?",
            "cta": "open_ended",
            "rationale": "Clinically grounded update for pediatric dentist.",
        })
        provider = MockProvider(provider_name="MockGemini", response_text=valid_response)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertIn("MockGemini", action["rationale"])
        self.assertEqual(action["cta"], "open_ended")
        self.assertIn("Journal of Clinical Pediatric Dentistry", action["body"])

    # B. Malformed JSON -> Fallback
    async def test_b_malformed_json(self):
        malformed = "Here is the response: { 'template_name': incomplete json..."
        provider = MockProvider(provider_name="MockGemini", response_text=malformed)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertIn("fallback", action["rationale"])
        self.assertIn("malformed_json", action["rationale"])
        # Action body is safe and grounded from deterministic composer
        self.assertIn("Pooja", action["body"])

    # C. Provider Timeout -> Fallback
    async def test_c_provider_timeout(self):
        provider = MockProvider(provider_name="MockGemini", should_timeout=True)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertIn("fallback", action["rationale"])

    # D. Gemini Failure -> Groq Fallback
    async def test_d_gemini_failure_groq_fallback(self):
        groq_response = json.dumps({
            "template_name": "vera_research_groq_v1",
            "template_params": [],
            "body": "Dr. Pooja, the latest Journal of Clinical Pediatric Dentistry trial is available. Shall we draft a patient summary?",
            "cta": "open_ended",
            "rationale": "Composed via Groq fallback.",
        })
        gemini_fail = MockProvider(provider_name="MockGemini", should_fail=True)
        groq_ok = MockProvider(provider_name="MockGroq", response_text=groq_response)

        mgr = LLMManager(primary_provider=gemini_fail, fallback_provider=groq_ok)
        composer = LLMComposer(llm_manager=mgr)

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertIn("MockGroq", action["rationale"])
        self.assertIn("Journal of Clinical Pediatric Dentistry", action["body"])

    # E. Both Providers Fail -> Deterministic Fallback
    async def test_e_both_providers_fail_deterministic_fallback(self):
        gemini_fail = MockProvider(provider_name="MockGemini", should_fail=True)
        groq_fail = MockProvider(provider_name="MockGroq", should_fail=True)

        mgr = LLMManager(primary_provider=gemini_fail, fallback_provider=groq_fail)
        composer = LLMComposer(llm_manager=mgr)

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertIn("fallback", action["rationale"])
        self.assertIn("all_providers_failed", action["rationale"])
        self.assertIn("Pooja", action["body"])

    # F. Fabricated Price ₹99999 -> Validator Rejects -> Fallback
    async def test_f_fabricated_price_rejected(self):
        hallucinated_price = json.dumps({
            "template_name": "vera_offer_v1",
            "template_params": [],
            "body": "Dr. Pooja, exclusive offer today: full smile makeover for only ₹99999! Want to book?",
            "cta": "open_ended",
            "rationale": "Hallucinated ₹99999 not in context.",
        })
        provider = MockProvider(provider_name="MockGemini", response_text=hallucinated_price)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertNotIn("₹99999", action["body"])
        self.assertIn("unverified_price", action["rationale"])

    # G. URL in Body -> Validator Rejects -> Fallback
    async def test_g_url_rejected(self):
        url_response = json.dumps({
            "template_name": "vera_link_v1",
            "template_params": [],
            "body": "Dr. Pooja, check out the study online at https://journal.dentistry.org/trial450 for full info. Want me to draft?",
            "cta": "open_ended",
            "rationale": "Included forbidden URL.",
        })
        provider = MockProvider(provider_name="MockGemini", response_text=url_response)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertNotIn("https://", action["body"])
        self.assertIn("contains_url", action["rationale"])

    # H. Taboo Medical Claim -> Validator Rejects -> Fallback
    async def test_h_taboo_medical_claim_rejected(self):
        taboo_response = json.dumps({
            "template_name": "vera_claim_v1",
            "template_params": [],
            "body": "Dr. Pooja, new technique provides a 100% cure guaranteed for early caries. Shall we share?",
            "cta": "open_ended",
            "rationale": "Medical claim violation.",
        })
        provider = MockProvider(provider_name="MockGemini", response_text=taboo_response)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertNotIn("100% cure", action["body"])
        self.assertIn("fallback", action["rationale"])

    # I. Internal Jargon -> Validator Rejects -> Fallback
    async def test_i_internal_jargon_rejected(self):
        jargon_response = json.dumps({
            "template_name": "vera_jargon_v1",
            "template_params": [],
            "body": "Dr. Pooja, trigger trg_test_01 fired with suppression_key supp:test:01. Would you like to proceed?",
            "cta": "open_ended",
            "rationale": "Leaked internal IDs.",
        })
        provider = MockProvider(provider_name="MockGemini", response_text=jargon_response)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertNotIn("trg_test_01", action["body"])
        self.assertNotIn("suppression_key", action["body"])
        self.assertIn("contains_internal_jargon", action["rationale"])

    # J. Multiple CTAs -> Validator Rejects -> Fallback
    async def test_j_multiple_ctas_rejected(self):
        multi_cta_response = json.dumps({
            "template_name": "vera_multi_v1",
            "template_params": [],
            "body": "Dr. Pooja, would you like to review the study? Or should we email the abstract? And what time works best for you?",
            "cta": "open_ended",
            "rationale": "Has 3 questions/competing CTAs.",
        })
        provider = MockProvider(provider_name="MockGemini", response_text=multi_cta_response)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertIn("fallback", action["rationale"])
        self.assertIn("multiple_ctas", action["rationale"])

    # K. Valid Grounded Response with Verified Context Data -> Accepted
    async def test_k_valid_grounded_response_accepted(self):
        grounded_response = json.dumps({
            "template_name": "vera_checkup_promo_v1",
            "template_params": ["500"],
            "body": "Dr. Pooja, your Preventive checkup ₹500 package is live in Koramangala. Shall we run a local showcase?",
            "cta": "open_ended",
            "rationale": "Verified ₹500 from active offer and Koramangala locality.",
        })
        provider = MockProvider(provider_name="MockGemini", response_text=grounded_response)
        composer = LLMComposer(llm_manager=LLMManager(primary_provider=provider))

        action = await composer.compose(make_candidate_trigger())
        self.assertIsNotNone(action)
        self.assertIn("MockGemini", action["rationale"])
        self.assertIn("₹500", action["body"])
        self.assertIn("Koramangala", action["body"])
        self.assertEqual(action["cta"], "open_ended")


if __name__ == "__main__":
    unittest.main()
