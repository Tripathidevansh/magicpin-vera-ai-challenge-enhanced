#!/usr/bin/env python3
"""
Unit Test Suite for Deterministic Decision Engine (Phase 2)
==========================================================
Tests A through M as mandated in the challenge specification.
"""

import asyncio
import unittest
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

from bot import ThreadSafeState
from decision_engine import (
    CandidateTrigger,
    DecisionEngine,
    DeterministicComposer,
    GroundingValidator,
    parse_iso_utc,
)

class TestDecisionEngine(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.state = ThreadSafeState()
        self.engine = DecisionEngine(max_candidates=5)
        self.now_iso = "2026-04-26T10:35:00Z"
        self.now_dt = parse_iso_utc(self.now_iso)

        # Standard baseline dentist category
        self.cat_dentist = {
            "slug": "dentists",
            "display_name": "Dentists",
            "voice": {
                "tone": "peer_clinical",
                "vocab_taboo": ["guaranteed", "100% safe", "completely cure"],
            },
            "offer_catalog": [
                {"id": "den_001", "title": "Dental Cleaning @ ₹299", "value": "299", "type": "service_at_price"}
            ],
            "digest": [
                {
                    "id": "d_001_fluoride",
                    "title": "3-month fluoride recall cuts caries 38% better",
                    "source": "JIDA Oct 2026, p.14",
                    "trial_n": 2100,
                    "patient_segment": "high_risk_adults",
                }
            ],
        }

        # Standard baseline merchant
        self.mer_meera = {
            "merchant_id": "m_001_drmeera",
            "category_slug": "dentists",
            "identity": {
                "name": "Dr. Meera's Dental Clinic",
                "owner_first_name": "Meera",
                "locality": "Lajpat Nagar",
            },
            "offers": [
                {"id": "o_001", "title": "Dental Cleaning @ ₹299", "status": "active"}
            ],
            "signals": ["high_risk_adult_cohort"],
        }

        # Standard customer
        self.cust_priya = {
            "customer_id": "c_001_priya",
            "merchant_id": "m_001_drmeera",
            "identity": {"name": "Priya", "language_pref": "hi-en mix"},
        }

        await self.state.put_context("category", "dentists", 1, self.cat_dentist, self.now_iso)
        await self.state.put_context("merchant", "m_001_drmeera", 1, self.mer_meera, self.now_iso)
        await self.state.put_context("customer", "c_001_priya", 1, self.cust_priya, self.now_iso)

    # -----------------------------------------------------------------------
    # A. Expired trigger -> suppressed
    # -----------------------------------------------------------------------
    async def test_a_expired_trigger_suppressed(self):
        expired_trg = {
            "id": "trg_expired",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 3,
            "suppression_key": "supp:expired",
            "expires_at": "2026-04-20T00:00:00Z",  # in the past relative to now_iso
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_expired", 1, expired_trg, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_expired"], self.now_iso, self.state)
        self.assertEqual(len(actions), 0, "Expired trigger must produce 0 actions")

    # -----------------------------------------------------------------------
    # B. Active suppression key -> suppressed
    # -----------------------------------------------------------------------
    async def test_b_active_suppression_key_suppressed(self):
        supp_key = "supp:active_key"
        await self.state.add_suppression(
            supp_key, (self.now_dt + timedelta(days=2)).isoformat().replace("+00:00", "Z")
        )
        active_trg = {
            "id": "trg_suppressed",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 4,
            "suppression_key": supp_key,
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_suppressed", 1, active_trg, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_suppressed"], self.now_iso, self.state)
        self.assertEqual(len(actions), 0, "Trigger with active suppression key must be suppressed")

    # -----------------------------------------------------------------------
    # C. Opted-out merchant -> suppressed
    # -----------------------------------------------------------------------
    async def test_c_opted_out_merchant_suppressed(self):
        await self.state.opt_out_merchant("m_001_drmeera")
        trg = {
            "id": "trg_opted_out",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 5,
            "suppression_key": "supp:optout",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_opted_out", 1, trg, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_opted_out"], self.now_iso, self.state)
        self.assertEqual(len(actions), 0, "Opted-out merchant must not receive proactive outreach")

    # -----------------------------------------------------------------------
    # D. Missing required context -> suppressed safely
    # -----------------------------------------------------------------------
    async def test_d_missing_context_suppressed_safely(self):
        trg_unknown_merchant = {
            "id": "trg_unknown_mer",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_unknown_nonexistent",
            "urgency": 4,
            "suppression_key": "supp:unknown",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {},
        }
        await self.state.put_context("trigger", "trg_unknown_mer", 1, trg_unknown_merchant, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_unknown_mer"], self.now_iso, self.state)
        self.assertEqual(len(actions), 0, "Missing merchant context must not crash and be skipped")

        # Missing customer context on customer-scoped trigger
        trg_missing_cust = {
            "id": "trg_missing_cust",
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": "m_001_drmeera",
            "customer_id": "c_nonexistent",
            "urgency": 4,
            "suppression_key": "supp:missing_cust",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {},
        }
        await self.state.put_context("trigger", "trg_missing_cust", 1, trg_missing_cust, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_missing_cust"], self.now_iso, self.state)
        self.assertEqual(len(actions), 0, "Missing customer context must be skipped safely")

    # -----------------------------------------------------------------------
    # E. Two triggers same merchant -> only one selected
    # -----------------------------------------------------------------------
    async def test_e_two_triggers_same_merchant_only_one_selected(self):
        trg1 = {
            "id": "trg_meera_low",
            "scope": "merchant",
            "kind": "curious_ask_due",
            "merchant_id": "m_001_drmeera",
            "urgency": 1,
            "suppression_key": "supp:meera_low",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {},
        }
        trg2 = {
            "id": "trg_meera_high",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 4,
            "suppression_key": "supp:meera_high",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_meera_low", 1, trg1, self.now_iso)
        await self.state.put_context("trigger", "trg_meera_high", 1, trg2, self.now_iso)

        actions = await self.engine.evaluate_triggers(
            ["trg_meera_low", "trg_meera_high"], self.now_iso, self.state
        )
        self.assertEqual(len(actions), 1, "Only ONE action per merchant may survive in a single tick")

    # -----------------------------------------------------------------------
    # F. Higher urgency beats lower urgency
    # -----------------------------------------------------------------------
    async def test_f_higher_urgency_beats_lower_urgency(self):
        trg_low = {
            "id": "trg_urgent_1",
            "scope": "merchant",
            "kind": "curious_ask_due",
            "merchant_id": "m_001_drmeera",
            "urgency": 1,
            "suppression_key": "supp:urg1",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {},
        }
        trg_high = {
            "id": "trg_urgent_5",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 5,
            "suppression_key": "supp:urg5",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_urgent_1", 1, trg_low, self.now_iso)
        await self.state.put_context("trigger", "trg_urgent_5", 1, trg_high, self.now_iso)

        actions = await self.engine.evaluate_triggers(
            ["trg_urgent_1", "trg_urgent_5"], self.now_iso, self.state
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["trigger_id"], "trg_urgent_5")

    # -----------------------------------------------------------------------
    # G. Low relevance / unsupported can result in no action (restraint)
    # -----------------------------------------------------------------------
    async def test_g_restraint_on_unsupported_or_empty(self):
        # Empty available triggers
        actions = await self.engine.evaluate_triggers([], self.now_iso, self.state)
        self.assertEqual(actions, [], "Empty triggers must return []")

        # Unsupported trigger kind with no grounded template
        trg_unsupported = {
            "id": "trg_unsupported_kind",
            "scope": "merchant",
            "kind": "totally_unknown_unsupported_event",
            "merchant_id": "m_001_drmeera",
            "urgency": 1,
            "suppression_key": "supp:unsupported",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {},
        }
        await self.state.put_context("trigger", "trg_unsupported_kind", 1, trg_unsupported, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_unsupported_kind"], self.now_iso, self.state)
        self.assertEqual(actions, [], "Unsupported trigger without grounded template must be skipped")

    # -----------------------------------------------------------------------
    # H. Customer trigger uses merchant_on_behalf
    # -----------------------------------------------------------------------
    async def test_h_customer_trigger_uses_merchant_on_behalf(self):
        trg_cust = {
            "id": "trg_recall_priya",
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": "m_001_drmeera",
            "customer_id": "c_001_priya",
            "urgency": 3,
            "suppression_key": "supp:recall_priya",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {
                "service_due": "Dental Cleaning",
                "available_slots": [
                    {"label": "Wed 5 Nov, 6pm"},
                    {"label": "Thu 6 Nov, 5pm"},
                ],
            },
        }
        await self.state.put_context("trigger", "trg_recall_priya", 1, trg_cust, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_recall_priya"], self.now_iso, self.state)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["send_as"], "merchant_on_behalf")
        self.assertEqual(actions[0]["customer_id"], "c_001_priya")

    # -----------------------------------------------------------------------
    # I. Fabricated price is never produced
    # -----------------------------------------------------------------------
    def test_i_fabricated_price_rejected(self):
        # A message with an ungrounded price (₹99999 is not in catalog or merchant offers)
        bad_body = "Hi Priya, get your Dental Cleaning @ ₹99999 today!"
        is_valid, reason = GroundingValidator.validate(
            body=bad_body,
            category=self.cat_dentist,
            merchant=self.mer_meera,
            trigger={"payload": {}},
            customer=self.cust_priya,
        )
        self.assertFalse(is_valid)
        self.assertIn("unverified_price", reason)

        # Grounded price (₹299 exists in merchant offers and category catalog)
        good_body = "Hi Priya, Dental Cleaning @ ₹299 ready for you."
        is_valid, reason = GroundingValidator.validate(
            body=good_body,
            category=self.cat_dentist,
            merchant=self.mer_meera,
            trigger={"payload": {}},
            customer=self.cust_priya,
        )
        self.assertTrue(is_valid, f"Expected valid price, got {reason}")

    # -----------------------------------------------------------------------
    # J. URL never appears in body
    # -----------------------------------------------------------------------
    def test_j_url_rejected_in_body(self):
        body_with_url = "Visit https://magicpin.com for your dental booking."
        is_valid, reason = GroundingValidator.validate(
            body=body_with_url,
            category=self.cat_dentist,
            merchant=self.mer_meera,
            trigger={"payload": {}},
            customer=self.cust_priya,
        )
        self.assertFalse(is_valid)
        self.assertEqual(reason, "contains_url")

    # -----------------------------------------------------------------------
    # K. Successful action creates suppression state
    # -----------------------------------------------------------------------
    async def test_k_successful_action_creates_suppression_state(self):
        trg = {
            "id": "trg_create_supp",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 3,
            "suppression_key": "supp:unique_test_k",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_create_supp", 1, trg, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_create_supp"], self.now_iso, self.state)
        self.assertEqual(len(actions), 1)

        # Verify suppression key is now actively recorded in state
        is_supp = await self.state.is_suppressed("supp:unique_test_k", self.now_dt)
        self.assertTrue(is_supp, "Action must persist suppression key")

        # Verify merchant cooldown is active
        is_cooldown = await self.state.is_merchant_in_cooldown("m_001_drmeera", self.now_dt)
        self.assertTrue(is_cooldown, "Action must place merchant in cooldown")

    # -----------------------------------------------------------------------
    # L. Repeated tick does not duplicate the same action
    # -----------------------------------------------------------------------
    async def test_l_repeated_tick_does_not_duplicate(self):
        trg = {
            "id": "trg_dedup_test",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 4,
            "suppression_key": "supp:dedup_l",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_dedup_test", 1, trg, self.now_iso)

        # Tick 1: Action produced
        actions1 = await self.engine.evaluate_triggers(["trg_dedup_test"], self.now_iso, self.state)
        self.assertEqual(len(actions1), 1)

        # Tick 2 (simulated 5 mins later with same trigger available): Must NOT duplicate
        tick2_now = (self.now_dt + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
        actions2 = await self.engine.evaluate_triggers(["trg_dedup_test"], tick2_now, self.state)
        self.assertEqual(len(actions2), 0, "Repeated tick must suppress duplicate action")

    # -----------------------------------------------------------------------
    # M. Adaptive/unseen trigger fields do not crash the bot
    # -----------------------------------------------------------------------
    async def test_m_adaptive_unseen_fields_do_not_crash(self):
        trg_adaptive = {
            "id": "trg_adaptive_unseen",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 4,
            "suppression_key": "supp:adaptive_m",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {
                "category": "dentists",
                "top_item_id": "d_001_fluoride",
                "unseen_nested_object": {"field_a": 123, "field_b": ["x", "y"]},
                "unexpected_flag": True,
                "surprise_null_field": None,
            },
            "unseen_root_field": "hello_adaptive_world",
        }
        await self.state.put_context("trigger", "trg_adaptive_unseen", 1, trg_adaptive, self.now_iso)
        actions = await self.engine.evaluate_triggers(["trg_adaptive_unseen"], self.now_iso, self.state)
        self.assertEqual(len(actions), 1, "Unseen adaptive fields must not crash the bot")

if __name__ == "__main__":
    unittest.main()
