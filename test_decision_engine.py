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

    # -----------------------------------------------------------------------
    # N. Unrelated numeric/string field cannot authorize a price
    # -----------------------------------------------------------------------
    def test_n_unrelated_numeric_field_cannot_authorize_price(self):
        cat_no_offer = {"slug": "gyms", "display_name": "Gyms", "voice": {}, "offer_catalog": []}
        mer_no_offer = {"merchant_id": "m_test_gym", "category_slug": "gyms", "identity": {"name": "Test Gym"}, "offers": []}

        # Case 1: Trigger payload with integer field (days_remaining: 299)
        trg_int = {"payload": {"days_remaining": 299}}
        bad_body1 = "Join Test Gym today for only ₹299! Reply YES to claim."
        is_valid1, reason1 = GroundingValidator.validate(
            body=bad_body1,
            category=cat_no_offer,
            merchant=mer_no_offer,
            trigger=trg_int,
            customer=None,
            cta="binary_yes_no",
        )
        self.assertFalse(is_valid1, "Arbitrary integer field 'days_remaining: 299' must NOT authorize ₹299")
        self.assertIn("unverified_price", reason1)

        # Case 2: Trigger payload with string field (days_remaining: "299")
        trg_str = {"payload": {"days_remaining": "299"}}
        bad_body2 = "Join Test Gym today for only ₹299! Reply YES to claim."
        is_valid2, reason2 = GroundingValidator.validate(
            body=bad_body2,
            category=cat_no_offer,
            merchant=mer_no_offer,
            trigger=trg_str,
            customer=None,
            cta="binary_yes_no",
        )
        self.assertFalse(is_valid2, "Arbitrary string field 'days_remaining: \"299\"' must NOT authorize ₹299")
        self.assertIn("unverified_price", reason2)

        # Case 3: Genuine price field as integer (renewal_amount: 299) MUST authorize ₹299
        trg_price_int = {"payload": {"renewal_amount": 299}}
        good_body1 = "Renew your Test Gym membership today for ₹299. Reply YES to confirm."
        is_valid3, reason3 = GroundingValidator.validate(
            body=good_body1,
            category=cat_no_offer,
            merchant=mer_no_offer,
            trigger=trg_price_int,
            customer=None,
            cta="binary_yes_no",
        )
        self.assertTrue(is_valid3, f"Genuine price field 'renewal_amount: 299' should authorize ₹299, got: {reason3}")

        # Case 4: Genuine price field as string (renewal_amount: "299") MUST authorize ₹299
        trg_price_str = {"payload": {"renewal_amount": "299"}}
        good_body2 = "Renew your Test Gym membership today for ₹299. Reply YES to confirm."
        is_valid4, reason4 = GroundingValidator.validate(
            body=good_body2,
            category=cat_no_offer,
            merchant=mer_no_offer,
            trigger=trg_price_str,
            customer=None,
            cta="binary_yes_no",
        )
        self.assertTrue(is_valid4, f"Genuine price field 'renewal_amount: \"299\"' should authorize ₹299, got: {reason4}")

    # -----------------------------------------------------------------------
    # O. Pharmacy slug activates medical-claim protection
    # -----------------------------------------------------------------------
    def test_o_pharmacy_slug_activates_medical_claim_protection(self):
        cat_pharmacy = {
            "slug": "pharmacies",
            "display_name": "Pharmacies",
            "voice": {"vocab_taboo": []},
            "offer_catalog": [],
        }
        mer_pharmacy = {
            "merchant_id": "m_test_pharmacy",
            "category_slug": "pharmacies",
            "identity": {"name": "Test Pharmacy"},
            "offers": [],
        }
        trg = {"payload": {}}

        # Message with medical taboo for pharmacy category
        bad_med_body = "Namaste — try our herbal cough syrup for a 100% cure with zero risk. Reply YES."
        is_valid, reason = GroundingValidator.validate(
            body=bad_med_body,
            category=cat_pharmacy,
            merchant=mer_pharmacy,
            trigger=trg,
            customer=None,
            cta="binary_yes_no",
        )
        self.assertFalse(is_valid, "Medical claim must be rejected when category slug is 'pharmacies'")
        self.assertIn("contains_medical_claim", reason)

    # -----------------------------------------------------------------------
    # P. Unsupported date claim is rejected (provenance-oriented)
    # -----------------------------------------------------------------------
    def test_p_unsupported_date_claim_is_rejected(self):
        # 1. Unverified date in context without 2026-05-20
        bad_date_body = "Hi Priya, Dr. Meera has an open slot on 2026-05-20. Reply YES to book."
        is_valid, reason = GroundingValidator.validate(
            body=bad_date_body,
            category=self.cat_dentist,
            merchant=self.mer_meera,
            trigger={"payload": {}},
            customer=self.cust_priya,
            cta="binary_yes_no",
        )
        self.assertFalse(is_valid, "Unsupported calendar date '2026-05-20' must be rejected")
        self.assertIn("unsupported_date", reason)

        # 2. Date mentioned only in arbitrary text/comment field must NOT authorize the date
        trg_arbitrary_text_date = {"payload": {"doctor_note": "Patient mentioned last visited 2026-05-20 for consultation"}}
        is_valid_arb, reason_arb = GroundingValidator.validate(
            body=bad_date_body,
            category=self.cat_dentist,
            merchant=self.mer_meera,
            trigger=trg_arbitrary_text_date,
            customer=self.cust_priya,
            cta="binary_yes_no",
        )
        self.assertFalse(is_valid_arb, "Arbitrary text field with date must NOT authorize an appointment slot date")
        self.assertIn("unsupported_date", reason_arb)

        # 3. Grounded slot date (Wed 5 Nov) present in trigger available_slots must pass
        trg_with_slot = {
            "payload": {
                "available_slots": [
                    {"iso": "2026-11-05T18:00:00+05:30", "label": "Wed 5 Nov, 6pm"},
                    {"iso": "2026-11-06T17:00:00+05:30", "label": "Thu 6 Nov, 5pm"},
                ]
            }
        }
        good_date_body = "Hi Priya, 2 slots ready: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. Reply 1 for Wed, 2 for Thu, or tell us a time."
        is_valid2, reason2 = GroundingValidator.validate(
            body=good_date_body,
            category=self.cat_dentist,
            merchant=self.mer_meera,
            trigger=trg_with_slot,
            customer=self.cust_priya,
            cta="multi_choice_slot",
        )
        self.assertTrue(is_valid2, f"Verified slot dates should pass, got: {reason2}")

    # -----------------------------------------------------------------------
    # Q. Unsupported operational/delivery claim is rejected (provenance-oriented)
    # -----------------------------------------------------------------------
    def test_q_unsupported_operational_claim_is_rejected(self):
        cat_general = {"slug": "retail", "voice": {"vocab_taboo": []}, "offer_catalog": []}
        mer_general = {"merchant_id": "m_test_retail", "category_slug": "retail", "identity": {"name": "Test Store"}, "offers": []}

        # 1. Operational commitment without verified offer or policy
        bad_op_body = "Hi Priya, we promise instant delivery for all orders. Reply YES."
        is_valid, reason = GroundingValidator.validate(
            body=bad_op_body,
            category=cat_general,
            merchant=mer_general,
            trigger={"payload": {}},
            customer=self.cust_priya,
            cta="binary_yes_no",
        )
        self.assertFalse(is_valid, "Unsupported operational commitment 'promise instant delivery' must be rejected")
        self.assertTrue(
            "unsupported_operational_commitment" in reason or "unsupported_delivery_claim" in reason,
            f"Expected operational rejection, got: {reason}",
        )

        # 2. Free delivery claim when only random review mentions delivery
        mer_with_review = {
            "merchant_id": "m_test_retail2",
            "category_slug": "retail",
            "identity": {"name": "Test Store"},
            "offers": [],
            "reviews": [{"comment": "delivery took 2 hours"}],  # Mere substring in reviews!
        }
        bad_deliv_body = "Hi Priya, we offer free home delivery on all orders. Reply YES to order."
        is_valid_deliv, reason_deliv = GroundingValidator.validate(
            body=bad_deliv_body,
            category=cat_general,
            merchant=mer_with_review,
            trigger={"payload": {}},
            customer=self.cust_priya,
            cta="binary_yes_no",
        )
        self.assertFalse(is_valid_deliv, "Mere mention of delivery in reviews must NOT authorize free home delivery claim")
        self.assertIn("unsupported_delivery_claim", reason_deliv)

        # 3. Explicit verified offer for free home delivery MUST authorize the claim
        mer_with_offer = {
            "merchant_id": "m_test_retail3",
            "category_slug": "retail",
            "identity": {"name": "Test Store"},
            "offers": [{"id": "o_deliv_01", "title": "Free Home Delivery > ₹499", "status": "active"}],
        }
        good_deliv_body = "Hi Priya, Free home delivery available on orders over ₹499. Reply YES to order."
        is_valid_deliv2, reason_deliv2 = GroundingValidator.validate(
            body=good_deliv_body,
            category=cat_general,
            merchant=mer_with_offer,
            trigger={"payload": {}},
            customer=self.cust_priya,
            cta="binary_yes_no",
        )
        self.assertTrue(is_valid_deliv2, f"Verified delivery offer should authorize delivery claim, got: {reason_deliv2}")

    # -----------------------------------------------------------------------
    # R. Concurrent identical tick calls produce at most one action
    # -----------------------------------------------------------------------
    async def test_r_concurrent_identical_ticks_produce_at_most_one_action(self):
        trg_concurrent = {
            "id": "trg_concurrent_race",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 5,
            "suppression_key": "supp:concurrent_race",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_concurrent_race", 1, trg_concurrent, self.now_iso)

        # Launch two identical evaluate_triggers calls concurrently
        task1 = self.engine.evaluate_triggers(["trg_concurrent_race"], self.now_iso, self.state)
        task2 = self.engine.evaluate_triggers(["trg_concurrent_race"], self.now_iso, self.state)
        res1, res2 = await asyncio.gather(task1, task2)

        total_actions = len(res1) + len(res2)
        self.assertLessEqual(total_actions, 1, f"Expected at most 1 action across concurrent ticks, got {total_actions}")
        self.assertEqual(total_actions, 1, "Exactly one concurrent tick should claim and commit the action")

        # Verify state remains valid: suppression and cooldown active
        is_supp = await self.state.is_suppressed("supp:concurrent_race", self.now_dt, merchant_id="m_001_drmeera")
        is_cooldown = await self.state.is_merchant_in_cooldown("m_001_drmeera", self.now_dt)
        self.assertTrue(is_supp, "Suppression must be active after commit")
        self.assertTrue(is_cooldown, "Merchant cooldown must be active after commit")

        # A subsequent tick must be suppressed
        tick3_now = (self.now_dt + timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
        res3 = await self.engine.evaluate_triggers(["trg_concurrent_race"], tick3_now, self.state)
        self.assertEqual(len(res3), 0, "Subsequent tick must be suppressed")

    # -----------------------------------------------------------------------
    # S. Different merchants + same suppression key can both proceed concurrently
    # -----------------------------------------------------------------------
    async def test_s_concurrent_different_merchants_same_suppression_key_both_proceed(self):
        # Register second merchant (Dr. Bharat)
        mer_bharat = {
            "merchant_id": "m_002_bharat",
            "category_slug": "dentists",
            "identity": {"name": "Bharat Dental Care", "owner_first_name": "Bharat", "locality": "Bandra"},
            "offers": [],
        }
        await self.state.put_context("merchant", "m_002_bharat", 1, mer_bharat, self.now_iso)

        # Trigger for Merchant A and Trigger for Merchant B sharing the SAME suppression key
        shared_supp_key = "research:dentists:shared_week_17"
        trg_a = {
            "id": "trg_shared_mer_a",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "urgency": 4,
            "suppression_key": shared_supp_key,
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        trg_b = {
            "id": "trg_shared_mer_b",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_002_bharat",
            "urgency": 4,
            "suppression_key": shared_supp_key,
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_001_fluoride"},
        }
        await self.state.put_context("trigger", "trg_shared_mer_a", 1, trg_a, self.now_iso)
        await self.state.put_context("trigger", "trg_shared_mer_b", 1, trg_b, self.now_iso)

        # Concurrently evaluate triggers for Merchant A and Merchant B
        task_a = self.engine.evaluate_triggers(["trg_shared_mer_a"], self.now_iso, self.state)
        task_b = self.engine.evaluate_triggers(["trg_shared_mer_b"], self.now_iso, self.state)
        res_a, res_b = await asyncio.gather(task_a, task_b)

        self.assertEqual(len(res_a), 1, "Merchant A must produce an action")
        self.assertEqual(len(res_b), 1, "Merchant B must also produce an action despite shared suppression key")
        self.assertEqual(res_a[0]["merchant_id"], "m_001_drmeera")
        self.assertEqual(res_b[0]["merchant_id"], "m_002_bharat")

if __name__ == "__main__":
    unittest.main()

