#!/usr/bin/env python3
"""
Validation Suite: Merchant-Scoped Suppression & Fresh-Data Stress Tests
======================================================================
1. Merchant-Scoped Suppression:
   - Create Merchant A and Merchant B.
   - Give both the same suppression_key.
   - Fire the relevant trigger for Merchant A.
   - Verify Merchant A is suppressed afterward.
   - Verify Merchant B is still eligible for the same suppression_key.

2. Four Fresh-Data Stress Tests:
   A. NEW DIGEST: Unseen digest item, verified grounded composition.
   B. UPDATED PERFORMANCE: Updated delta_pct / metric reflected dynamically.
   C. NEW TRIGGER KIND: Generic handling with sufficient facts; safe restraint when insufficient.
   D. FRESH CUSTOMER + recall_due: Fresh customer and slots used without fabrication.
"""

import asyncio
from datetime import datetime, timezone
import re

from bot import ThreadSafeState
from decision_engine import DecisionEngine, GroundingValidator


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

INTERNAL_JARGON_RE = re.compile(
    r"\b(trg_\w+|m_\d{3}\w*|c_\d{3}\w*|conv_\w+|suppression_key|payload|context_id)\b",
    re.IGNORECASE,
)


def verify_message_quality(body: str, expected_cta: str, cta: str):
    assert body and body.strip(), "Body must not be empty"
    assert not INTERNAL_JARGON_RE.search(body), f"Internal jargon leaked: {body}"
    assert "http" not in body.lower(), f"URL leaked in body: {body}"
    assert cta == expected_cta, f"Expected CTA '{expected_cta}', got '{cta}'"


# ---------------------------------------------------------------------------
# 1. MERCHANT-SCOPED SUPPRESSION REGRESSION
# ---------------------------------------------------------------------------

async def test_merchant_scoped_suppression():
    print("\n--- [TASK 1] MERCHANT-SCOPED SUPPRESSION REGRESSION ---")
    state = ThreadSafeState()
    engine = DecisionEngine()
    now_iso = "2026-04-26T10:00:00Z"

    # Common category
    await state.put_context(
        scope="category",
        context_id="dentists",
        version=1,
        payload={
            "slug": "dentists",
            "display_name": "Dentists",
            "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed"]},
            "offer_catalog": [{"id": "o_clean", "title": "Dental Cleaning", "value": "299"}],
            "digest": [{
                "id": "d_common",
                "title": "Fluoride varnish efficacy",
                "source": "JIDA 2026",
                "trial_n": 1000,
                "patient_segment": "children",
            }],
        },
        delivered_at=now_iso,
    )

    # Merchant A
    await state.put_context(
        scope="merchant",
        context_id="mer_alpha",
        version=1,
        payload={
            "merchant_id": "mer_alpha",
            "category_slug": "dentists",
            "identity": {"name": "Alpha Dental Care", "owner_first_name": "Aarav", "city": "Delhi"},
            "offers": [{"id": "o_a", "title": "Dental Cleaning", "status": "active"}],
        },
        delivered_at=now_iso,
    )

    # Merchant B
    await state.put_context(
        scope="merchant",
        context_id="mer_beta",
        version=1,
        payload={
            "merchant_id": "mer_beta",
            "category_slug": "dentists",
            "identity": {"name": "Beta Dental Studio", "owner_first_name": "Bhavna", "city": "Mumbai"},
            "offers": [{"id": "o_b", "title": "Dental Cleaning", "status": "active"}],
        },
        delivered_at=now_iso,
    )

    shared_supp_key = "research:dentists:shared_batch_2026"

    # Trigger for Merchant A
    await state.put_context(
        scope="trigger",
        context_id="trg_shared_alpha",
        version=1,
        payload={
            "id": "trg_shared_alpha",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "mer_alpha",
            "customer_id": None,
            "urgency": 3,
            "suppression_key": shared_supp_key,
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_common"},
        },
        delivered_at=now_iso,
    )

    # Trigger for Merchant B (same suppression_key)
    await state.put_context(
        scope="trigger",
        context_id="trg_shared_beta",
        version=1,
        payload={
            "id": "trg_shared_beta",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "mer_beta",
            "customer_id": None,
            "urgency": 3,
            "suppression_key": shared_supp_key,
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_common"},
        },
        delivered_at=now_iso,
    )

    # Step 1: Fire trigger for Merchant A
    actions_1 = await engine.evaluate_triggers(
        available_triggers=["trg_shared_alpha"],
        now_iso=now_iso,
        state=state,
    )
    assert len(actions_1) == 1, f"Expected 1 action for Merchant A, got {len(actions_1)}"
    assert actions_1[0]["merchant_id"] == "mer_alpha"
    print("  [PASS] Step 1: Merchant A action generated successfully.")

    # Step 2: Verify Merchant A is suppressed on repeated trigger
    actions_2 = await engine.evaluate_triggers(
        available_triggers=["trg_shared_alpha"],
        now_iso=now_iso,
        state=state,
    )
    assert len(actions_2) == 0, f"Expected Merchant A to be suppressed, got {len(actions_2)} actions"
    print("  [PASS] Step 2: Merchant A is properly suppressed on repeat.")

    # Step 3: Verify Merchant B is STILL ELIGIBLE for the same suppression_key
    actions_3 = await engine.evaluate_triggers(
        available_triggers=["trg_shared_beta"],
        now_iso=now_iso,
        state=state,
    )
    assert len(actions_3) == 1, (
        f"REGRESSION: Merchant B was incorrectly suppressed by Merchant A's suppression_key! "
        f"Expected 1 action, got {len(actions_3)}"
    )
    assert actions_3[0]["merchant_id"] == "mer_beta"
    print("  [PASS] Step 3: Merchant B remains eligible and received action for the shared suppression_key.")


# ---------------------------------------------------------------------------
# 2. FOUR FRESH-DATA STRESS TESTS
# ---------------------------------------------------------------------------

async def test_fresh_data_a_new_digest():
    print("\n--- [TASK 2A] FRESH DATA: UNSEEN RESEARCH DIGEST ---")
    state = ThreadSafeState()
    engine = DecisionEngine()
    now_iso = "2026-04-26T10:00:00Z"

    # Pushing newly published clinical trial
    new_digest_item = {
        "id": "d_unseen_chlorhexidine_2026",
        "title": "Chlorhexidine 0.2% pre-rinse reduces surgical bacteremia by 44%",
        "source": "Lancet Dental Infectious Diseases Nov 2026",
        "trial_n": 1850,
        "patient_segment": "implant_and_extraction_patients",
    }

    await state.put_context(
        scope="category",
        context_id="dentists",
        version=2,
        payload={
            "slug": "dentists",
            "display_name": "Dentists",
            "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed"]},
            "offer_catalog": [{"id": "o_clean", "title": "Dental Cleaning", "value": "299"}],
            "digest": [new_digest_item],
        },
        delivered_at=now_iso,
    )

    await state.put_context(
        scope="merchant",
        context_id="mer_dr_vikram",
        version=1,
        payload={
            "merchant_id": "mer_dr_vikram",
            "category_slug": "dentists",
            "identity": {"name": "Vikram Multispeciality Dental", "owner_first_name": "Vikram", "city": "Bangalore"},
            "offers": [{"id": "o_v", "title": "Dental Consultation", "status": "active"}],
        },
        delivered_at=now_iso,
    )

    trg_id = "trg_unseen_digest_vikram"
    await state.put_context(
        scope="trigger",
        context_id=trg_id,
        version=1,
        payload={
            "id": trg_id,
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "mer_dr_vikram",
            "customer_id": None,
            "urgency": 2,
            "suppression_key": "research:dentists:unseen_w44",
            "expires_at": "2026-05-15T00:00:00Z",
            "payload": {"category": "dentists", "top_item_id": "d_unseen_chlorhexidine_2026"},
        },
        delivered_at=now_iso,
    )

    actions = await engine.evaluate_triggers([trg_id], now_iso, state)
    assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"
    action = actions[0]
    body = action["body"]

    # Verify grounded facts from new digest
    assert "Lancet Dental Infectious Diseases Nov 2026" in body, "Missing source citation"
    assert "1,850-patient trial" in body or "1850" in body, "Missing trial N"
    assert "44%" in body, "Missing 44% metric"
    assert "implant and extraction patients" in body, "Missing patient segment"
    verify_message_quality(body, "open_ended", action["cta"])
    print(f"  [PASS] Fresh Digest output verified: \"{body[:80]}...\"")


async def test_fresh_data_b_updated_performance():
    print("\n--- [TASK 2B] FRESH DATA: UPDATED PERFORMANCE METRIC ---")
    state = ThreadSafeState()
    engine = DecisionEngine()
    now_iso = "2026-04-26T10:00:00Z"

    await state.put_context(
        scope="category",
        context_id="gyms",
        version=1,
        payload={
            "slug": "gyms",
            "display_name": "Gyms & Fitness",
            "voice": {"tone": "coach_direct", "vocab_taboo": ["miracle"]},
        },
        delivered_at=now_iso,
    )

    await state.put_context(
        scope="merchant",
        context_id="mer_iron_gym",
        version=1,
        payload={
            "merchant_id": "mer_iron_gym",
            "category_slug": "gyms",
            "identity": {"name": "Iron Clad Gym", "owner_first_name": "Rohan", "city": "Pune"},
            "offers": [{"id": "o_gym", "title": "Annual Membership", "status": "active"}],
        },
        delivered_at=now_iso,
    )

    # Dynamic performance dip: 45% drop in inquiries (different from seed 30% views)
    trg_id = "trg_dynamic_perf_drop"
    await state.put_context(
        scope="trigger",
        context_id=trg_id,
        version=1,
        payload={
            "id": trg_id,
            "scope": "merchant",
            "kind": "perf_dip",
            "merchant_id": "mer_iron_gym",
            "customer_id": None,
            "urgency": 4,
            "suppression_key": "perf_dip:iron_clad:inquiries:2026",
            "expires_at": "2026-05-10T00:00:00Z",
            "payload": {"metric": "inquiries", "delta_pct": -0.45, "window": "7d"},
        },
        delivered_at=now_iso,
    )

    actions = await engine.evaluate_triggers([trg_id], now_iso, state)
    assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"
    action = actions[0]
    body = action["body"]

    # Verify updated metric and percentage are dynamically reflected, not seed defaults
    assert "45%" in body, f"Expected updated 45% in body, got: {body}"
    assert "inquiries" in body, f"Expected updated metric 'inquiries' in body, got: {body}"
    assert "30%" not in body, f"Stale seed percentage (30%) found in body: {body}"
    verify_message_quality(body, "binary_yes_no", action["cta"])
    print(f"  [PASS] Dynamic Performance output verified: \"{body[:80]}...\"")


async def test_fresh_data_c_new_trigger_kind():
    print("\n--- [TASK 2C] FRESH DATA: UNSEEN TRIGGER KIND & RESTRAINT ---")
    state = ThreadSafeState()
    engine = DecisionEngine()
    now_iso = "2026-04-26T10:00:00Z"

    await state.put_context(
        scope="category",
        context_id="restaurants",
        version=1,
        payload={
            "slug": "restaurants",
            "display_name": "Restaurants",
            "voice": {"tone": "foodie_peer", "vocab_taboo": ["best in town"]},
        },
        delivered_at=now_iso,
    )

    await state.put_context(
        scope="merchant",
        context_id="mer_taste_of_punjab",
        version=1,
        payload={
            "merchant_id": "mer_taste_of_punjab",
            "category_slug": "restaurants",
            "identity": {"name": "Taste of Punjab", "owner_first_name": "Harpreet", "city": "Chandigarh"},
            "offers": [{"id": "o_thali", "title": "Special Dal Makhani Combo", "status": "active"}],
        },
        delivered_at=now_iso,
    )

    # Subtest C1: Unseen trigger kind WITH sufficient facts -> Generic Grounded Output
    trg_c1 = "trg_unseen_food_fest"
    await state.put_context(
        scope="trigger",
        context_id=trg_c1,
        version=1,
        payload={
            "id": trg_c1,
            "scope": "merchant",
            "kind": "local_culinary_festival",
            "merchant_id": "mer_taste_of_punjab",
            "customer_id": None,
            "urgency": 3,
            "suppression_key": "supp:fest_punjab_2026",
            "expires_at": "2026-05-01T00:00:00Z",
            "payload": {
                "headline": "Sector 17 Street Food Carnival",
                "recommended_action": "offer a special delivery combo for evening visitors",
            },
        },
        delivered_at=now_iso,
    )

    actions_c1 = await engine.evaluate_triggers([trg_c1], now_iso, state)
    assert len(actions_c1) == 1, f"Expected 1 generic action for well-specified unseen trigger, got {len(actions_c1)}"
    body_c1 = actions_c1[0]["body"]
    assert "Sector 17 Street Food Carnival" in body_c1
    assert "offer a special delivery combo for evening visitors" in body_c1
    verify_message_quality(body_c1, "binary_yes_no", actions_c1[0]["cta"])
    print(f"  [PASS] C1 (Sufficient facts): Generic output verified: \"{body_c1[:80]}...\"")

    # Subtest C2: Unseen trigger kind WITH INSUFFICIENT facts -> Safe Restraint (No action)
    trg_c2 = "trg_unseen_cryptic_signal"
    await state.put_context(
        scope="trigger",
        context_id=trg_c2,
        version=1,
        payload={
            "id": trg_c2,
            "scope": "merchant",
            "kind": "unsupported_cryptic_event",
            "merchant_id": "mer_taste_of_punjab",
            "customer_id": None,
            "urgency": 2,
            "suppression_key": "supp:cryptic_2026",
            "expires_at": "2026-05-01T00:00:00Z",
            "payload": {"mystery_code": 992},  # Insufficient facts to justify sending
        },
        delivered_at=now_iso,
    )

    actions_c2 = await engine.evaluate_triggers([trg_c2], now_iso, state)
    assert len(actions_c2) == 0, f"Expected safe restraint (0 actions), got {len(actions_c2)}"
    print("  [PASS] C2 (Insufficient facts): Bot exercised safe restraint (0 actions, no hallucination).")


async def test_fresh_data_d_fresh_customer_recall_due():
    print("\n--- [TASK 2D] FRESH DATA: FRESH CUSTOMER + RECALL_DUE ---")
    state = ThreadSafeState()
    engine = DecisionEngine()
    now_iso = "2026-04-26T10:00:00Z"

    await state.put_context(
        scope="category",
        context_id="dentists",
        version=1,
        payload={
            "slug": "dentists",
            "display_name": "Dentists",
            "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed"]},
            "offer_catalog": [{"id": "o_clean", "title": "Comprehensive Dental Hygiene", "value": "499"}],
        },
        delivered_at=now_iso,
    )

    mer_id = "mer_smile_craft"
    await state.put_context(
        scope="merchant",
        context_id=mer_id,
        version=1,
        payload={
            "merchant_id": mer_id,
            "category_slug": "dentists",
            "identity": {"name": "Smile Craft Clinic", "owner_first_name": "Sneha", "city": "Gurgaon"},
            "offers": [{"id": "o_sc", "title": "Comprehensive Dental Hygiene @ ₹499", "status": "active"}],
        },
        delivered_at=now_iso,
    )

    # Fresh customer context injected mid-test
    cust_id = "c_unseen_ananya_sharma"
    await state.put_context(
        scope="customer",
        context_id=cust_id,
        version=1,
        payload={
            "customer_id": cust_id,
            "merchant_id": mer_id,
            "identity": {"name": "Ananya Sharma", "language_pref": "en"},
        },
        delivered_at=now_iso,
    )

    trg_id = "trg_unseen_recall_ananya"
    await state.put_context(
        scope="trigger",
        context_id=trg_id,
        version=1,
        payload={
            "id": trg_id,
            "scope": "customer",
            "kind": "recall_due",
            "merchant_id": mer_id,
            "customer_id": cust_id,
            "urgency": 3,
            "suppression_key": "recall:c_unseen_ananya:6mo",
            "expires_at": "2026-05-20T00:00:00Z",
            "payload": {
                "service_due": "Dental Hygiene",
                "available_slots": [
                    {"label": "Sat 14 Nov, 11am"},
                    {"label": "Sun 15 Nov, 4pm"},
                ],
            },
        },
        delivered_at=now_iso,
    )

    actions = await engine.evaluate_triggers([trg_id], now_iso, state)
    assert len(actions) == 1, f"Expected 1 action, got {len(actions)}"
    action = actions[0]
    body = action["body"]

    # Verify newly supplied facts are used accurately
    assert "Ananya Sharma" in body, f"Fresh customer name missing in body: {body}"
    assert "Sat 14 Nov, 11am" in body, f"Slot 1 missing in body: {body}"
    assert "Sun 15 Nov, 4pm" in body, f"Slot 2 missing in body: {body}"
    assert "Comprehensive Dental Hygiene @ ₹499" in body, f"Active merchant offer missing: {body}"
    assert action["send_as"] == "merchant_on_behalf"
    verify_message_quality(body, "multi_choice_slot", action["cta"])
    print(f"  [PASS] Fresh Customer Recall output verified: \"{body[:80]}...\"")


async def main():
    await test_merchant_scoped_suppression()
    await test_fresh_data_a_new_digest()
    await test_fresh_data_b_updated_performance()
    await test_fresh_data_c_new_trigger_kind()
    await test_fresh_data_d_fresh_customer_recall_due()
    print("\n============================================================")
    print("ALL 5 VALIDATION AND FRESH-DATA STRESS TESTS PASSED!")
    print("============================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
