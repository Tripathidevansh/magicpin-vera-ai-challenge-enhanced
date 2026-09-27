#!/usr/bin/env python3
"""
Unit tests for Conversational Reply State Machine (Phase 4)
===========================================================
Tests cover:
- TEST 1: Initial outreach -> repeated auto replies -> eventual CLOSED/terminal state
- TEST 2: Previous action proposal -> Merchant: 'Ok lets do it. Whats next?' -> concrete next-step action (no wait)
- TEST 3: Merchant: 'Stop messaging me. This is useless spam.' -> OPTED_OUT -> terminal/exit behavior -> suppressed
- TEST 4: Variants & punctuation: 'unsubscribe', 'yes, let's proceed', 'what's next?'
- TEST 5: Negative feedback like 'That's too expensive' does NOT become OPT_OUT
"""

import asyncio
from datetime import datetime, timezone

from bot import ThreadSafeState, ConversationRecord
from decision_engine import DecisionEngine
from reply_state_machine import (
    ConversationState,
    ReplyStateMachine,
    is_auto_reply,
    is_commitment,
    is_opt_out_or_hostile,
)


# ---------------------------------------------------------------------------
# Detector Unit Tests
# ---------------------------------------------------------------------------

def test_opt_out_detector():
    positives = [
        "Stop messaging me. This is useless spam.",
        "stop messaging me",
        "stop contacting me",
        "don't message me",
        "don't contact me",
        "do not text me",
        "unsubscribe",
        "UNSUBSCRIBE!!!",
        "remove me",
        "no more messages",
        "this is spam",
        "useless spam",
        "not interested",
        "leave me alone",
        "opt out",
        "opt-out",
        "STOP",
        "stop.",
        "band karo",
        "message mat karo",
    ]
    for msg in positives:
        assert is_opt_out_or_hostile(msg), f"Expected '{msg}' to be detected as opt-out"

    negatives = [
        "That's too expensive",
        "Can we do 15% discount instead?",
        "We are closed on Mondays",
        "Call me tomorrow at 4pm",
        "I need to check with my partner first",
        "Ok lets do it. Whats next?",
        "Thank you for contacting us!",
    ]
    for msg in negatives:
        assert not is_opt_out_or_hostile(msg), f"Expected '{msg}' NOT to be detected as opt-out"


def test_auto_reply_detector():
    positives = [
        "Thank you for contacting us! Our team will respond shortly.",
        "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
        "This is an automated reply.",
        "I am currently out of office.",
        "We are closed right now.",
        "Our business hours are 9am to 6pm.",
    ]
    for msg in positives:
        assert is_auto_reply(msg), f"Expected '{msg}' to be detected as auto-reply"

    negatives = [
        "Ok lets do it. Whats next?",
        "Yes, proceed.",
        "Stop messaging me.",
        "That's too expensive",
    ]
    for msg in negatives:
        assert not is_auto_reply(msg), f"Expected '{msg}' NOT to be detected as auto-reply"


def test_commitment_detector():
    positives = [
        "Ok lets do it. Whats next?",
        "ok let's do it",
        "lets do it",
        "yes",
        "sounds good",
        "proceed",
        "go ahead",
        "what's next",
        "what next",
        "what's next?",
        "how do we proceed",
        "let's proceed",
        "yes, let's proceed",
        "YES, LET'S PROCEED!",
        "confirm",
        "chalo",
        "theek hai",
    ]
    for msg in positives:
        assert is_commitment(msg), f"Expected '{msg}' to be detected as commitment"

    negatives = [
        "That's too expensive",
        "Stop messaging me. This is useless spam.",
        "not ok, price is high",
        "not sure about this",
        "no",
    ]
    for msg in negatives:
        assert not is_commitment(msg), f"Expected '{msg}' NOT to be detected as commitment"


# ---------------------------------------------------------------------------
# State Machine & Integration Tests
# ---------------------------------------------------------------------------

async def test_case_1_auto_reply_termination():
    """
    TEST 1: Initial outreach -> repeated auto replies -> eventual CLOSED/terminal state.
    """
    state = ThreadSafeState()
    conv_id = "conv_auto_test_1"
    mid = "mer_auto_1"

    # Initial outreach committed
    now_dt = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)
    action = {
        "conversation_id": conv_id,
        "merchant_id": mid,
        "trigger_id": "trig_test_1",
        "body": "Want to run an IPL pizza combo?",
    }
    await state.commit_action(action, now_dt)

    conv = await state.get_conversation(conv_id)
    assert conv.state == ConversationState.OUTREACH_SENT

    auto_msg = "Thank you for contacting us! Our team will respond shortly."

    # Turn 1 auto-reply: Wait 14400s (initial backoff)
    res1 = await ReplyStateMachine.handle_reply(
        conversation_id=conv_id,
        merchant_id=mid,
        customer_id=None,
        from_role="merchant",
        message=auto_msg,
        turn_number=2,
        state=state,
    )
    assert res1["action"] == "wait"
    assert res1["wait_seconds"] == 14400
    conv = await state.get_conversation(conv_id)
    assert conv.state == ConversationState.WAITING

    # Turn 2 auto-reply (judge repeats across conversation turns): Threshold reached -> CLOSED
    res2 = await ReplyStateMachine.handle_reply(
        conversation_id="conv_auto_test_2",
        merchant_id=mid,
        customer_id=None,
        from_role="merchant",
        message=auto_msg,
        turn_number=3,
        state=state,
    )
    assert res2["action"] == "end"
    assert "auto-reply loop detected" in res2["rationale"].lower()


async def test_case_2_commitment_and_grounded_next_step():
    """
    TEST 2: Previous action proposal -> Merchant: 'Ok lets do it. Whats next?'
    -> NOT WAIT -> concrete next-step response/action.
    """
    state = ThreadSafeState()
    conv_id = "conv_intent_test_1"
    mid = "mer_intent_1"

    # Initial proposal proposing Swiggy banner + Insta story
    now_dt = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)
    action = {
        "conversation_id": conv_id,
        "merchant_id": mid,
        "trigger_id": "trig_test_2",
        "body": "Want me to draft the Swiggy banner + Insta story for the match tonight?",
    }
    await state.commit_action(action, now_dt)

    merchant_msg = "Ok lets do it. Whats next?"
    res = await ReplyStateMachine.handle_reply(
        conversation_id=conv_id,
        merchant_id=mid,
        customer_id=None,
        from_role="merchant",
        message=merchant_msg,
        turn_number=2,
        state=state,
    )

    # Must NOT be wait
    assert res["action"] == "send"
    body = res["body"]

    # Must contain actioning words
    actioning = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]
    assert any(w in body.lower() for w in actioning), f"Body lacks actioning language: '{body}'"

    # Must NOT contain qualifying hesitation
    qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
    assert not any(w in body.lower() for w in qualifying), f"Body contains qualifying hesitation: '{body}'"

    # Verify state transition to COMMITTED
    conv = await state.get_conversation(conv_id)
    assert conv.state == ConversationState.COMMITTED


async def test_case_3_hostile_opt_out_and_suppression():
    """
    TEST 3: Merchant: 'Stop messaging me. This is useless spam.'
    -> OPTED_OUT -> terminal/exit behavior -> subsequent tick/reply suppressed.
    """
    state = ThreadSafeState()
    conv_id = "conv_hostile_test_1"
    mid = "mer_hostile_1"

    now_dt = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)
    action = {
        "conversation_id": conv_id,
        "merchant_id": mid,
        "trigger_id": "trig_test_3",
        "body": "Quick idea for your dental clinic.",
    }
    await state.commit_action(action, now_dt)

    hostile_msg = "Stop messaging me. This is useless spam."
    res = await ReplyStateMachine.handle_reply(
        conversation_id=conv_id,
        merchant_id=mid,
        customer_id=None,
        from_role="merchant",
        message=hostile_msg,
        turn_number=2,
        state=state,
    )

    # Must be terminal exit action
    assert res["action"] == "end"
    assert "opted out" in res["rationale"].lower()

    # Verify merchant is recorded as opted out
    assert await state.is_merchant_opted_out(mid)

    # Verify conversation state is OPTED_OUT
    conv = await state.get_conversation(conv_id)
    assert conv.state == ConversationState.OPTED_OUT

    # Verify subsequent reply for this merchant/conversation remains suppressed
    res_subsequent = await ReplyStateMachine.handle_reply(
        conversation_id=conv_id,
        merchant_id=mid,
        customer_id=None,
        from_role="merchant",
        message="Are you still there?",
        turn_number=3,
        state=state,
    )
    assert res_subsequent["action"] == "end"

    # Verify DecisionEngine suppresses proactive outreach for opted out merchant
    engine = DecisionEngine()
    actions = await engine.evaluate_triggers(
        available_triggers=["trig_test_3"],
        now_iso="2026-04-26T12:00:00Z",
        state=state,
    )
    assert len(actions) == 0


async def test_case_4_negative_feedback_not_opt_out():
    """
    TEST 4: Ordinary negative feedback ('That's too expensive') does NOT become OPT_OUT.
    """
    state = ThreadSafeState()
    conv_id = "conv_neg_test_1"
    mid = "mer_neg_1"

    now_dt = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)
    action = {
        "conversation_id": conv_id,
        "merchant_id": mid,
        "trigger_id": "trig_test_4",
        "body": "Special weekend promotion proposal.",
    }
    await state.commit_action(action, now_dt)

    feedback_msg = "That's too expensive"
    res = await ReplyStateMachine.handle_reply(
        conversation_id=conv_id,
        merchant_id=mid,
        customer_id=None,
        from_role="merchant",
        message=feedback_msg,
        turn_number=2,
        state=state,
    )

    # Must NOT end or opt-out
    assert res["action"] == "send"
    assert not await state.is_merchant_opted_out(mid)

    conv = await state.get_conversation(conv_id)
    assert conv.state == ConversationState.ENGAGED


if __name__ == "__main__":
    asyncio.run(test_case_1_auto_reply_termination())
    asyncio.run(test_case_2_commitment_and_grounded_next_step())
    asyncio.run(test_case_3_hostile_opt_out_and_suppression())
    asyncio.run(test_case_4_negative_feedback_not_opt_out())
    test_opt_out_detector()
    test_auto_reply_detector()
    test_commitment_detector()
    print("ALL REPLY STATE MACHINE UNIT TESTS PASSED!")
