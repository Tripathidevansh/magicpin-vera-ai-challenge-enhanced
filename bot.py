#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — Candidate Bot
==========================================
HTTP service exposing the 5 official challenge endpoints with thread-safe
in-memory state management and deterministic decision engine.

Endpoints:
- GET  /v1/healthz   : High-speed liveness and context inventory probe
- GET  /v1/metadata  : Bot metadata and model declaration
- POST /v1/context   : Atomic context ingestion and version tracking (idempotent / 409)
- POST /v1/tick      : Proactive trigger evaluation & deterministic composition
- POST /v1/reply     : Multi-turn simulated merchant/customer replies
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from decision_engine import DecisionEngine, parse_iso_utc
from reply_state_machine import ConversationState, ReplyStateMachine

# ---------------------------------------------------------------------------
# Global Application & Uptime Tracker
# ---------------------------------------------------------------------------

START_TIME = time.time()
app = FastAPI(
    title="magicpin Vera AI Candidate",
    description="Merchant AI Assistant for magicpin WhatsApp engagement",
    version="0.2.0-phase2",
)

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# ---------------------------------------------------------------------------
# Data Models / Schemas
# ---------------------------------------------------------------------------

class ContextPushRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any] = Field(default_factory=dict)
    delivered_at: Optional[str] = None


class TickRequest(BaseModel):
    now: str
    available_triggers: List[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# ---------------------------------------------------------------------------
# Typed State Structures
# ---------------------------------------------------------------------------

@dataclass
class ContextItem:
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: Optional[str]
    stored_at: str


@dataclass
class SuppressionRecord:
    suppression_key: str
    expires_at_iso: Optional[str] = None
    created_at_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )


@dataclass
class MerchantCooldownRecord:
    merchant_id: str
    last_sent_at_iso: str


@dataclass
class ConversationTurn:
    turn_number: int
    from_role: str
    message: str
    timestamp_iso: str


@dataclass
class ConversationRecord:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    trigger_id: Optional[str] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    state: str = ConversationState.NEW  # NEW, OUTREACH_SENT, WAITING, ENGAGED, COMMITTED, OPTED_OUT, CLOSED
    auto_reply_count: int = 0
    created_at_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    updated_at_iso: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )


class ThreadSafeState:
    """
    Central thread-safe in-memory store for contexts, versions, suppressions,
    cooldowns, and conversation replay histories.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._contexts: Dict[Tuple[str, str], ContextItem] = {}
        self._counts: Dict[str, int] = {
            "category": 0,
            "merchant": 0,
            "customer": 0,
            "trigger": 0,
        }
        self._suppressions: Dict[str, SuppressionRecord] = {}
        self._cooldowns: Dict[str, MerchantCooldownRecord] = {}
        self._conversations: Dict[str, ConversationRecord] = {}
        self._opted_out_merchants: Set[str] = set()
        self._merchant_auto_replies: Dict[str, int] = {}
        self._conv_auto_replies: Dict[str, int] = {}
        self._reserved_merchants: Set[str] = set()
        self._reserved_suppressions: Set[str] = set()

    def get_context_counts(self) -> Dict[str, int]:
        """Fast non-blocking read for /v1/healthz probe."""
        return dict(self._counts)

    def get_context(self, scope: str, context_id: str) -> Optional[ContextItem]:
        """Lookup a context item by (scope, context_id)."""
        return self._contexts.get((scope, context_id))

    async def put_context(
        self,
        scope: str,
        context_id: str,
        version: int,
        payload: Dict[str, Any],
        delivered_at: Optional[str],
    ) -> Tuple[bool, Optional[int], str]:
        """
        Atomically update or insert a context entry.
        Returns:
            (success: bool, current_or_stale_version: int | None, ack_or_reason: str)
        """
        async with self._lock:
            key = (scope, context_id)
            current = self._contexts.get(key)

            if current is not None and current.version >= version:
                return False, current.version, "stale_version"

            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            item = ContextItem(
                scope=scope,
                context_id=context_id,
                version=version,
                payload=payload,
                delivered_at=delivered_at,
                stored_at=now_iso,
            )

            is_new = current is None
            self._contexts[key] = item

            if is_new:
                self._counts[scope] = self._counts.get(scope, 0) + 1

            ack_id = f"ack_{context_id}_v{version}"
            return True, version, ack_id

    async def is_suppressed(
        self,
        key: str,
        now_dt: Optional[datetime] = None,
        merchant_id: Optional[str] = None,
    ) -> bool:
        """Check if a suppression key is currently active for a merchant or globally, or currently reserved."""
        async with self._lock:
            scoped_key = f"{merchant_id}:{key}" if merchant_id else None
            # Check reservation (strictly merchant-scoped when merchant_id provided)
            if scoped_key and scoped_key in self._reserved_suppressions:
                return True
            if not merchant_id and key in self._reserved_suppressions:
                return True

            rec = None
            matched_key = None
            if merchant_id:
                if scoped_key in self._suppressions:
                    rec = self._suppressions[scoped_key]
                    matched_key = scoped_key
                elif key in self._suppressions:
                    rec = self._suppressions[key]
                    matched_key = key
            else:
                if key in self._suppressions:
                    rec = self._suppressions[key]
                    matched_key = key
                else:
                    suffix = f":{key}"
                    for k, v in self._suppressions.items():
                        if k.endswith(suffix):
                            rec = v
                            matched_key = k
                            break

            if not rec:
                return False
            if rec.expires_at_iso and now_dt:
                exp_dt = parse_iso_utc(rec.expires_at_iso)
                if exp_dt and now_dt >= exp_dt:
                    if matched_key and matched_key in self._suppressions:
                        del self._suppressions[matched_key]
                    return False
            return True

    async def add_suppression(
        self,
        key: str,
        expires_at_iso: Optional[str] = None,
        merchant_id: Optional[str] = None,
    ) -> None:
        async with self._lock:
            record = SuppressionRecord(
                suppression_key=key, expires_at_iso=expires_at_iso
            )
            if merchant_id:
                self._suppressions[f"{merchant_id}:{key}"] = record
            else:
                self._suppressions[key] = record

    async def is_merchant_opted_out(self, merchant_id: str) -> bool:
        async with self._lock:
            return merchant_id in self._opted_out_merchants

    async def opt_out_merchant(self, merchant_id: str) -> None:
        async with self._lock:
            self._opted_out_merchants.add(merchant_id)

    async def is_merchant_in_cooldown(self, merchant_id: str, now_dt: datetime) -> bool:
        """Check if merchant received a proactive outreach in the last simulated 24h or is currently reserved."""
        async with self._lock:
            if merchant_id in self._reserved_merchants:
                return True
            rec = self._cooldowns.get(merchant_id)
            if not rec:
                return False
            last_sent_dt = parse_iso_utc(rec.last_sent_at_iso)
            if last_sent_dt and (now_dt - last_sent_dt).total_seconds() < 86400:
                return True
            return False

    async def reserve_candidate(
        self,
        merchant_id: str,
        suppression_key: Optional[str] = None,
        now_dt: Optional[datetime] = None,
    ) -> bool:
        """
        Atomically reserve a merchant and suppression key for proactive outreach.
        Guarantees that concurrent ticks cannot select the same merchant or trigger,
        while maintaining strict merchant-scoped isolation for shared suppression keys.
        """
        async with self._lock:
            # 1. Check merchant cooldown or active reservation
            if merchant_id in self._reserved_merchants:
                return False
            rec = self._cooldowns.get(merchant_id)
            if rec and now_dt:
                last_sent_dt = parse_iso_utc(rec.last_sent_at_iso)
                if last_sent_dt and (now_dt - last_sent_dt).total_seconds() < 86400:
                    return False

            # 2. Check suppression key (strictly merchant-scoped)
            scoped_supp = f"{merchant_id}:{suppression_key}" if suppression_key else None
            if scoped_supp:
                if scoped_supp in self._reserved_suppressions:
                    return False

                supp_rec = self._suppressions.get(scoped_supp) or self._suppressions.get(suppression_key)
                if supp_rec:
                    if not supp_rec.expires_at_iso or not now_dt:
                        return False
                    exp_dt = parse_iso_utc(supp_rec.expires_at_iso)
                    if exp_dt and now_dt < exp_dt:
                        return False

            # 3. Claim reservation
            self._reserved_merchants.add(merchant_id)
            if scoped_supp:
                self._reserved_suppressions.add(scoped_supp)
            return True

    async def release_reservation(
        self,
        merchant_id: str,
        suppression_key: Optional[str] = None,
    ) -> None:
        """Release reservation if composition fails or is aborted."""
        async with self._lock:
            self._reserved_merchants.discard(merchant_id)
            if suppression_key:
                scoped_supp = f"{merchant_id}:{suppression_key}"
                self._reserved_suppressions.discard(scoped_supp)

    async def commit_action(self, action: Dict[str, Any], now_dt: datetime) -> None:
        """Atomically persist suppression key, merchant cooldown, and conversation state."""
        async with self._lock:
            now_iso = now_dt.isoformat().replace("+00:00", "Z")
            supp_key = action.get("suppression_key")
            mid = action.get("merchant_id")
            cid = action.get("customer_id")
            tid = action.get("trigger_id")
            conv_id = action.get("conversation_id")

            # 1. Suppression key (scoped per merchant to prevent cross-merchant suppression)
            if supp_key:
                exp_iso = (now_dt + timedelta(days=7)).isoformat().replace("+00:00", "Z")
                record = SuppressionRecord(
                    suppression_key=supp_key,
                    expires_at_iso=exp_iso,
                    created_at_iso=now_iso,
                )
                if mid:
                    self._suppressions[f"{mid}:{supp_key}"] = record
                else:
                    self._suppressions[supp_key] = record

            # 2. Merchant cooldown
            if mid:
                self._cooldowns[mid] = MerchantCooldownRecord(
                    merchant_id=mid,
                    last_sent_at_iso=now_iso,
                )

            # 3. Conversation record
            if conv_id:
                conv = ConversationRecord(
                    conversation_id=conv_id,
                    merchant_id=mid,
                    customer_id=cid,
                    trigger_id=tid,
                    state=ConversationState.OUTREACH_SENT,
                    created_at_iso=now_iso,
                    updated_at_iso=now_iso,
                    turns=[
                        ConversationTurn(
                            turn_number=1,
                            from_role=action.get("send_as", "vera"),
                            message=action.get("body", ""),
                            timestamp_iso=now_iso,
                        )
                    ],
                )
                self._conversations[conv_id] = conv

            # 4. Clear reservations now that permanent records are established
            if mid:
                self._reserved_merchants.discard(mid)
            if supp_key and mid:
                self._reserved_suppressions.discard(f"{mid}:{supp_key}")

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationRecord]:
        async with self._lock:
            return self._conversations.get(conversation_id)

    async def set_conversation_state(self, conversation_id: str, conv_state: str) -> None:
        async with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv:
                conv.state = conv_state
                conv.updated_at_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    async def record_auto_reply(self, merchant_id: Optional[str], conversation_id: str) -> int:
        async with self._lock:
            cnt = 0
            if merchant_id:
                self._merchant_auto_replies[merchant_id] = self._merchant_auto_replies.get(merchant_id, 0) + 1
                cnt = self._merchant_auto_replies[merchant_id]
            else:
                self._conv_auto_replies[conversation_id] = self._conv_auto_replies.get(conversation_id, 0) + 1
                cnt = self._conv_auto_replies[conversation_id]

            conv = self._conversations.get(conversation_id)
            if conv:
                conv.auto_reply_count = cnt
            return cnt

    async def reset_auto_reply(self, merchant_id: Optional[str], conversation_id: str) -> None:
        async with self._lock:
            if merchant_id and merchant_id in self._merchant_auto_replies:
                self._merchant_auto_replies[merchant_id] = 0
            if conversation_id in self._conv_auto_replies:
                self._conv_auto_replies[conversation_id] = 0

    async def record_reply_turn(
        self,
        conversation_id: str,
        merchant_id: Optional[str],
        customer_id: Optional[str],
        from_role: str,
        message: str,
        received_at: str,
        turn_number: int,
    ) -> ConversationRecord:
        """Atomically record incoming conversation turns."""
        async with self._lock:
            conv = self._conversations.get(conversation_id)
            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            if conv is None:
                conv = ConversationRecord(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    state=ConversationState.NEW,
                    created_at_iso=now_iso,
                    updated_at_iso=now_iso,
                )
                self._conversations[conversation_id] = conv
            else:
                if merchant_id and not conv.merchant_id:
                    conv.merchant_id = merchant_id
                if customer_id and not conv.customer_id:
                    conv.customer_id = customer_id
                conv.updated_at_iso = now_iso

            turn = ConversationTurn(
                turn_number=turn_number,
                from_role=from_role,
                message=message,
                timestamp_iso=received_at,
            )
            conv.turns.append(turn)
            return conv


# Global state & decision engine instances
state = ThreadSafeState()
decision_engine = DecisionEngine()

# ---------------------------------------------------------------------------
# Global Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensure malformed requests return standard JSON 400 Bad Request."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "accepted": False,
            "reason": "malformed_request",
            "details": str(exc),
        },
    )

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    """
    Liveness and readiness probe polled by judge harness.
    Must respond in <2 seconds with context inventory counts.
    """
    uptime = int(time.time() - START_TIME)
    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "contexts_loaded": state.get_context_counts(),
    }


@app.get("/v1/metadata")
async def metadata():
    """Bot identity and model declaration."""
    return {
        "team_name": "Vera AI Team",
        "team_members": ["Devansh Tripathi"],
        "model": "gemini-2.5-flash + llama-3.3-70b-versatile",
        "approach": "Deterministic decision engine + grounded hybrid LLM composer + multi-turn state machine",
        "contact_email": "candidate@magicpin.com",
        "version": "0.2.0-phase2",
        "submitted_at": "2026-04-26T08:00:00Z",
    }


@app.post("/v1/context")
async def push_context(request: Request, body: ContextPushRequest):
    """
    Atomic context ingestion endpoint.
    - Enforces 500 KB payload size cap (challenge rate limits & operational constraints)
    - Validates scope
    - Enforces idempotency and version bump rules
    - Returns HTTP 409 Conflict if version <= current version
    """
    # 500 KB payload cap enforcement per Challenge Testing Brief Section 5
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > 500 * 1024:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": "payload_too_large",
                "details": f"Context payload size ({content_length} bytes) exceeds 500 KB cap.",
            },
        )
    raw_body = await request.body()
    if len(raw_body) > 500 * 1024:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": "payload_too_large",
                "details": f"Context payload size ({len(raw_body)} bytes) exceeds 500 KB cap.",
            },
        )

    if body.scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": "invalid_scope",
                "details": f"Scope '{body.scope}' is invalid. Allowed: {sorted(list(VALID_SCOPES))}",
            },
        )

    success, current_version, ack_or_reason = await state.put_context(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
        delivered_at=body.delivered_at,
    )

    if not success:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": current_version,
            },
        )

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "accepted": True,
        "ack_id": ack_or_reason,
        "stored_at": now_iso,
    }


@app.post("/v1/tick")
async def tick(body: TickRequest):
    """
    Periodic simulated time tick endpoint.
    Runs trigger eligibility, merchant dedup, ranking, deterministic composition,
    and grounding validation.
    """
    actions = await decision_engine.evaluate_triggers(
        available_triggers=body.available_triggers,
        now_iso=body.now,
        state=state,
    )
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    """
    Synchronous multi-turn reply endpoint.
    Records incoming turn in thread-safe state and delegates to ReplyStateMachine.
    """
    await state.record_reply_turn(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        received_at=body.received_at,
        turn_number=body.turn_number,
    )

    response = await ReplyStateMachine.handle_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        state=state,
    )

    # Record bot turn in conversation history if action is send
    if response.get("action") == "send" and response.get("body"):
        now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        await state.record_reply_turn(
            conversation_id=body.conversation_id,
            merchant_id=body.merchant_id,
            customer_id=body.customer_id,
            from_role="vera",
            message=response["body"],
            received_at=now_iso,
            turn_number=body.turn_number + 1,
        )

    return response


# ---------------------------------------------------------------------------
# Local Dev Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("bot:app", host="0.0.0.0", port=8080, log_level="info")
