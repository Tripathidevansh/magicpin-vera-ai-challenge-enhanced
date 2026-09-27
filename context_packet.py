#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — Verified Context Packet (Phase 3)
=============================================================
Extracts a minimal, clean, non-leaky factual context packet from candidate
data to feed into the LLM prompt.

Strict isolation:
- No internal identifiers, suppression keys, or unrelated store dumps
- Only relevant, grounded facts for this specific trigger
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerifiedContextPacket:
    # Scope & Roles
    scope: str  # "merchant" or "customer"
    send_as: str  # "vera" or "merchant_on_behalf"

    # Category Context
    category_slug: str
    category_display_name: str
    tone: str
    register: str
    vocab_allowed: List[str]
    vocab_taboo: List[str]
    code_mix: str

    # Merchant Context
    owner_name: str
    merchant_name: str
    locality: str
    city: str
    merchant_languages: List[str]
    active_offers: List[str]
    signals: List[str]
    performance_summary: Dict[str, Any]

    # Trigger Context
    trigger_kind: str
    urgency: int
    deadline: Optional[str]
    why_now: str
    trigger_facts: Dict[str, Any]

    # Customer Context (Optional)
    customer_name: Optional[str] = None
    customer_language_pref: Optional[str] = None
    customer_slots: List[str] = field(default_factory=list)
    customer_state: Optional[str] = None
    customer_history: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate(cls, candidate: Any) -> VerifiedContextPacket:
        cat = candidate.category or {}
        mer = candidate.merchant or {}
        trg = candidate.trigger or {}
        cust = candidate.customer or {}

        # Category Voice extraction
        voice = cat.get("voice", {})
        tone = voice.get("tone", "collegial")
        register = voice.get("register", "peer")
        vocab_allowed = voice.get("vocab_allowed", [])[:10]
        vocab_taboo = voice.get("vocab_taboo", [])[:8]
        code_mix = voice.get("code_mix", "english_primary")

        # Merchant Extraction
        ident = mer.get("identity", {})
        owner = ident.get("owner_first_name") or ident.get("name", "there")
        mer_name = ident.get("name", "our business")
        locality = ident.get("locality", "")
        city = ident.get("city", "")
        languages = ident.get("languages", ["en"])

        # Active Offers
        active_offers = [
            o.get("title", "") for o in mer.get("offers", []) if o.get("status") == "active"
        ]
        if not active_offers and cat.get("offer_catalog"):
            active_offers = [
                o.get("title", "") for o in cat.get("offer_catalog", [])[:3]
            ]

        # Performance summary
        perf = mer.get("performance", {})
        perf_summary = {
            "views_30d": perf.get("views"),
            "calls_30d": perf.get("calls"),
            "ctr": perf.get("ctr"),
            "delta_7d": perf.get("delta_7d", {}),
        }

        # Trigger Extraction
        payload = trg.get("payload", {})
        deadline = payload.get("deadline_iso") or trg.get("expires_at")
        scope = trg.get("scope", "merchant")
        send_as = "merchant_on_behalf" if candidate.customer_id else "vera"

        # Customer Extraction
        cust_name = None
        cust_lang = None
        cust_slots = []
        cust_state = None
        cust_hist = {}

        if cust:
            c_ident = cust.get("identity", {})
            cust_name = c_ident.get("name")
            cust_lang = c_ident.get("language_pref", "en")
            cust_state = cust.get("state")
            cust_hist = {
                "visits_total": cust.get("relationship", {}).get("visits_total"),
                "last_visit": cust.get("relationship", {}).get("last_visit"),
                "previous_focus": payload.get("previous_focus"),
            }

        # Slots from trigger payload if present
        for slot in payload.get("available_slots", []):
            if isinstance(slot, dict) and "label" in slot:
                cust_slots.append(slot["label"])

        return cls(
            scope=scope,
            send_as=send_as,
            category_slug=cat.get("slug", candidate.category_slug),
            category_display_name=cat.get("display_name", candidate.category_slug.title()),
            tone=tone,
            register=register,
            vocab_allowed=vocab_allowed,
            vocab_taboo=vocab_taboo,
            code_mix=code_mix,
            owner_name=owner,
            merchant_name=mer_name,
            locality=locality,
            city=city,
            merchant_languages=languages,
            active_offers=active_offers,
            signals=mer.get("signals", [])[:5],
            performance_summary=perf_summary,
            trigger_kind=candidate.trigger_kind,
            urgency=candidate.urgency,
            deadline=deadline,
            why_now=candidate.why_now,
            trigger_facts=payload,
            customer_name=cust_name,
            customer_language_pref=cust_lang,
            customer_slots=cust_slots,
            customer_state=cust_state,
            customer_history=cust_hist,
        )

    def to_clean_prompt_dict(self) -> Dict[str, Any]:
        """Convert into clean JSON structure for LLM prompt."""
        packet: Dict[str, Any] = {
            "business_vertical": self.category_display_name,
            "tone_profile": {
                "tone": self.tone,
                "register": self.register,
                "code_mix": self.code_mix,
                "preferred_vocabulary": self.vocab_allowed,
                "taboo_words_strictly_forbidden": self.vocab_taboo,
            },
            "business_details": {
                "owner_name": self.owner_name,
                "business_name": self.merchant_name,
                "locality": self.locality,
                "city": self.city,
                "languages": self.merchant_languages,
                "verified_active_offers": self.active_offers,
                "recent_signals": self.signals,
            },
            "trigger_facts": {
                "kind": self.trigger_kind,
                "urgency_level": self.urgency,
                "deadline": self.deadline,
                "details": self.trigger_facts,
            },
            "messaging_mode": self.send_as,
        }

        if self.customer_name:
            packet["customer_details"] = {
                "name": self.customer_name,
                "language_preference": self.customer_language_pref,
                "available_slots": self.customer_slots,
                "relationship_state": self.customer_state,
                "history": self.customer_history,
            }

        return packet
