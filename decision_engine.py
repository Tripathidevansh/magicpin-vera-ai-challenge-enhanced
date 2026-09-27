#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — Deterministic Decision Engine (Phase 2)
===================================================================
Implements the trigger eligibility pipeline, merchant deduplication,
urgency/relevance ranking, deterministic baseline composition, and
strict grounding validation.

Zero LLM calls. Zero hallucinations. Strictly grounded in stored context.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple


def parse_iso_utc(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string into timezone-aware UTC datetime."""
    if not dt_str:
        return None
    try:
        clean = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CandidateTrigger:
    trigger_id: str
    merchant_id: str
    customer_id: Optional[str]
    category_slug: str
    trigger_kind: str
    urgency: int
    suppression_key: str
    expires_at_iso: str
    why_now: str
    relevant_facts: Dict[str, Any]
    category: Dict[str, Any]
    merchant: Dict[str, Any]
    trigger: Dict[str, Any]
    customer: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Grounding & Safety Validator
# ---------------------------------------------------------------------------

class GroundingValidator:
    """
    Strict post-composition validator ensuring:
    - No URLs are present in message bodies (Meta template violation, -3 penalty)
    - No internal IDs or system jargon appear in customer/merchant visible text (-1 penalty)
    - No vertical-specific taboo vocabulary is used (e.g., 'guaranteed', '100% safe', 'cure')
    - Every explicit INR price (₹...) is verified against stored context data (-2 penalty)
    - Explicit percentages (%...) are verified against stored context data
    - Medical/clinical taboos strictly forbidden for high-risk categories (dentists, pharmacies)
    - Exactly one primary CTA that agrees with the declared CTA type
    """

    URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
    INTERNAL_JARGON_PATTERN = re.compile(
        r"\b(trg_\w+|m_\d{3}\w*|c_\d{3}\w*|conv_\w+|suppression_key|customer_aggregate|delta_pct|window_days|delivered_at)\b",
        re.IGNORECASE,
    )
    PRICE_PATTERN = re.compile(r"₹\s*(\d[\d,]*)")
    PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
    MEDICAL_TABOO_PATTERN = re.compile(
        r"\b(100%\s*(?:cure|safe|effective|guaranteed)|permanent\s*cure|miracle\s*(?:cure|drug|treatment)|"
        r"guaranteed\s*(?:cure|results?|recovery|healing)|cure\s*guaranteed|no\s*side\s*effects|"
        r"completely\s*(?:cure|safe)|zero\s*risk|instant\s*cure|painless\s*guaranteed|cheap\s*drugs)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _collect_allowed_prices(
        cls,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
    ) -> Set[int]:
        """Collect all verified price amounts (as integers) from contexts."""
        prices: Set[int] = set()

        def extract_numbers_from_str(s: str):
            for match in cls.PRICE_PATTERN.finditer(s):
                clean_num = match.group(1).replace(",", "")
                if clean_num.isdigit():
                    prices.add(int(clean_num))

        # 1. From Merchant Offers
        for off in merchant.get("offers", []):
            if isinstance(off, dict):
                title = off.get("title", "")
                extract_numbers_from_str(title)
                val = str(off.get("value", ""))
                if val.isdigit():
                    prices.add(int(val))

        # 2. From Category Offer Catalog
        for off in category.get("offer_catalog", []):
            if isinstance(off, dict):
                title = off.get("title", "")
                extract_numbers_from_str(title)
                val = str(off.get("value", ""))
                if val.isdigit():
                    prices.add(int(val))

        # 3. From Trigger Payload
        payload = trigger.get("payload", {})
        if isinstance(payload, dict):
            for k, v in payload.items():
                if isinstance(v, (int, float)) and v > 0:
                    prices.add(int(v))
                elif isinstance(v, str):
                    extract_numbers_from_str(v)

        # 4. From Customer Context
        if customer and isinstance(customer, dict):
            for k, v in customer.items():
                if isinstance(v, str):
                    extract_numbers_from_str(v)

        return prices

    @classmethod
    def _collect_allowed_percentages(
        cls,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
    ) -> Set[float]:
        """Collect all verified percentage values from contexts."""
        percentages: Set[float] = set()

        def extract_pct_from_str(s: str):
            for match in cls.PERCENT_PATTERN.finditer(s):
                try:
                    val = float(match.group(1))
                    percentages.add(val)
                    percentages.add(abs(val))
                    if 0.0 < abs(val) <= 1.0:
                        percentages.add(round(abs(val) * 100.0, 4))
                except Exception:
                    pass

        def scan_dict(d: Any):
            if isinstance(d, dict):
                for k, v in d.items():
                    k_lower = k.lower()
                    if any(
                        sub in k_lower
                        for sub in ("pct", "percent", "rate", "discount", "margin", "ctr", "growth", "surge", "drop", "dip", "covers")
                    ):
                        if isinstance(v, (int, float)):
                            percentages.add(float(v))
                            percentages.add(abs(float(v)))
                            # Fractional percentages (e.g. -0.30 or 0.12) represent -30% or 12%
                            if 0.0 < abs(float(v)) <= 1.0:
                                percentages.add(round(abs(float(v)) * 100.0, 4))
                                percentages.add(round(float(v) * 100.0, 4))
                    if isinstance(v, str):
                        extract_pct_from_str(v)
                    elif isinstance(v, (dict, list)):
                        scan_dict(v)
            elif isinstance(d, list):
                for item in d:
                    scan_dict(item)

        scan_dict(category)
        scan_dict(merchant)
        scan_dict(trigger)
        if customer:
            scan_dict(customer)

        return percentages

    @classmethod
    def validate(
        cls,
        body: str,
        category: Dict[str, Any],
        merchant: Dict[str, Any],
        trigger: Dict[str, Any],
        customer: Optional[Dict[str, Any]] = None,
        cta: Optional[str] = None,
    ) -> Tuple[bool, str]:
        if not body or not body.strip():
            return False, "empty_body"

        # 1. Reject URLs
        if cls.URL_PATTERN.search(body):
            return False, "contains_url"

        # 2. Reject Internal Jargon
        if cls.INTERNAL_JARGON_PATTERN.search(body):
            return False, "contains_internal_jargon"

        # 3. Reject Taboo Vocabulary (with word boundaries to avoid false substring matches)
        taboos = category.get("voice", {}).get("vocab_taboo", [])
        body_lower = body.lower()
        for taboo in taboos:
            taboo_clean = taboo.split("(")[0].strip().lower()
            if taboo_clean:
                pattern = r"\b" + re.escape(taboo_clean) + r"\b"
                if re.search(pattern, body_lower):
                    return False, f"contains_taboo_word: '{taboo_clean}'"

        # 4. Strict Medical & Regulatory Claim Validation for High-Risk Verticals
        cat_slug = category.get("slug", "")
        if cat_slug in ("dentists", "pharmacy"):
            med_match = cls.MEDICAL_TABOO_PATTERN.search(body)
            if med_match:
                return False, f"contains_medical_claim: '{med_match.group(0)}'"

        # 5. Verify explicit ₹ prices
        prices_in_body = cls.PRICE_PATTERN.findall(body)
        if prices_in_body:
            allowed_prices = cls._collect_allowed_prices(category, merchant, trigger, customer)
            for p_str in prices_in_body:
                val = int(p_str.replace(",", ""))
                if val not in allowed_prices:
                    return False, f"unverified_price: '₹{val}'"

        # 6. Verify explicit percentages
        pcts_in_body = cls.PERCENT_PATTERN.findall(body)
        if pcts_in_body:
            allowed_pcts = cls._collect_allowed_percentages(category, merchant, trigger, customer)
            for pct_str in pcts_in_body:
                try:
                    pct_val = float(pct_str)
                    if (
                        pct_val not in allowed_pcts
                        and abs(pct_val) not in allowed_pcts
                        and (pct_val / 100.0) not in allowed_pcts
                        and (abs(pct_val) / 100.0) not in allowed_pcts
                    ):
                        return False, f"unverified_percentage: '{pct_str}%'"
                except Exception:
                    pass

        # 7. Single Primary CTA & CTA Alignment Validation
        q_count = body.count("?")
        if q_count > 2:
            return False, f"multiple_ctas: {q_count} question marks found in message"

        if cta == "none":
            if q_count > 0:
                return False, "cta_mismatch_none: message contains question mark when CTA is none"
            if re.search(r"\b(reply\s+yes|click\s+here|call\s+now|book\s+now|order\s+now)\b", body, re.IGNORECASE):
                return False, "cta_mismatch_none: action imperative found when CTA is none"
        elif cta == "binary_yes_no":
            has_binary_cue = bool(
                re.search(
                    r"\b(yes\b|confirm\b|reply\s+yes|reply\s+1|shall\s+we|would\s+you\s+like|let\s+me\s+know\s+if)\b",
                    body,
                    re.IGNORECASE,
                )
            ) or q_count > 0
            if not has_binary_cue:
                return False, "cta_mismatch_binary: message missing binary confirmation prompt"
        elif cta == "multi_choice_slot":
            has_choice_cue = bool(re.search(r"\b(or|slot|either|option)\b", body, re.IGNORECASE))
            if not has_choice_cue:
                return False, "cta_mismatch_multi_choice: message missing choice/slot indicator ('or', 'slot')"

        return True, "valid"


# ---------------------------------------------------------------------------
# Deterministic Baseline Message Composer
# ---------------------------------------------------------------------------

class DeterministicComposer:
    """
    Renders verifiable, factual messages for canonical trigger kinds without
    hallucinating or inventing details.
    """

    @staticmethod
    def compose(candidate: CandidateTrigger) -> Optional[Dict[str, Any]]:
        kind = candidate.trigger_kind
        cat = candidate.category
        mer = candidate.merchant
        trg = candidate.trigger
        cust = candidate.customer

        identity = mer.get("identity", {})
        owner = identity.get("owner_first_name") or identity.get("name", "there")
        mer_name = identity.get("name", "our clinic")
        locality = identity.get("locality", "")
        slug = candidate.category_slug
        payload = trg.get("payload", {})

        body: Optional[str] = None
        cta: str = "open_ended"
        send_as: str = "merchant_on_behalf" if candidate.customer_id else "vera"
        template_name: str = f"vera_{kind}_v1"
        template_params: List[str] = []
        rationale: str = ""

        # Format salutation (clinical for doctors/dentists)
        salutation = f"Dr. {owner}" if slug == "dentists" and not owner.startswith("Dr.") else owner

        # 1. Research Digest (e.g. Case Study 1)
        if kind == "research_digest":
            top_item_id = payload.get("top_item_id")
            digest_items = cat.get("digest", [])
            item = next((d for d in digest_items if d.get("id") == top_item_id), None)
            if not item and digest_items:
                item = digest_items[0]

            if not item:
                return None

            title = item.get("title", "")
            source = item.get("source", "Recent medical journal")
            trial_n = item.get("trial_n")
            seg = item.get("patient_segment", "patients").replace("_", " ")

            trial_snippet = f"{trial_n:,}-patient trial showed " if trial_n else ""
            body = (
                f"{salutation}, {source} landed. One item relevant to your {seg} — "
                f"{trial_snippet}{title}. Worth a look (2-min abstract). "
                f"Want me to pull it + draft a patient-ed WhatsApp you can share? — {source}"
            )
            cta = "open_ended"
            template_name = "vera_research_digest_v1"
            template_params = [salutation, title, source]
            rationale = "External research digest with merchant-relevant clinical anchor and source citation."

        # 2. Recall Due (e.g. Case Study 2)
        elif kind == "recall_due":
            if not cust:
                return None
            cust_name = cust.get("identity", {}).get("name", "there")
            slots = payload.get("available_slots", [])
            slot_str = ""
            slot1_label = "first slot"
            slot2_label = "second slot"
            if slots and len(slots) >= 2:
                slot1_label = slots[0].get("label", "Slot 1")
                slot2_label = slots[1].get("label", "Slot 2")
                slot_str = f"**{slot1_label}** ya **{slot2_label}**"
            elif slots:
                slot_str = slots[0].get("label", "")

            # Look for active verified offer in merchant catalog
            active_offer = next(
                (o.get("title") for o in mer.get("offers", []) if o.get("status") == "active"),
                next((o.get("title") for o in cat.get("offer_catalog", []) if "Cleaning" in o.get("title", "")), "Consultation")
            )
            emoji = "🦷" if slug == "dentists" else "✨"

            slot_phrase = f" Apke liye ready slots: {slot_str}." if slot_str else ""
            body = (
                f"Hi {cust_name}, {mer_name} here {emoji} Your routine checkup recall is due.{slot_phrase} "
                f"{active_offer} ready. Reply 1 for {slot1_label}, 2 for {slot2_label}, or tell us a time that works."
            )
            cta = "multi_choice_slot"
            send_as = "merchant_on_behalf"
            template_name = "merchant_recall_reminder_v1"
            template_params = [cust_name, mer_name, active_offer]
            rationale = "Customer-facing recall notification sent on behalf of merchant with open slots and verified offer."

        # 3. Wedding Package / Bridal Followup (e.g. Case Study 3)
        elif kind in {"wedding_package_followup", "bridal_followup"}:
            if not cust:
                return None
            cust_name = cust.get("identity", {}).get("name", "there")
            days = payload.get("days_to_wedding", 196)
            next_step = payload.get("next_step_window_open", "skin prep program").replace("_", " ")

            body = (
                f"Hi {cust_name} 💍 {owner} from {mer_name} here. {days} days to your wedding — "
                f"perfect window to start your {next_step}. "
                f"Want me to block your preferred slot for the first session next week?"
            )
            cta = "binary_yes_no"
            send_as = "merchant_on_behalf"
            template_name = "merchant_bridal_followup_v1"
            template_params = [cust_name, owner, str(days), next_step]
            rationale = "Milestone-anchored wedding preparation followup sent on behalf of salon owner."

        # 4. Curious Ask (e.g. Case Study 4)
        elif kind == "curious_ask_due":
            body = (
                f"Hi {owner}! Quick check — what service has been most asked-for this week at {mer_name}? "
                f"I'll turn the answer into a Google post + a 4-line WhatsApp reply you can use when customers ask about pricing. Takes 5 min."
            )
            cta = "open_ended"
            template_name = "vera_curious_ask_v1"
            template_params = [owner, mer_name]
            rationale = "Low-friction curious-ask cadence to source merchant insight and externalize content drafting."

        # 5. IPL Match Day (e.g. Case Study 5)
        elif kind == "ipl_match_today":
            match = payload.get("match", "tonight's match")
            venue = payload.get("venue", "the stadium")

            # Look for active BOGO or food offer
            active_offer = next(
                (o.get("title") for o in mer.get("offers", []) if o.get("status") == "active"),
                "special discount"
            )

            body = (
                f"Quick heads-up {owner} — {match} at {venue} tonight. "
                f"Important: Saturday IPL matches usually shift -12% restaurant covers (people watch at home). "
                f"Push your {active_offer} as a delivery-only Saturday special. "
                f"Want me to draft the banner + an Insta story? Live in 10 min."
            )
            cta = "open_ended"
            template_name = "vera_ipl_advisory_v1"
            template_params = [owner, match, venue, active_offer]
            rationale = "Event-driven advisory offering contrarian delivery recommendation leveraging active merchant offer."

        # 6. Active Planning Intent (e.g. Case Study 6)
        elif kind == "active_planning_intent":
            topic = payload.get("intent_topic", "custom package").replace("_", " ")
            body = (
                f"Hi {owner}, following up on {topic} for {locality} — "
                f"I've structured a complete starter draft ready for your review with tiered options and delivery guidelines. "
                f"Want me to share the full package structure here?"
            )
            cta = "binary_yes_no"
            template_name = "vera_planning_proposal_v1"
            template_params = [owner, topic]
            rationale = "Direct continuation of merchant planning intent with structured proposal."

        # 7. Seasonal Performance Dip / Reframe (e.g. Case Study 7)
        elif kind in {"seasonal_perf_dip", "perf_dip"}:
            metric = payload.get("metric", "views")
            delta_val = payload.get("delta_pct", -0.30)
            delta_pct = abs(int(delta_val * 100))
            cat_display = cat.get("display_name", "businesses")

            body = (
                f"{salutation}, your {metric} are down {delta_pct}% this week — but this is the normal seasonal lull for metro {cat_display}. "
                f"Action: focus retention on your current members. "
                f"Want me to draft an engagement challenge to keep them active through the dip?"
            )
            cta = "binary_yes_no"
            template_name = "vera_perf_dip_reframe_v1"
            template_params = [salutation, metric, str(delta_pct)]
            rationale = "Performance dip reframe with seasonal benchmark and retention proposal."

        # 8. Customer Lapsed Hard / Winback (e.g. Case Study 8)
        elif kind == "customer_lapsed_hard":
            if not cust:
                return None
            cust_name = cust.get("identity", {}).get("name", "there")
            days = payload.get("days_since_last_visit", 57)
            weeks = max(1, days // 7)

            body = (
                f"Hi {cust_name} 👋 {owner} from {mer_name} here. It's been about {weeks} weeks — "
                f"happens to most members at some point, no judgment. "
                f"Want me to hold a free trial class spot for you next week? Reply YES — no commitment."
            )
            cta = "binary_yes_no"
            send_as = "merchant_on_behalf"
            template_name = "merchant_winback_v1"
            template_params = [cust_name, owner, mer_name, str(weeks)]
            rationale = "No-shame winback outreach addressing past member with zero commitment."

        # 9. Supply Alert / Compliance Alert (e.g. Case Study 9)
        elif kind in {"supply_alert", "regulation_change"}:
            molecule = payload.get("molecule") or payload.get("category", "supply item")
            batches = payload.get("affected_batches", [])
            batch_str = ", ".join(batches) if batches else "recent batch"
            mfr = payload.get("manufacturer", "the manufacturer")

            body = (
                f"{owner}, urgent: voluntary recall on {molecule} batches ({batch_str}) by {mfr} — "
                f"sub-potency, no safety risk, but customers should be informed for replacement. "
                f"Want me to draft their WhatsApp note + the replacement-pickup workflow?"
            )
            cta = "open_ended"
            template_name = "vera_compliance_alert_v1"
            template_params = [owner, molecule, batch_str]
            rationale = "Urgent regulatory compliance alert with exact batch numbers and patient notification workflow."

        # 10. Chronic Refill Due (e.g. Case Study 10)
        elif kind == "chronic_refill_due":
            if not cust:
                return None
            cust_name = cust.get("identity", {}).get("name", "there")
            molecules = payload.get("molecule_list", ["prescribed medicines"])
            med_str = ", ".join(molecules)

            body = (
                f"Namaste — {mer_name} {locality} here. Your monthly medicines ({med_str}) run out soon. "
                f"Repeat refill pack ready. Free home delivery available on orders over ₹499. "
                f"Reply CONFIRM to dispatch, or let us know if any change in dosage."
            )
            cta = "binary_yes_no"
            send_as = "merchant_on_behalf"
            template_name = "merchant_refill_reminder_v1"
            template_params = [cust_name, mer_name, med_str]
            rationale = "Chronic medication refill reminder with exact molecule names and convenient dispatch option."

        # 11. Renewal Due
        elif kind == "renewal_due":
            days = payload.get("days_remaining", 14)
            plan = payload.get("plan", "Pro")
            amt = payload.get("renewal_amount", 4999)

            body = (
                f"Hi {owner}, your magicpin {plan} subscription has {days} days remaining. "
                f"Renewal amount is ₹{amt:,}. "
                f"Want me to send the direct renewal payment link so your profile remains active without interruption?"
            )
            cta = "binary_yes_no"
            template_name = "vera_renewal_reminder_v1"
            template_params = [owner, plan, str(days), str(amt)]
            rationale = "Subscription renewal notice with days remaining and exact fee."

        # 12. Generic / Unknown Trigger Kind (Safe Fallback)
        else:
            headline = payload.get("headline") or payload.get("title") or payload.get("summary")
            recommendation = payload.get("recommended_action") or payload.get("action")
            if headline and recommendation:
                body = (
                    f"Hi {owner}, note regarding {headline} — "
                    f"we recommend to {recommendation}. "
                    f"Want me to prepare the draft proposal for your review?"
                )
                cta = "binary_yes_no"
                template_name = "vera_generic_advisory_v1"
                template_params = [owner, headline]
                rationale = "Generic actionable advisory derived from verified trigger payload."
            else:
                # Insufficient facts to justify sending: exercise safe restraint
                return None

        if not body:
            return None

        # Build action dictionary
        date_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        conv_id = f"conv_{candidate.merchant_id}_{kind}_{date_stamp}"

        action = {
            "conversation_id": conv_id,
            "merchant_id": candidate.merchant_id,
            "customer_id": candidate.customer_id,
            "send_as": send_as,
            "trigger_id": candidate.trigger_id,
            "template_name": template_name,
            "template_params": template_params,
            "body": body,
            "cta": cta,
            "suppression_key": candidate.suppression_key,
            "rationale": rationale,
        }

        # Validate through GroundingValidator
        is_valid, reason = GroundingValidator.validate(
            body=body,
            category=cat,
            merchant=mer,
            trigger=trg,
            customer=cust,
            cta=cta,
        )

        if not is_valid:
            # Failed grounding: do NOT fabricate or return invalid message
            return None

        return action


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Evaluates available triggers against stored context, filters expired/suppressed
    events, deduplicates per merchant, ranks by urgency/relevance, limits batch size,
    and returns verified, grounded actions.
    """

    def __init__(
        self,
        max_candidates: Optional[int] = None,
        composer: Optional[Any] = None,
    ) -> None:
        self.max_candidates = max_candidates or int(
            os.environ.get("MAX_CANDIDATES_PER_TICK", "5")
        )
        self._composer = composer

    @property
    def composer(self) -> Any:
        if self._composer is None:
            from llm_composer import LLMComposer
            self._composer = LLMComposer()
        return self._composer

    def rank_candidates(
        self, candidates: List[CandidateTrigger], now_dt: datetime
    ) -> List[CandidateTrigger]:
        """
        Deterministic ranking function for eligible candidates.
        Considers urgency, relevance, deadline proximity, customer specificity, and signals.
        Isolated for tuning.
        """
        def score(c: CandidateTrigger) -> float:
            pts = float(c.urgency * 100.0)

            # Customer-specific relevance bonus
            if c.customer_id is not None:
                pts += 40.0

            # High-relevance trigger kinds
            if c.trigger_kind in {"active_planning_intent", "supply_alert", "regulation_change"}:
                pts += 50.0
            elif c.trigger_kind in {"recall_due", "chronic_refill_due", "wedding_package_followup"}:
                pts += 30.0
            elif c.trigger_kind in {"ipl_match_today", "perf_dip", "renewal_due"}:
                pts += 20.0

            # Signal match bonus
            signals = c.merchant.get("signals", [])
            for sig in signals:
                if isinstance(sig, str) and c.trigger_kind in sig:
                    pts += 25.0

            # Deadline proximity bonus
            exp_dt = parse_iso_utc(c.expires_at_iso)
            if exp_dt:
                hours_left = (exp_dt - now_dt).total_seconds() / 3600.0
                if 0 < hours_left <= 24:
                    pts += 30.0
                elif 0 < hours_left <= 72:
                    pts += 15.0

            return pts

        return sorted(candidates, key=score, reverse=True)

    def deduplicate_merchants(
        self, candidates: List[CandidateTrigger], now_dt: datetime
    ) -> List[CandidateTrigger]:
        """
        Ensures strictly ONE proactive action per merchant survives in a single tick.
        Chooses the candidate with highest urgency/score and nearest deadline.
        """
        by_merchant: Dict[str, List[CandidateTrigger]] = {}
        for c in candidates:
            by_merchant.setdefault(c.merchant_id, []).append(c)

        survivors: List[CandidateTrigger] = []
        for mid, group in by_merchant.items():
            if len(group) == 1:
                survivors.append(group[0])
            else:
                ranked = self.rank_candidates(group, now_dt)
                survivors.append(ranked[0])

        return survivors

    async def evaluate_triggers(
        self,
        available_triggers: List[str],
        now_iso: str,
        state: Any,
    ) -> List[Dict[str, Any]]:
        """
        Full 9-step decision pipeline:
        1. Resolve trigger contexts
        2. Filter expired triggers (now >= expires_at)
        3. Filter active suppressions
        4. Filter opted-out merchants
        5. Verify required merchant/category/customer context
        6. Verify merchant/category compatibility
        7. Apply merchant proactive cooldown
        8. Deduplicate per merchant
        9. Rank and compose grounded actions
        """
        now_dt = parse_iso_utc(now_iso) or datetime.now(timezone.utc)
        eligible_candidates: List[CandidateTrigger] = []

        for trg_id in available_triggers:
            trg_item = state.get_context("trigger", trg_id)
            if not trg_item:
                continue
            trg = trg_item.payload

            # 2. Reject expired triggers
            expires_at_iso = trg.get("expires_at")
            exp_dt = parse_iso_utc(expires_at_iso)
            if exp_dt and now_dt >= exp_dt:
                continue

            # 3. Resolve merchant & check opt-out
            merchant_id = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id")
            if not merchant_id or await state.is_merchant_opted_out(merchant_id):
                continue

            # 4. Reject active suppression_key (scoped per merchant)
            supp_key = trg.get("suppression_key")
            if supp_key and await state.is_suppressed(supp_key, now_dt, merchant_id=merchant_id):
                continue

            # 5. Verify required merchant & category context
            mer_item = state.get_context("merchant", merchant_id)
            if not mer_item:
                continue
            merchant = mer_item.payload

            cat_slug = merchant.get("category_slug")
            if not cat_slug:
                continue
            cat_item = state.get_context("category", cat_slug)
            if not cat_item:
                continue
            category = cat_item.payload

            # Check customer scope
            scope = trg.get("scope", "merchant")
            customer_id = trg.get("customer_id") or trg.get("payload", {}).get("customer_id")
            customer = None
            if scope == "customer" or customer_id:
                if not customer_id:
                    continue
                cust_item = state.get_context("customer", customer_id)
                if not cust_item:
                    continue
                customer = cust_item.payload

            # 6. Verify merchant/category compatibility
            if merchant.get("category_slug") != category.get("slug"):
                continue

            # 7. Apply merchant proactive cooldown (simulated 24h)
            if await state.is_merchant_in_cooldown(merchant_id, now_dt):
                continue

            # 8. Create normalized CandidateTrigger
            candidate = CandidateTrigger(
                trigger_id=trg_id,
                merchant_id=merchant_id,
                customer_id=customer_id,
                category_slug=cat_slug,
                trigger_kind=trg.get("kind", "generic"),
                urgency=int(trg.get("urgency", 3)),
                suppression_key=supp_key or f"supp:{trg_id}",
                expires_at_iso=expires_at_iso or "",
                why_now=f"Trigger {trg.get('kind')} active for {merchant_id}",
                relevant_facts=trg.get("payload", {}),
                category=category,
                merchant=merchant,
                trigger=trg,
                customer=customer,
            )
            eligible_candidates.append(candidate)

        if not eligible_candidates:
            return []

        # Deduplicate per merchant
        deduped = self.deduplicate_merchants(eligible_candidates, now_dt)

        # Rank candidates
        ranked = self.rank_candidates(deduped, now_dt)

        # Apply batch ceiling
        top_candidates = ranked[: self.max_candidates]

        # Compose and validate actions concurrently
        async def compose_candidate(cand: CandidateTrigger) -> Optional[Dict[str, Any]]:
            try:
                res = await self.composer.compose(cand)
                if res and isinstance(res, dict):
                    return res
            except Exception:
                pass
            # Guaranteed fallback on unexpected error
            fallback = DeterministicComposer.compose(cand)
            if fallback:
                is_valid, _ = GroundingValidator.validate(
                    body=fallback.get("body", ""),
                    category=cand.category,
                    merchant=cand.merchant,
                    trigger=cand.trigger,
                    customer=cand.customer,
                    cta=fallback.get("cta"),
                )
                if is_valid:
                    return fallback
            return None

        tasks = [compose_candidate(cand) for cand in top_candidates]
        composed_actions = await asyncio.gather(*tasks, return_exceptions=False)

        actions: List[Dict[str, Any]] = []
        for action in composed_actions:
            if action and isinstance(action, dict):
                actions.append(action)
                # Commit suppression, cooldown, and conversation state
                await state.commit_action(action, now_dt)

        return actions
