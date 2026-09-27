# magicpin Vera AI Assistant — Candidate Implementation

Candidate bot submission for the **magicpin Vera AI Challenge**. This repository implements an autonomous WhatsApp merchant engagement assistant designed to engage merchants and customers across local-commerce verticals.

---

## Current Status: Phase 2 Complete (Deterministic Decision Engine)

- [x] **FastAPI HTTP Service**: 5 official REST endpoints implemented with strict schema validation.
- [x] **Thread-Safe In-Memory State**: Atomic context storage, version tracking, idempotency enforcement (HTTP 409 on stale/equal versions).
- [x] **Trigger Eligibility Pipeline**: Rejects expired triggers (`now >= expires_at`), active suppression keys, opted-out merchants, and missing contexts.
- [x] **Merchant Deduplication**: Ensures strictly at most ONE proactive action per merchant survives per tick.
- [x] **Candidate Ranking & Batching**: Configurable `MAX_CANDIDATES_PER_TICK` (default 5), ranking by urgency, relevance, signal match, and deadline proximity.
- [x] **Deterministic Baseline Composer**: Grounded composition for canonical trigger kinds (`research_digest`, `recall_due`, `bridal_followup`, `curious_ask_due`, `ipl_match_today`, `active_planning_intent`, `seasonal_perf_dip`, `customer_lapsed_hard`, `supply_alert`, `chronic_refill_due`, `renewal_due`).
- [x] **Grounding & Safety Validator**: Strict rejection of URLs, internal jargon (`trg_`, `conv_`, `payload`), category taboo words, and unverified `₹` prices.
- [x] **Suppression & Cooldown Persistence**: Atomic commit of suppression keys, merchant 24h cooldown, and conversation history upon action generation.
- [ ] *Phase 3 (Upcoming)*: Dual-provider LLM pipeline (Gemini 2.5 Flash + Groq Llama-3.3-70b), multi-turn `/v1/reply` state machine (auto-reply detection, action mode intent transitions, hostile handling).

---

## Architecture Overview

```
                      Judge Harness / HTTP Client
                                  │
      ┌──────────────┬────────────┼─────────────┬─────────────┐
      ▼              ▼            ▼             ▼             ▼
GET /healthz   GET /metadata POST /context  POST /tick   POST /reply
      │              │            │             │             │
      │              │            ▼             ▼             │
      │              │      Thread-Safe    Decision Engine    │
      │              │      Context Store  ├─ Eligibility     │
      │              │     (Atomic Mutex)  ├─ Dedup per Mx    │
      │              │            │        ├─ Ranking         │
      │              │            │        ├─ Grounded Comp   │
      │              │            │        └─ Safety Checks   │
      └──────────────┴────────────┴───────────────────────────┘
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- A virtual environment (recommended)

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
- `MAX_CANDIDATES_PER_TICK`: Maximum number of proactive actions emitted per tick (default: `5`).
- `BOT_URL`: Target URL for testing (default: `http://localhost:8080`).

---

## Running the Bot

Start the FastAPI application on port `8080`:

```bash
# Using uvicorn directly
uvicorn bot:app --host 0.0.0.0 --port 8080

# Or run bot.py directly
python3 bot.py
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Primary LLM: Google Gemini 2.5 Flash API Key |
| `GROQ_API_KEY` | *(empty)* | Fallback LLM: Groq Llama 3.3 70B API Key |
| `LLM_MODE` | `hybrid` | Routing mode: `deterministic`, `hybrid`, `llm` |
| `ENABLE_LLM_COMPOSER` | `true` | Master flag to toggle LLM composition layer |
| `LLM_CALL_TIMEOUT` | `4.5` | Strict per-call timeout in seconds (well below 30s SLA) |
| `MAX_CANDIDATES_PER_TICK` | `5` | Maximum candidates composed per tick |

---

## Testing & Verification

### 1. Run Decision Engine Unit Tests (Pipeline A–M)
```bash
python3 -m unittest -v test_decision_engine.py
```
Validates:
- Expired triggers are suppressed
- Active suppression keys prevent duplicate sends
- Opted-out merchants receive no outreach
- Missing contexts are safely skipped
- Strict 1-action-per-merchant deduplication
- Urgency and relevance priority ranking
- Restraint and empty actions on low-value/unsupported triggers
- Customer triggers use `send_as = merchant_on_behalf`
- Fabricated prices and URLs are strictly rejected
- Atomic suppression and merchant cooldown persistence
- Adaptive/unseen trigger payloads do not crash the bot

### 2. Run LLM Composition Layer Test Suite (Cases A–K)
```bash
python3 -m unittest -v test_llm_layer.py
```
Validates without requiring live API keys:
- **A**: Valid structured JSON output accepted
- **B**: Malformed JSON triggers seamless fallback to `DeterministicComposer`
- **C**: Provider timeout triggers seamless fallback to `DeterministicComposer`
- **D**: Gemini failure automatically fails over to Groq fallback
- **E**: Both providers failing defaults to verified deterministic baseline
- **F**: Fabricated ₹ prices rejected by `GroundingValidator`
- **G**: Forbidden URLs rejected by `GroundingValidator`
- **H**: Taboo medical claims rejected by `GroundingValidator`
- **I**: Leaked internal system jargon rejected by `GroundingValidator`
- **J**: Multiple competing CTAs rejected or normalized
- **K**: Valid grounded response with verified context accepted

### 3. Run Canonical Case Study Evaluation (All 10 Cases)
```bash
python3 test_canonical_cases.py
```
Evaluates all 10 canonical scenarios across:
1. Dentist research digest (`trg_001`)
2. Dentist recall (`trg_003`)
3. Salon bridal follow-up (`trg_007`)
4. Salon curiosity (`trg_008`)
5. IPL restaurant recommendation (`trg_010`)
6. Corporate thali planning (`trg_013`)
7. Gym seasonal dip (`trg_014`)
8. Gym winback (`trg_015`)
9. Pharmacy recall (`trg_018`)
10. Pharmacy refill (`trg_019`)

Outputs latency, validation results, and observable properties to `canonical_evaluation_results.json`.

### 4. Run Smoke Test Suite
With the bot running on port 8080:
```bash
python3 smoke_test.py
```
Validates all 5 HTTP endpoints against the live server.

