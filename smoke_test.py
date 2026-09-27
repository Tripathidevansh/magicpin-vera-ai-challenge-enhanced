#!/usr/bin/env python3
"""
magicpin Vera AI Challenge — Smoke Test Suite (Phase 2)
=======================================================
Comprehensive verification script for candidate bot endpoints and
decision engine. Uses run-unique IDs so tests are repeatable against
a long-running daemon.

Tests:
1. GET  /v1/healthz   - Baseline readiness & context count
2. GET  /v1/metadata  - Identity, model, approach schema
3. POST /v1/context   - Ingest category context (200 OK)
4. POST /v1/context   - Ingest merchant context (200 OK)
5. POST /v1/context   - Re-ingest same version (409 Conflict check)
6. POST /v1/context   - Ingest higher version (200 OK version bump)
7. POST /v1/context   - Ingest invalid scope (400 Bad Request check)
8. POST /v1/context   - Ingest customer context (200 OK)
9. POST /v1/context   - Ingest trigger context (200 OK)
10. POST /v1/tick     - Evaluate tick & verify grounded composed action
11. POST /v1/tick     - Repeated tick verifies active suppression (actions == [])
12. POST /v1/reply    - Deliver reply turn (200 OK, valid action)
13. GET  /v1/healthz  - Verify all 4 context counters accurately updated
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest, error as urlerror

BOT_URL = os.environ.get("BOT_URL", "http://localhost:8080").rstrip("/")

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def log_pass(msg: str):
    print(f"  {Colors.GREEN}[PASS]{Colors.RESET} {msg}")

def log_fail(msg: str):
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {msg}")

def log_info(msg: str):
    print(f"  {Colors.BLUE}[INFO]{Colors.RESET} {msg}")

def http_req(method: str, path: str, body: dict = None) -> tuple[int, dict]:
    url = f"{BOT_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    req = urlrequest.Request(url, data=data, method=method, headers=headers)

    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urlerror.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw) if raw else {}
        except Exception:
            return e.code, {"error": raw}
    except Exception as e:
        return 0, {"exception": str(e)}

def run_smoke_test() -> bool:
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}magicpin Vera Bot — Smoke Test Suite (Phase 2){Colors.RESET}")
    print(f"Target URL: {BOT_URL}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    all_passed = True
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    epoch_suffix = int(time.time() * 1000)

    # 1. GET /v1/healthz (initial)
    status_code, resp = http_req("GET", "/v1/healthz")
    if status_code == 200 and resp.get("status") == "ok":
        log_pass(f"GET /v1/healthz responded 200 OK (uptime: {resp.get('uptime_seconds')}s)")
    else:
        log_fail(f"GET /v1/healthz failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 2. GET /v1/metadata
    status_code, resp = http_req("GET", "/v1/metadata")
    if status_code == 200 and "team_name" in resp and "model" in resp:
        log_pass(f"GET /v1/metadata responded 200 OK (team: '{resp.get('team_name')}')")
    else:
        log_fail(f"GET /v1/metadata failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 3. POST /v1/context (Category - run unique)
    cat_slug = f"dentists_run_{epoch_suffix}"
    cat_payload = {
        "slug": cat_slug,
        "display_name": "Dentists",
        "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed", "cure"]},
        "offer_catalog": [{"id": "den_001", "title": "Dental Cleaning @ ₹299", "value": "299"}],
        "digest": [{
            "id": "d_001_fluoride",
            "title": "3-month fluoride recall cuts caries 38% better",
            "source": "JIDA Oct 2026, p.14",
            "trial_n": 2100,
            "patient_segment": "high_risk_adults",
        }],
    }
    cat_req = {
        "scope": "category",
        "context_id": cat_slug,
        "version": 1,
        "payload": cat_payload,
        "delivered_at": now_iso,
    }
    status_code, resp = http_req("POST", "/v1/context", cat_req)
    if status_code == 200 and resp.get("accepted") is True:
        log_pass(f"POST /v1/context (category '{cat_slug}') stored successfully (HTTP 200)")
    else:
        log_fail(f"POST /v1/context (category) failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 4. POST /v1/context (Merchant - run unique)
    mer_id = f"m_smoke_{epoch_suffix}"
    mer_payload = {
        "merchant_id": mer_id,
        "category_slug": cat_slug,
        "identity": {"name": "Dr. Meera's Clinic", "owner_first_name": "Meera", "city": "Delhi"},
        "offers": [{"id": "o_001", "title": "Dental Cleaning @ ₹299", "status": "active"}],
        "signals": ["high_risk_adult_cohort"],
    }
    mer_req = {
        "scope": "merchant",
        "context_id": mer_id,
        "version": 1,
        "payload": mer_payload,
        "delivered_at": now_iso,
    }
    status_code, resp = http_req("POST", "/v1/context", mer_req)
    if status_code == 200 and resp.get("accepted") is True:
        log_pass("POST /v1/context (merchant) stored successfully (HTTP 200)")
    else:
        log_fail(f"POST /v1/context (merchant) failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 5. POST /v1/context (Stale / Idempotency check -> 409 Conflict)
    status_code, resp = http_req("POST", "/v1/context", mer_req)
    if status_code == 409 and resp.get("accepted") is False and resp.get("reason") == "stale_version":
        log_pass("POST /v1/context correctly returned HTTP 409 Conflict on equal/stale version")
    else:
        log_fail(f"POST /v1/context failed idempotency test: expected 409, got {status_code}, body: {resp}")
        all_passed = False

    # 6. POST /v1/context (Version bump -> 200 OK)
    mer_req_v2 = dict(mer_req)
    mer_req_v2["version"] = 2
    mer_req_v2["payload"]["identity"]["verified"] = True
    status_code, resp = http_req("POST", "/v1/context", mer_req_v2)
    if status_code == 200 and resp.get("accepted") is True:
        log_pass("POST /v1/context version bump (v2) accepted (HTTP 200)")
    else:
        log_fail(f"POST /v1/context version bump failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 7. POST /v1/context (Invalid scope -> 400 Bad Request)
    invalid_req = dict(mer_req)
    invalid_req["scope"] = "invalid_scope_name"
    status_code, resp = http_req("POST", "/v1/context", invalid_req)
    if status_code == 400 and resp.get("reason") == "invalid_scope":
        log_pass("POST /v1/context correctly returned HTTP 400 on invalid scope")
    else:
        log_fail(f"POST /v1/context invalid scope test failed: expected 400, got {status_code}, body: {resp}")
        all_passed = False

    # 8. POST /v1/context (Customer - run unique)
    cust_id = f"c_smoke_{epoch_suffix}"
    cust_payload = {
        "customer_id": cust_id,
        "merchant_id": mer_id,
        "identity": {"name": "Priya", "language_pref": "hi-en mix"},
    }
    cust_req = {
        "scope": "customer",
        "context_id": cust_id,
        "version": 1,
        "payload": cust_payload,
        "delivered_at": now_iso,
    }
    status_code, resp = http_req("POST", "/v1/context", cust_req)
    if status_code == 200 and resp.get("accepted") is True:
        log_pass("POST /v1/context (customer) stored successfully (HTTP 200)")
    else:
        log_fail(f"POST /v1/context (customer) failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 9. POST /v1/context (Trigger - run unique)
    trg_id = f"trg_smoke_{epoch_suffix}"
    trg_payload = {
        "id": trg_id,
        "scope": "customer",
        "kind": "recall_due",
        "merchant_id": mer_id,
        "customer_id": cust_id,
        "urgency": 4,
        "suppression_key": f"supp:smoke_{epoch_suffix}",
        "expires_at": "2030-12-31T00:00:00Z",
        "payload": {
            "service_due": "Dental Cleaning",
            "available_slots": [
                {"label": "Wed 5 Nov, 6pm"},
                {"label": "Thu 6 Nov, 5pm"},
            ],
        },
    }
    trg_req = {
        "scope": "trigger",
        "context_id": trg_id,
        "version": 1,
        "payload": trg_payload,
        "delivered_at": now_iso,
    }
    status_code, resp = http_req("POST", "/v1/context", trg_req)
    if status_code == 200 and resp.get("accepted") is True:
        log_pass("POST /v1/context (trigger) stored successfully (HTTP 200)")
    else:
        log_fail(f"POST /v1/context (trigger) failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 10. POST /v1/tick (Evaluate tick & verify action composition)
    tick_req = {
        "now": now_iso,
        "available_triggers": [trg_id],
    }
    status_code, resp = http_req("POST", "/v1/tick", tick_req)
    actions = resp.get("actions", [])
    if status_code == 200 and len(actions) == 1:
        action = actions[0]
        if (
            action.get("send_as") == "merchant_on_behalf"
            and action.get("customer_id") == cust_id
            and "Dental Cleaning @ ₹299" in action.get("body", "")
            and "http" not in action.get("body", "").lower()
        ):
            log_pass(f"POST /v1/tick composed grounded action with send_as='{action.get('send_as')}'")
        else:
            log_fail(f"POST /v1/tick action payload invalid: {action}")
            all_passed = False
    else:
        log_fail(f"POST /v1/tick failed: expected 1 action, got {len(actions)}, body: {resp}")
        all_passed = False

    # 11. POST /v1/tick (Repeated tick verifies suppression)
    status_code, resp = http_req("POST", "/v1/tick", tick_req)
    actions_repeat = resp.get("actions", [])
    if status_code == 200 and len(actions_repeat) == 0:
        log_pass("POST /v1/tick repeated call correctly suppressed previously sent action (actions: [])")
    else:
        log_fail(f"POST /v1/tick suppression failed on repeat: expected 0 actions, got {len(actions_repeat)}")
        all_passed = False

    # 12. POST /v1/reply
    reply_req = {
        "conversation_id": f"conv_smoke_{epoch_suffix}",
        "merchant_id": mer_id,
        "customer_id": None,
        "from_role": "merchant",
        "message": "Yes please send the abstract",
        "received_at": now_iso,
        "turn_number": 2,
    }
    status_code, resp = http_req("POST", "/v1/reply", reply_req)
    if status_code == 200 and "action" in resp and resp.get("action") in ["send", "wait", "end"]:
        log_pass(f"POST /v1/reply responded 200 OK with action='{resp.get('action')}'")
    else:
        log_fail(f"POST /v1/reply failed: HTTP {status_code}, body: {resp}")
        all_passed = False

    # 13. GET /v1/healthz (verify context counts)
    status_code, resp = http_req("GET", "/v1/healthz")
    counts = resp.get("contexts_loaded", {})
    if status_code == 200 and all(counts.get(k, 0) >= 1 for k in ["category", "merchant", "customer", "trigger"]):
        log_pass(f"GET /v1/healthz verified all 4 context scopes loaded: {counts}")
    else:
        log_fail(f"GET /v1/healthz counter check failed: {counts}")
        all_passed = False

    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}ALL 13 SMOKE TESTS PASSED SUCCESSFULLY!{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}ONE OR MORE SMOKE TESTS FAILED.{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    return all_passed

if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
