#!/usr/bin/env python3
"""
Deployment smoke checks for Railway/Vercel runtime compatibility.

Required checks:
- /healthz responds 200 and status=ok
- /readyz responds 200 and no required failures
- /api/portfolio responds with key portfolio fields
- /api/strategies responds with a non-empty list
- /api/trades/recent responds with crypto-style symbols
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    status_code: Optional[int] = None


def _request_json(url: str, timeout: float) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(
        url=url,
        headers={"Accept": "application/json", "User-Agent": "deployment-smoke-check"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status_code = resp.getcode()
        payload = json.loads(resp.read().decode("utf-8"))
    return status_code, payload


def _run_checks(base_url: str, timeout: float) -> List[CheckResult]:
    checks: List[CheckResult] = []

    # 1) Liveness
    try:
        status_code, payload = _request_json(f"{base_url}/healthz", timeout=timeout)
        ok = status_code == 200 and payload.get("status") == "ok"
        checks.append(CheckResult("healthz", ok, f"status={payload.get('status')}", status_code))
    except Exception as exc:
        checks.append(CheckResult("healthz", False, f"request failed ({exc})"))

    # 2) Readiness
    try:
        status_code, payload = _request_json(f"{base_url}/readyz", timeout=timeout)
        required_failures = payload.get("required_failures", [])
        ok = status_code == 200 and not required_failures
        checks.append(
            CheckResult(
                "readyz",
                ok,
                f"required_failures={required_failures}",
                status_code,
            )
        )
    except Exception as exc:
        checks.append(CheckResult("readyz", False, f"request failed ({exc})"))

    # 3) Portfolio endpoint contract
    try:
        status_code, payload = _request_json(f"{base_url}/api/portfolio", timeout=timeout)
        required_fields = {"total_value", "cash", "buying_power"}
        missing = sorted(field for field in required_fields if field not in payload)
        ok = status_code == 200 and not missing
        checks.append(
            CheckResult(
                "api_portfolio",
                ok,
                "missing_fields=" + ",".join(missing) if missing else "contract ok",
                status_code,
            )
        )
    except Exception as exc:
        checks.append(CheckResult("api_portfolio", False, f"request failed ({exc})"))

    # 4) Strategies endpoint contract
    try:
        status_code, payload = _request_json(f"{base_url}/api/strategies", timeout=timeout)
        strategies = payload.get("strategies", [])
        ok = status_code == 200 and isinstance(strategies, list) and len(strategies) > 0
        checks.append(
            CheckResult(
                "api_strategies",
                ok,
                f"strategies_count={len(strategies) if isinstance(strategies, list) else 'n/a'}",
                status_code,
            )
        )
    except Exception as exc:
        checks.append(CheckResult("api_strategies", False, f"request failed ({exc})"))

    # 5) Trades endpoint should return crypto pair style symbols
    try:
        status_code, payload = _request_json(f"{base_url}/api/trades/recent?limit=5", timeout=timeout)
        trades = payload.get("trades", [])
        symbols = [str(t.get("symbol", "")) for t in trades if isinstance(t, dict)]
        non_crypto = [s for s in symbols if "/" not in s]
        ok = status_code == 200 and isinstance(trades, list) and not non_crypto
        checks.append(
            CheckResult(
                "api_trades_recent",
                ok,
                f"symbols={symbols}" if symbols else "no symbols returned",
                status_code,
            )
        )
    except Exception as exc:
        checks.append(CheckResult("api_trades_recent", False, f"request failed ({exc})"))

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deployment smoke checks.")
    parser.add_argument(
        "--base-url",
        default="",
        help="Backend base URL (can also use DEPLOY_SMOKE_BASE_URL).",
    )
    parser.add_argument(
        "--timeout",
        default=8.0,
        type=float,
        help="Request timeout seconds.",
    )
    args = parser.parse_args()

    base_url = args.base_url.strip()
    if not base_url:
        base_url = os.environ.get("DEPLOY_SMOKE_BASE_URL", "").strip()
    if not base_url:
        print("ERROR: missing --base-url (or DEPLOY_SMOKE_BASE_URL)")
        return 2

    base_url = base_url.rstrip("/")
    print(f"Running deployment smoke checks against: {base_url}")

    checks = _run_checks(base_url=base_url, timeout=args.timeout)
    failures = [check for check in checks if not check.ok]

    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        status_code = f" ({check.status_code})" if check.status_code is not None else ""
        print(f"[{status}] {check.name}{status_code} - {check.detail}")

    if failures:
        print(f"\nSmoke check failed: {len(failures)} failing check(s)")
        return 1

    print("\nSmoke check passed: all required checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
