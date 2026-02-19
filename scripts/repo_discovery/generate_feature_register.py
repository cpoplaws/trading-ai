#!/usr/bin/env python3
"""
Generate docs/feature-evidence-register.csv from repository-wide evidence scans.
"""

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "feature-evidence-register.csv"

INCLUDE_SUFFIXES = {
    ".md",
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".env",
    ".txt",
    ".cfg",
    ".ini",
    ".sh",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    ".pytest_cache",
    "htmlcov",
    "__pycache__",
    ".mypy_cache",
}

FEATURE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("trading_strategies", re.compile(r"\b(strategy|mean[_ -]?reversion|momentum|rsi|ml[_ -]?ensemble|ppo[_ -]?rl|grid|dca|arbitrage|yield optimizer)\b", re.IGNORECASE)),
    ("agent_swarm", re.compile(r"\b(agent|swarm|supervisor)\b", re.IGNORECASE)),
    ("market_intelligence", re.compile(r"\b(intelligence|signal|regime|sentiment)\b", re.IGNORECASE)),
    ("risk_management", re.compile(r"\b(risk|drawdown|var|cvar|position sizing|circuit breaker)\b", re.IGNORECASE)),
    ("portfolio_analytics", re.compile(r"\b(portfolio|allocation|positions|trades|performance)\b", re.IGNORECASE)),
    ("api_runtime", re.compile(r"\b(fastapi|apirouter|endpoint|router|uvicorn|/api/)\b", re.IGNORECASE)),
    ("websocket_realtime", re.compile(r"\b(websocket|socket\.io|ws://|wss://|real[- ]?time)\b", re.IGNORECASE)),
    ("deployment_railway", re.compile(r"\b(railway|procfile|startcommand|nixpacks)\b", re.IGNORECASE)),
    ("deployment_vercel", re.compile(r"\b(vercel|next_public_api_url|next_public_ws_url)\b", re.IGNORECASE)),
    ("monitoring_alerting", re.compile(r"\b(healthz|readyz|health check|alert|monitoring|logging|runbook|incident)\b", re.IGNORECASE)),
    ("data_dependencies", re.compile(r"\b(database|postgres|timescale|redis|cache)\b", re.IGNORECASE)),
    ("cex_integrations", re.compile(r"\b(alpaca|binance|coinbase|cex)\b", re.IGNORECASE)),
    ("dex_onchain", re.compile(r"\b(dex|defi|uniswap|jupiter|onchain|wallet|gas)\b", re.IGNORECASE)),
    ("backtesting_paper", re.compile(r"\b(backtest|paper trading|simulation)\b", re.IGNORECASE)),
    ("security_auth", re.compile(r"\b(secret|api[_ -]?key|password|auth|cors)\b", re.IGNORECASE)),
    ("ml_rl_models", re.compile(r"\b(ml|model|rl|lstm|transformer|neural|ensemble)\b", re.IGNORECASE)),
]

STOCK_TERMS = re.compile(r"\b(stock|equity|alpaca|aapl|tsla|nasdaq|nyse)\b", re.IGNORECASE)
CRYPTO_TERMS = re.compile(r"\b(crypto|btc|eth|solana|base|arbitrum|optimism|bsc|defi|dex|cex|wallet|rpc|binance)\b", re.IGNORECASE)


def iter_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue

        if path.name in {"Procfile", "Dockerfile", "vercel.json"}:
            yield path
            continue

        if path.name.startswith(".env"):
            yield path
            continue

        if path.suffix.lower() in INCLUDE_SUFFIXES:
            yield path


def source_type_for(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("docs/") or rel.endswith(".md"):
        return "doc"
    if rel.startswith("tests/"):
        return "test"
    if rel.startswith("examples/"):
        return "example"
    if rel.startswith("config/") or rel.startswith(".github/"):
        return "config"
    if any(rel.endswith(ext) for ext in [".yml", ".yaml", ".json", ".toml", ".ini"]):
        return "config"
    if path.name.startswith(".env") or path.name in {"Procfile", "Dockerfile", "vercel.json"}:
        return "config"
    return "code"


def runtime_surface_for(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith("apps/api/") or rel.startswith("src/api/") or rel.startswith("api/"):
        return "api"
    if rel.startswith("apps/dashboard/") or rel.startswith("src/dashboard/") or rel.startswith("src/web-dashboard/"):
        return "dashboard"
    if rel.startswith("infrastructure/") or rel.startswith("docker/") or rel.startswith("k8s/") or rel.startswith(".github/"):
        return "infra"
    if rel.startswith("tests/"):
        return "worker"
    return "worker"


def market_scope_for(excerpt: str) -> str:
    has_stock = bool(STOCK_TERMS.search(excerpt))
    has_crypto = bool(CRYPTO_TERMS.search(excerpt))
    if has_stock and has_crypto:
        return "cross-market"
    if has_stock and not has_crypto:
        return "legacy-non-crypto"
    return "crypto"


def implemented_state_for(source_type: str, excerpt: str) -> str:
    low = excerpt.lower()
    if any(term in low for term in ["todo", "planned", "roadmap", "coming soon", "next phase", "future"]):
        return "planned"
    if any(term in low for term in ["mock", "placeholder", "demo mode", "stub", "fallback"]):
        return "partial"
    if source_type in {"code", "config", "test"}:
        return "implemented"
    if any(term in low for term in ["complete", "ready", "implemented"]):
        return "implemented"
    return "unclear"


def dependency_risk_for(excerpt: str) -> str:
    low = excerpt.lower()
    if any(term in low for term in ["api_key", "secret", "password", "database", "redis", "rpc", "wallet"]):
        return "high"
    if any(term in low for term in ["websocket", "railway", "vercel", "uvicorn", "docker"]):
        return "medium"
    return "low"


def deployment_risk_for(path: Path, excerpt: str) -> str:
    rel = path.relative_to(ROOT).as_posix().lower()
    low = excerpt.lower()
    if any(term in rel for term in ["railway", "vercel", ".github/workflows", "docker", "k8s", "procfile"]):
        return "high"
    if any(term in low for term in ["cors", "port", "startcommand", "healthz", "readyz", "next_public_api_url"]):
        return "high"
    if any(term in low for term in ["api", "websocket", "strategy"]):
        return "medium"
    return "low"


def owner_for(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(("apps/api/", "src/api/", "api/")):
        return "backend-team"
    if rel.startswith(("apps/dashboard/", "src/web-dashboard/", "src/dashboard/")):
        return "frontend-team"
    if rel.startswith(("infrastructure/", "docker/", "k8s/", ".github/")):
        return "platform-team"
    if rel.startswith(("src/strategy/", "src/crypto_strategies/", "src/advanced_strategies/")):
        return "quant-team"
    return "unassigned"


def verification_method_for(surface: str, source_type: str) -> str:
    if surface == "api":
        return "exercise endpoint via smoke/integration tests"
    if surface == "dashboard":
        return "validate API call/render path in dashboard runtime"
    if source_type == "config":
        return "verify in deploy pipeline and env sync review"
    if source_type == "test":
        return "execute targeted pytest or integration scenario"
    return "trace to runtime consumer and confirm behavior"


def main() -> None:
    rows: List[Dict[str, str]] = []
    seen = set()

    for file_path in iter_files():
        rel = file_path.relative_to(ROOT).as_posix()
        source_type = source_type_for(file_path)
        runtime_surface = runtime_surface_for(file_path)

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue

        for line_number, raw_line in enumerate(content, start=1):
            line = raw_line.strip()
            if not line or len(line) < 4:
                continue

            matches = [feature for feature, pattern in FEATURE_PATTERNS if pattern.search(line)]
            if not matches:
                continue

            excerpt = re.sub(r"\s+", " ", line)[:220]
            market_scope = market_scope_for(excerpt)
            implemented_state = implemented_state_for(source_type, excerpt)
            dependency_risk = dependency_risk_for(excerpt)
            deployment_risk = deployment_risk_for(file_path, excerpt)
            owner = owner_for(file_path)
            verification_method = verification_method_for(runtime_surface, source_type)

            for feature in matches[:3]:
                key = (feature, rel, line_number)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "feature_name": feature,
                        "market_scope": market_scope,
                        "source_type": source_type,
                        "source_path": rel,
                        "source_line": str(line_number),
                        "mention_excerpt": excerpt,
                        "implemented_state": implemented_state,
                        "runtime_surface": runtime_surface,
                        "dependency_risk": dependency_risk,
                        "deployment_risk": deployment_risk,
                        "owner": owner,
                        "verification_method": verification_method,
                    }
                )

    rows.sort(key=lambda r: (r["feature_name"], r["source_path"], int(r["source_line"])))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "feature_id",
                "feature_name",
                "market_scope",
                "source_type",
                "source_path",
                "source_line",
                "mention_excerpt",
                "implemented_state",
                "runtime_surface",
                "dependency_risk",
                "deployment_risk",
                "owner",
                "verification_method",
            ]
        )
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    f"FEAT-{idx:05d}",
                    row["feature_name"],
                    row["market_scope"],
                    row["source_type"],
                    row["source_path"],
                    row["source_line"],
                    row["mention_excerpt"],
                    row["implemented_state"],
                    row["runtime_surface"],
                    row["dependency_risk"],
                    row["deployment_risk"],
                    row["owner"],
                    row["verification_method"],
                ]
            )

    print(f"Generated {len(rows)} feature evidence rows at {OUTPUT}")


if __name__ == "__main__":
    main()
