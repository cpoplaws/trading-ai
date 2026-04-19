"""CLI entrypoints for Quantlytics."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def _find_project_root() -> Path:
    env_root = os.getenv("QUANTLYTICS_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "examples").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def _run_python_script(script_path: Path) -> int:
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    prefix = f"{PROJECT_ROOT}{os.pathsep}{PROJECT_ROOT / 'src'}"
    env["PYTHONPATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    return subprocess.call([sys.executable, str(script_path)], cwd=PROJECT_ROOT, env=env)


def start_dashboard() -> int:
    dashboard_path = PROJECT_ROOT / "src" / "dashboard" / "unified_dashboard.py"
    if not dashboard_path.exists():
        print("❌ Dashboard module not found.")
        return 1
    if importlib.util.find_spec("streamlit") is None:
        print("❌ streamlit is not installed. Install dependencies first:")
        print("   pip install -r requirements-secure.txt")
        return 1
    return subprocess.call(["streamlit", "run", str(dashboard_path)], cwd=PROJECT_ROOT)


def start_paper_mode() -> int:
    print("ℹ️  Starting paper-trading demo mode (simulated orders, no real funds).")
    return _run_python_script(PROJECT_ROOT / "examples" / "strategies" / "demo_crypto_paper_trading.py")


def start_live_mode() -> int:
    required = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        print("❌ Live mode requires configured broker credentials.")
        print(f"   Missing environment variables: {', '.join(missing)}")
        return 2

    print("⚠️  Live mode can place real orders. Confirm your risk controls first.")
    return _run_python_script(PROJECT_ROOT / "examples" / "strategies" / "demo_live_trading.py")


def run_backtests() -> int:
    backtest_script = PROJECT_ROOT / "scripts" / "run_all_backtests.py"
    if backtest_script.exists():
        return _run_python_script(backtest_script)

    return _run_python_script(PROJECT_ROOT / "examples" / "strategies" / "simple_backtest_demo.py")


def show_status() -> int:
    packages = ["numpy", "pandas", "fastapi", "streamlit", "requests"]
    print("Quantlytics Platform Status")
    print("Internal engine: Itera")
    print("-" * 48)
    for pkg in packages:
        installed = importlib.util.find_spec(pkg) is not None
        print(f"{'✅' if installed else '❌'} {pkg}")
    return 0


def run_api() -> None:
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantlytics launcher (Itera engine)")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "paper", "live", "backtest", "status"],
        default="dashboard",
        help="Launch mode",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    actions = {
        "dashboard": start_dashboard,
        "paper": start_paper_mode,
        "live": start_live_mode,
        "backtest": run_backtests,
        "status": show_status,
    }
    exit_code = actions[args.mode]()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
