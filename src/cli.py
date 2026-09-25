"""Command-line interface argument parsing and validation."""

from __future__ import annotations

import argparse
import sys

VALID_MODES = ("paper", "live", "scan", "status", "backtest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the scalping bot."""
    parser = argparse.ArgumentParser(
        prog="crypto-scalper",
        description="Crypto scalping bot — Delta Exchange market data + execution",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_MODES,
        default="paper",
        help="Operating mode (default: paper)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=("scalp",),
        default="scalp",
        help="Strategy profile to execute (strictly: scalp)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Path to config directory (default: config/)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=None,
        help="Override log level from .env",
    )
    parser.add_argument(
        "--kill-switch",
        action="store_true",
        default=False,
        help="Activate emergency kill switch on start",
    )
    parser.add_argument(
        "--paper-balance",
        type=float,
        default=10_000.0,
        help="Starting balance for paper trading (default: 10000)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Universe preset to scan/trade (e.g. 'majors', 'top10', 'gold')",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated custom symbols to scan/trade (e.g. 'BTCUSD,SOLUSD,XAUTUSD')",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Skip interactive confirmation prompt for live trading",
    )
    args = parser.parse_args(argv)

    if args.mode == "live":
        import os
        if os.getenv("LIVE_TRADING", "false").strip().lower() != "true":
            print(
                "\n❌  LIVE mode requires LIVE_TRADING=true in .env\n"
                "    Set LIVE_TRADING=true and restart.\n"
            )
            sys.exit(1)
        if not getattr(args, "yes", False):
            if sys.stdin and sys.stdin.isatty():
                confirm = input(
                    "\n⚠️  You are about to start LIVE trading with REAL money.\n"
                    "    Type 'CONFIRM' to proceed: "
                )
                if confirm.strip() != "CONFIRM":
                    print("Aborted.")
                    sys.exit(0)
            else:
                print("⚠️  Running in non-interactive environment for live trading; pass --yes / -y to proceed.")
                sys.exit(1)

    return args
