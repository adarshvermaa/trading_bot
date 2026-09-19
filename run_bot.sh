#!/usr/bin/env bash
# ==============================================================================
# Crypto Scalping Bot — CLI Runner & Management Script
# Binance Futures Data + Delta Exchange India Futures Execution
# ==============================================================================

set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for terminal output
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
NC="\033[0m" # No Color

# Determine Python binary
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    echo -e "${RED}Error: Neither .venv/bin/python nor python3 was found.${NC}"
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

show_help() {
    echo -e "${BOLD}${CYAN}======================================================================${NC}"
    echo -e "${BOLD}${GREEN}        Crypto Scalping Bot — CLI Runner & Management                ${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "Usage: ${BOLD}./run_bot.sh <command> [options]${NC}\n"
    echo -e "${BOLD}Commands:${NC}"
    echo -e "  ${GREEN}status${NC}        Check configuration, credentials, and account balance"
    echo -e "  ${GREEN}scan${NC}          Run multi-timeframe market structure scanner once"
    echo -e "  ${GREEN}paper${NC}         Launch interactive terminal dashboard in PAPER mode"
    echo -e "  ${GREEN}live${NC}          Run bot in LIVE mode (requires LIVE_TRADING=true in .env)"
    echo -e "  ${GREEN}verify${NC}        Verify Delta API futures endpoints, wallet & rounded SL/TP"
    echo -e "                Options: ${CYAN}--dry-run${NC}, ${CYAN}--test-order${NC}, ${CYAN}--api-key <k>${NC}, ${CYAN}--api-secret <s>${NC}"
    echo -e "  ${GREEN}test${NC}          Run automated unit & integration test suite (pytest)"
    echo -e "  ${GREEN}kill${NC}          Emergency stop / activate kill-switch"
    echo -e "  ${GREEN}help${NC}          Show this help message"
    echo -e "\n${BOLD}Examples:${NC}"
    echo -e "  ./run_bot.sh status"
    echo -e "  ./run_bot.sh verify --dry-run"
    echo -e "  ./run_bot.sh verify --test-order"
    echo -e "  ./run_bot.sh paper"
    echo -e "${CYAN}======================================================================${NC}"
}

COMMAND="$1"
shift || true

case "$COMMAND" in
    status)
        echo -e "${BLUE}▶ Checking bot configuration and account status...${NC}"
        "$PYTHON" -m src --mode status "$@"
        ;;
    scan)
        echo -e "${BLUE}▶ Running market structure scanner across top-10 futures assets...${NC}"
        "$PYTHON" -m src --mode scan "$@"
        ;;
    paper)
        echo -e "${GREEN}▶ Launching scalping bot in PAPER mode (simulation)...${NC}"
        "$PYTHON" -m src --mode paper "$@"
        ;;
    live)
        echo -e "${YELLOW}▶ Launching scalping bot in LIVE mode...${NC}"
        "$PYTHON" -m src --mode live "$@"
        ;;
    verify)
        echo -e "${BLUE}▶ Running Delta API futures & SL/TP precision verification...${NC}"
        "$PYTHON" scripts/verify_delta_execution.py "$@"
        ;;
    test)
        echo -e "${BLUE}▶ Running automated test suite with coverage...${NC}"
        if [ -f ".venv/bin/pytest" ]; then
            .venv/bin/pytest -v tests/ "$@"
        else
            "$PYTHON" -m pytest -v tests/ "$@"
        fi
        ;;
    kill)
        echo -e "${RED}▶ Triggering emergency kill-switch...${NC}"
        "$PYTHON" -m src --kill-switch "$@"
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}\n"
        show_help
        exit 1
        ;;
esac
