#!/usr/bin/env bash
# ==============================================================================
# ALL-IN-ONE CRYPTO FUTURES SCALPING BOT MASTER SCRIPT
# Delta Exchange Market Data + Futures Execution
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ANSI Colors
BOLD="\033[1m"
DIM="\033[2m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
CYAN="\033[0;36m"
MAGENTA="\033[0;35m"
NC="\033[0m"

# ------------------------------------------------------------------------------
# 1. Environment & Python Resolution
# ------------------------------------------------------------------------------
resolve_python() {
    if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
        PYTHON="$SCRIPT_DIR/.venv/bin/python"
    elif command -v python3 &> /dev/null; then
        PYTHON="python3"
    else
        echo -e "${RED}❌ Fatal: Python 3 was not found on your system.${NC}"
        exit 1
    fi
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
}

resolve_python

# ------------------------------------------------------------------------------
# 2. Banner
# ------------------------------------------------------------------------------
print_banner() {
    clear 2>/dev/null || true
    echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}${GREEN}          ALL-IN-ONE CRYPTO FUTURES SCALPING BOT MASTER CLI                ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}║${DIM}   Delta Exchange India (Market Data + Futures Execution)                   ${NC}${CYAN}║${NC}"
    echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ------------------------------------------------------------------------------
# 3. Environment & Credential Check
# ------------------------------------------------------------------------------
check_env_file() {
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        echo -e "${YELLOW}⚠️  .env file not found. Initializing from .env.example...${NC}"
        if [ -f "$SCRIPT_DIR/.env.example" ]; then
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            echo -e "${GREEN}✅ Created .env from template.${NC}"
        else
            cat << 'EOF' > "$SCRIPT_DIR/.env"
DELTA_API_KEY=
DELTA_API_SECRET=
DELTA_API_URL=https://api.india.delta.exchange
DELTA_WS_URL=wss://socket.india.delta.exchange
LITELLM_API_KEY=
LITELLM_MODEL=gpt-4o-mini
LIVE_TRADING=false
LOG_LEVEL=INFO
LOG_DIR=logs
EOF
            echo -e "${GREEN}✅ Created new .env file.${NC}"
        fi
    fi
}

get_env_val() {
    local key="$1"
    grep "^${key}=" "$SCRIPT_DIR/.env" 2>/dev/null | cut -d '=' -f2- | tr -d '\r"' || true
}

mask_string() {
    local str="$1"
    local len="${#str}"
    if [ "$len" -le 8 ]; then
        echo "******** ($len chars)"
    else
        local start="${str:0:4}"
        local end="${str: -4}"
        echo "${start}...${end} ($len chars)"
    fi
}

display_config_summary() {
    check_env_file
    local api_key
    local api_secret
    local live_trading
    local api_url
    api_key="$(get_env_val "DELTA_API_KEY")"
    api_secret="$(get_env_val "DELTA_API_SECRET")"
    live_trading="$(get_env_val "LIVE_TRADING")"
    api_url="$(get_env_val "DELTA_API_URL")"

    echo -e "${BOLD}Current Configuration:${NC}"
    echo -e "  • Delta API URL   : ${CYAN}${api_url:-https://api.india.delta.exchange}${NC}"
    echo -e "  • Live Trading    : $([ "$live_trading" = "true" ] && echo -e "${GREEN}${BOLD}ENABLED (LIVE)${NC}" || echo -e "${YELLOW}DISABLED (PAPER)${NC}")"
    echo -e "  • Delta API Key   : $([ -n "$api_key" ] && echo -e "${GREEN}$(mask_string "$api_key")${NC}" || echo -e "${RED}NOT CONFIGURED${NC}")"
    echo -e "  • Delta API Secret: $([ -n "$api_secret" ] && echo -e "${GREEN}$(mask_string "$api_secret")${NC}" || echo -e "${RED}NOT CONFIGURED${NC}")"
    echo ""
}

configure_credentials() {
    print_banner
    echo -e "${BOLD}${CYAN}Configure Delta Exchange API Credentials${NC}\n"
    echo -e "Enter your credentials below (stored securely only in .env):"
    
    read -rp "Enter DELTA_API_KEY: " user_key
    read -s -rp "Enter DELTA_API_SECRET: " user_secret
    echo ""
    read -rp "Enable LIVE_TRADING? (true/false) [default: false]: " user_live
    user_live="${user_live:-false}"

    if [ -n "$user_key" ]; then
        sed -i "s|^DELTA_API_KEY=.*|DELTA_API_KEY=$user_key|" "$SCRIPT_DIR/.env"
    fi
    if [ -n "$user_secret" ]; then
        sed -i "s|^DELTA_API_SECRET=.*|DELTA_API_SECRET=$user_secret|" "$SCRIPT_DIR/.env"
    fi
    sed -i "s|^LIVE_TRADING=.*|LIVE_TRADING=$user_live|" "$SCRIPT_DIR/.env"

    echo -e "\n${GREEN}✅ Credentials updated in .env successfully!${NC}"
    sleep 1
}

# ------------------------------------------------------------------------------
# 4. Actions
# ------------------------------------------------------------------------------
action_status() {
    print_banner
    display_config_summary
    echo -e "${BLUE}▶ Fetching real-time account status and balance...${NC}"
    "$PYTHON" -m src --mode status "$@"
    echo ""
}

action_scan() {
    print_banner
    echo -e "${BLUE}▶ Bootstrapping 3,000 candles and running one-shot scan across top-10 universe...${NC}"
    "$PYTHON" -m src --mode scan "$@"
    echo ""
}

action_paper() {
    print_banner
    echo -e "${GREEN}▶ Launching scalping bot in PAPER TRADING simulation mode ($10,000 virtual balance)...${NC}"
    echo -e "${DIM}Press Ctrl+C to stop the bot at any time.${NC}\n"
    sleep 1
    "$PYTHON" -m src --mode paper "$@"
}

action_live() {
    print_banner
    check_env_file
    local api_key
    local api_secret
    local live_trading
    api_key="$(get_env_val "DELTA_API_KEY")"
    api_secret="$(get_env_val "DELTA_API_SECRET")"
    live_trading="$(get_env_val "LIVE_TRADING")"

    if [ -z "$api_key" ] || [ -z "$api_secret" ]; then
        echo -e "${RED}❌ Error: Delta Exchange API key or secret is missing in .env!${NC}"
        echo -e "   Please configure credentials first using option 7 or editing .env."
        exit 1
    fi

    if [ "$live_trading" != "true" ]; then
        echo -e "${YELLOW}⚠️  Notice: LIVE_TRADING is currently set to 'false' in .env.${NC}"
        read -rp "Do you want to set LIVE_TRADING=true and proceed with real orders? (y/N): " confirm_live
        if [[ "$confirm_live" =~ ^[Yy]$ ]]; then
            sed -i "s|^LIVE_TRADING=.*|LIVE_TRADING=true|" "$SCRIPT_DIR/.env"
            echo -e "${GREEN}✅ LIVE_TRADING enabled.${NC}"
        else
            echo -e "${CYAN}Defaulting to PAPER trading mode.${NC}"
            "$PYTHON" -m src --mode paper "$@"
            return
        fi
    fi

    echo -e "${RED}${BOLD}======================================================================${NC}"
    echo -e "${RED}${BOLD}                       ⚠️  WARNING: LIVE TRADING  ⚠️                   ${NC}"
    echo -e "${RED}${BOLD}======================================================================${NC}"
    echo -e "  • Real orders will be submitted to Delta Exchange India (Futures)."
    echo -e "  • RiskManager strict rules enforced (Max loss limit, 1-position limit)."
    echo -e "  • All stop-loss and take-profit orders are strictly rounded to exchange tick sizes."
    echo -e "${RED}${BOLD}======================================================================${NC}\n"

    echo -e "${YELLOW}Pre-flight: Checking live Delta wallet balance...${NC}"
    "$PYTHON" -m src --mode status || true

    echo -e "\n${GREEN}▶ Launching scalping bot in LIVE mode...${NC}"
    echo -e "${DIM}Press Ctrl+C to gracefully shut down the bot.${NC}\n"
    sleep 1
    "$PYTHON" -m src --mode live "$@"
}

action_verify() {
    print_banner
    echo -e "${BLUE}▶ Running comprehensive Delta Exchange Futures & Tick Size verification...${NC}"
    "$PYTHON" scripts/verify_delta_execution.py "$@"
    echo ""
}

action_test() {
    print_banner
    echo -e "${BLUE}▶ Running automated test suite with pytest...${NC}"
    if [ -f "$SCRIPT_DIR/.venv/bin/pytest" ]; then
        "$SCRIPT_DIR/.venv/bin/pytest" -v tests/ "$@"
    else
        "$PYTHON" -m pytest -v tests/ "$@"
    fi
    echo ""
}

action_kill() {
    print_banner
    echo -e "${RED}▶ Terminating any running bot processes...${NC}"
    pkill -f "python.*-m src" 2>/dev/null && echo -e "${GREEN}✅ Terminated active bot instances.${NC}" || echo -e "${YELLOW}No active bot processes found.${NC}"
    echo ""
}

# ------------------------------------------------------------------------------
# 5. Interactive Menu Loop
# ------------------------------------------------------------------------------
interactive_menu() {
    while true; do
        print_banner
        display_config_summary
        echo -e "${BOLD}Please select an action:${NC}"
        echo -e "  ${GREEN}1)${NC} ${BOLD}Run LIVE Futures Trading${NC}        ${RED}[Real Delta Exchange Execution]${NC}"
        echo -e "  ${GREEN}2)${NC} ${BOLD}Run PAPER Trading Mode${NC}          ${CYAN}[$10,000 virtual balance simulation]${NC}"
        echo -e "  ${GREEN}3)${NC} ${BOLD}One-Shot Market Scanner${NC}         ${YELLOW}[Bootstraps 3000 candles & scans universe]${NC}"
        echo -e "  ${GREEN}4)${NC} ${BOLD}Check Account Balance & Status${NC}  ${BLUE}[Live wallet equity & active positions]${NC}"
        echo -e "  ${GREEN}5)${NC} ${BOLD}Verify Delta Exchange API${NC}       ${MAGENTA}[Validates BTC & ETH, tick rounding, wallet]${NC}"
        echo -e "  ${GREEN}6)${NC} ${BOLD}Run Automated Test Suite${NC}        ${CYAN}[48 Pytest unit & integration tests]${NC}"
        echo -e "  ${GREEN}7)${NC} ${BOLD}Configure API Credentials (.env)${NC}"
        echo -e "  ${GREEN}8)${NC} ${BOLD}Emergency Kill Switch / Stop${NC}    ${RED}[Terminates bot processes]${NC}"
        echo -e "  ${GREEN}9)${NC} Exit"
        echo ""
        read -rp "Enter selection [1-9]: " choice

        case "$choice" in
            1) action_live ;;
            2) action_paper ;;
            3) action_scan ;;
            4) action_status ;;
            5) action_verify ;;
            6) action_test ;;
            7) configure_credentials ;;
            8) action_kill ;;
            9|q|Q) echo -e "\n${GREEN}Goodbye!${NC}"; exit 0 ;;
            *) echo -e "\n${RED}Invalid option: $choice${NC}"; sleep 1 ;;
        esac

        if [ "$choice" != "1" ] && [ "$choice" != "2" ]; then
            echo -e "\n${DIM}Press Enter to return to main menu...${NC}"
            read -r
        fi
    done
}

# ------------------------------------------------------------------------------
# 6. CLI Argument Dispatcher
# ------------------------------------------------------------------------------
CMD="$1"
if [ -n "$CMD" ]; then
    shift || true
fi

case "$CMD" in
    live|--live)
        action_live "$@"
        ;;
    paper|--paper)
        action_paper "$@"
        ;;
    scan|--scan)
        action_scan "$@"
        ;;
    status|--status)
        action_status "$@"
        ;;
    verify|--verify)
        action_verify "$@"
        ;;
    test|--test)
        action_test "$@"
        ;;
    kill|--kill)
        action_kill "$@"
        ;;
    config|--config)
        configure_credentials
        ;;
    help|--help|-h)
        print_banner
        echo -e "Usage: ${BOLD}./all_in_one.sh [command] [options]${NC}\n"
        echo -e "Commands:"
        echo -e "  ${GREEN}live, --live${NC}      Run live trading with Delta Exchange Futures"
        echo -e "  ${GREEN}paper, --paper${NC}    Run paper trading simulation"
        echo -e "  ${GREEN}scan, --scan${NC}      Run one-shot market scanner"
        echo -e "  ${GREEN}status, --status${NC}  Check account balance and open positions"
        echo -e "  ${GREEN}verify, --verify${NC}  Run Delta API & precision verification"
        echo -e "  ${GREEN}test, --test${NC}      Run 48-test pytest suite"
        echo -e "  ${GREEN}kill, --kill${NC}      Emergency terminate bot processes"
        echo -e "  ${GREEN}config, --config${NC}  Configure .env credentials"
        echo -e "  (no arguments)    Launch interactive full-screen menu"
        echo ""
        ;;
    "")
        interactive_menu
        ;;
    *)
        echo -e "${RED}Unknown command: $CMD${NC}"
        echo -e "Run ${BOLD}./all_in_one.sh --help${NC} for usage."
        exit 1
        ;;
esac
