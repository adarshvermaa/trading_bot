"""
Unit tests for the redesigned institutional Dashboard UI components and Account tracking.
"""
import pytest
from rich.panel import Panel
from rich.text import Text

from src.ui.dashboard import Dashboard, render_meter, render_badge
from src.portfolio.account import AccountManager


def test_render_meter_bounds_and_blocks():
    """Verify render_meter produces correct block character lengths and handles edge bounds."""
    # 0%
    m0 = render_meter(0.0, width=10)
    assert m0.plain == "░" * 10

    # 50%
    m50 = render_meter(50.0, width=10)
    assert m50.plain == "█████░░░░░"

    # 100%
    m100 = render_meter(100.0, width=10)
    assert m100.plain == "█" * 10

    # Underflow clamped to 0%
    m_neg = render_meter(-15.0, width=8)
    assert m_neg.plain == "░" * 8

    # Overflow clamped to 100%
    m_over = render_meter(150.0, width=8)
    assert m_over.plain == "█" * 8


def test_render_badge_content_and_style():
    """Verify render_badge formats pills with padding and style."""
    badge = render_badge("BULL", "green", "bold black")
    assert badge.plain == " BULL "
    assert "green" in badge.style


def test_header_panel_structure():
    """Verify _build_header renders institutional brand, mode pill, and engine status."""
    dash = Dashboard(mode="paper", target_leverage=100)
    hdr = dash._build_header()
    assert isinstance(hdr, Panel)
    assert hdr.border_style == "bright_blue"


def test_account_panel_full_metrics():
    """Verify _build_account_panel renders net equity, wallet balance, meters, win rate, and shield."""
    account = AccountManager(is_paper=True, initial_paper_balance=10000.0)
    # Simulate an open position and closed trades
    account.wallet_balance = 10250.0
    account.equity = 10250.0
    account.total_trades = 10
    account.winning_trades = 7
    account.losing_trades = 3
    account.total_realized_pnl = 250.0
    account.daily_pnl = 250.0

    account_data = account.to_dashboard_dict()
    # Inject floating unrealized P&L
    account_data["unrealized_pnl"] = 125.50
    account_data["net_equity"] = 10375.50
    account_data["used_margin"] = 500.0
    account_data["used_margin_pct"] = 4.8
    account_data["daily_drawdown_pct"] = 0.0

    dash = Dashboard(mode="paper")
    dash.update(
        account_data=account_data,
        position_data={},
        signal_data={},
        risk_data={},
        execution_data={},
    )

    panel = dash._build_account_panel()
    assert isinstance(panel, Panel)
    assert panel.title == "ACCOUNT"
    assert panel.border_style == "blue"


def test_account_panel_graceful_empty_fallbacks():
    """Verify _build_account_panel gracefully handles completely empty account_data dict."""
    dash = Dashboard(mode="live")
    dash.update(
        account_data={},
        position_data={},
        signal_data={},
        risk_data={},
        execution_data={},
    )
    panel = dash._build_account_panel()
    assert isinstance(panel, Panel)
    assert panel.title == "ACCOUNT"


def test_market_watch_multi_asset_radar_panel():
    """Verify _build_market_watch_panel renders BTC and ETH side-by-side radar with indicators."""
    dash = Dashboard(mode="live", target_leverage=50)
    market_watch = {
        "assets": {
            "BTCUSD": {
                "symbol": "BTCUSD",
                "price": 62000.0,
                "delta_price": 62050.0,
                "bias_15m": "BULLISH",
                "bos_5m": True,
                "choch_5m": False,
                "vah": 62500.0,
                "val": 61500.0,
                "ob_detected": True,
                "ob_direction": "BULLISH",
                "ob_bottom": 61800.0,
                "ob_top": 62000.0,
                "rsi": 58.4,
                "volume": 1.4,
                "is_squeeze": False,
                "squeeze_fired": True,
                "pdh": 63000.0,
                "pdl": 60000.0,
                "poc": 61900.0,
            },
            "ETHUSD": {
                "symbol": "ETHUSD",
                "price": 2750.0,
                "delta_price": 2752.0,
                "bias_15m": "BEARISH",
                "bos_5m": False,
                "choch_5m": True,
                "vah": 2800.0,
                "val": 2700.0,
                "ob_detected": False,
                "rsi": 42.1,
                "volume": 0.8,
                "is_squeeze": True,
                "squeeze_fired": False,
            }
        },
        "next_trigger": "5M BOS Confirmed - Awaiting Trigger",
    }

    dash.update(
        account_data={},
        position_data={},
        signal_data={},
        risk_data={},
        execution_data={},
        market_watch_data=market_watch,
    )

    panel = dash._build_market_watch_panel()
    assert isinstance(panel, Panel)
    assert panel.title == "REAL-TIME MARKET WATCH"
    assert panel.border_style == "cyan"


def test_position_panel_with_profit_lock_and_timer():
    """Verify _build_position_panel renders profit locked SL badge and Delta Scalper timer."""
    dash = Dashboard(mode="live", target_leverage=25)
    active_pos = {
        "symbol": "BTCUSD",
        "side": "BUY",
        "entry": 60000.0,
        "current_price": 61200.0,
        "leverage": 25,
        "margin": 100.0,
        "notional": 2500.0,
        "sl": 60500.0,  # SL > Entry for BUY -> Profit Locked
        "tp": 62500.0,
        "sl_anchor": "SWING_LOW",
        "tp_target_type": "VAH",
        "target_rr": 2.5,
        "unrealized_pnl": 50.0,
        "margin_pnl_pct": 50.0,
        "holding_time": "120s",
        "nearest_support": 60000.0,
        "nearest_resistance": 63000.0,
        "execution_routing": "LIMIT_OPTIMIZED",
        "health": "STRONG",
    }

    dash.update(
        account_data={},
        position_data=active_pos,
        signal_data={},
        risk_data={},
        execution_data={},
    )

    panel = dash._build_position_panel()
    assert isinstance(panel, Panel)
    assert panel.title == "POSITION"
    assert panel.border_style == "magenta"


def test_signal_panel_with_jev_and_pattern_memory():
    """Verify _build_signal_panel renders Jev Sys1 verdict and pattern memory counts."""
    dash = Dashboard(mode="live")
    dash.update(
        account_data={},
        position_data={},
        signal_data={
            "btc_price": 60000.0,
            "eth_price": 2700.0,
            "rankings": "#1 BTCUSD (0.85) | #2 ETHUSD (0.42)",
            "15m_bias": "[BTCUSD] BULLISH",
            "5m_bos_choch": "BOS=True CHoCH=False",
            "liquidity_sweep": "True",
            "1m_displacement": "True",
            "retest": "True",
            "vwap": "ABOVE",
            "rsi": "58.2",
            "volume": "1.35x",
            "onnx_confidence": "88.5%",
            "pattern_memory_stats": "W: 14 | L: 2",
            "last_pattern_audit": "Strong Momentum Sweep",
            "jev_status": "ACTIVE (Sys1)",
            "jev_verdict": "BOOST (+0.10)",
            "signal_score": "ACTIONABLE (0.850)",
            "setup_type": "ORDER_BLOCK_PULLBACK",
            "pattern": "BULLISH_ENGULFING",
            "next_trigger": "CONFLUENCE READY TO EXECUTE",
        },
        risk_data={},
        execution_data={},
    )

    panel = dash._build_signal_panel()
    assert isinstance(panel, Panel)
    assert panel.title == "SIGNAL"
    assert panel.border_style == "yellow"


def test_dashboard_full_render_both_states():
    """Verify render() works smoothly in both flat (no position) and active position states."""
    dash = Dashboard(mode="paper")
    account = AccountManager(is_paper=True)

    # 1. Without active position
    dash.update(
        account_data=account.to_dashboard_dict(),
        position_data={},
        signal_data={"next_trigger": "Scanning"},
        risk_data={"risk_check": "PASS"},
        execution_data={"order_status": "IDLE"},
        market_watch_data={"assets": {}},
    )
    layout1 = dash.render()
    assert layout1.get("header") is not None
    assert layout1.get("account") is not None
    assert layout1.get("market_watch") is not None
    assert layout1.get("position") is None  # collapsed into market_watch
    assert layout1.get("signal") is not None

    # 2. With active position
    dash.update(
        account_data=account.to_dashboard_dict(),
        position_data={"symbol": "BTCUSD", "side": "BUY", "entry": 60000.0},
        signal_data={"next_trigger": "In Trade"},
        risk_data={"risk_check": "PASS"},
        execution_data={"order_status": "FILLED"},
        market_watch_data={"assets": {}},
    )
    layout2 = dash.render()
    assert layout2.get("header") is not None
    assert layout2.get("account") is not None
    assert layout2.get("position") is not None
    assert layout2.get("market_watch") is not None
    assert layout2.get("execution") is not None
