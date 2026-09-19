import asyncio
import json
import pytest
from rich.panel import Panel

from src.data.delta_ws import DeltaWSClient, Candle
from src.ui.dashboard import Dashboard
from src.portfolio.account import AccountManager
from src.risk.risk_manager import RiskManager
from src.config import RiskConfig


def test_delta_ws_get_latest_price_tick_and_fallback():
    """Verify DeltaWSClient stores and returns latest prices on ticks and falls back to candles."""
    client = DeltaWSClient(symbols=["BTCUSD", "ETHUSD"])

    # Initial state: None
    assert client.get_latest_price("BTCUSD") is None

    # Simulate candle in store
    client.store.add_candle(
        "BTCUSD",
        "1m",
        Candle(
            open_time=1000,
            open=80000.0,
            high=81000.0,
            low=79900.0,
            close=80500.0,
            volume=10.0,
            close_time=2000,
            is_closed=True,
        ),
    )
    # Should fall back to candle store close price
    assert client.get_latest_price("BTCUSD") == 80500.0

    # Simulate live incoming tick price
    client._latest_prices["BTCUSD"] = 81250.0
    assert client.get_latest_price("BTCUSD") == 81250.0
    # Case insensitivity
    assert client.get_latest_price("btcusd") == 81250.0


@pytest.mark.asyncio
async def test_delta_ws_handle_message_records_unclosed_tick_price():
    """Verify incoming candlestick and ticker messages update _latest_prices."""
    client = DeltaWSClient(symbols=["BTCUSD"])

    msg = json.dumps({
        "type": "candlestick_1m",
        "symbol": "BTCUSD",
        "resolution": "1m",
        "candle_start_time": 1000 * 1_000_000,
        "open": 81000.0,
        "high": 81400.0,
        "low": 80950.0,
        "close": 81345.50,
        "volume": 15.5,
    })

    await client._handle_message(msg)

    # Price should be immediately captured
    assert client.get_latest_price("BTCUSD") == 81345.50
    assert client.get_latest_price("btcusd") == 81345.50


def test_dashboard_market_watch_rendered_when_no_active_position():
    """Verify Dashboard renders REAL-TIME MARKET WATCH panel when position is empty or --."""
    dashboard = Dashboard(mode="live")
    account = AccountManager(is_paper=False)
    rm = RiskManager(RiskConfig())

    market_watch_data = {
        "assets": {
            "BTCUSD": {
                "symbol": "BTCUSD",
                "price": 81500.0,
                "bias_15m": "BULLISH",
                "bos_5m": False,
                "choch_5m": False,
                "support": 80000.0,
                "resistance": 82000.0,
                "direction": "NONE",
                "score": 0.65,
                "rsi": 52.0,
                "actionable": False,
            },
            "ETHUSD": {
                "symbol": "ETHUSD",
                "price": 2650.0,
                "bias_15m": "BULLISH",
                "bos_5m": False,
                "choch_5m": False,
                "support": 2600.0,
                "resistance": 2700.0,
                "direction": "NONE",
                "score": 0.58,
                "rsi": 49.5,
                "actionable": False,
            },
        },
        "next_trigger": "Awaiting 5M BOS/CHoCH + ML >= 65%",
    }

    dashboard.update(
        account_data=account.to_dashboard_dict(),
        position_data={},
        signal_data={
            "btc_price": 81500.0,
            "eth_price": 2650.0,
            "rankings": "#1 BTCUSD (0.65, NONE) | #2 ETHUSD (0.58, NONE)",
            "15m_bias": "[BTCUSD] BULLISH",
            "5m_bos_choch": "BOS=False CHoCH=False",
            "liquidity_sweep": "False",
            "1m_displacement": "False",
            "retest": "False",
            "vwap": "ABOVE",
            "rsi": "52.0",
            "volume": "1.10x",
            "onnx_confidence": "--",
            "llm_status": "Disabled",
            "signal_score": "RANK #1 BTCUSD (0.65)",
            "next_trigger": "Awaiting 5M BOS/CHoCH + ML >= 65%",
        },
        risk_data=rm.get_risk_status(),
        execution_data={},
        market_watch_data=market_watch_data,
    )

    pos_panel = dashboard._build_position_panel()
    assert isinstance(pos_panel, Panel)
    assert pos_panel.title == "REAL-TIME MARKET WATCH"

    layout = dashboard.render()
    assert layout is not None


def test_dashboard_position_panel_rendered_when_position_is_active():
    """Verify Dashboard renders POSITION panel when an active trade is open."""
    dashboard = Dashboard(mode="live")
    account = AccountManager(is_paper=False)
    rm = RiskManager(RiskConfig())

    active_pos_data = {
        "symbol": "BTCUSD",
        "side": "BUY",
        "entry": 81000.0,
        "current_price": 81250.0,
        "leverage": 150,
        "margin": 2.0,
        "notional": 300.0,
        "sl": 80730.0,
        "tp": 81540.0,
        "unrealized_pnl": 0.25,
        "margin_pnl_pct": 12.5,
        "holding_time": "15s",
        "nearest_support": 80500.0,
        "nearest_resistance": 82000.0,
        "health": "STRONG",
    }

    dashboard.update(
        account_data=account.to_dashboard_dict(),
        position_data=active_pos_data,
        signal_data={"btc_price": 81250.0, "eth_price": 2650.0},
        risk_data=rm.get_risk_status(),
        execution_data={"order_id": "delta_999", "order_status": "FILLED"},
    )

    pos_panel = dashboard._build_position_panel()
    assert isinstance(pos_panel, Panel)
    assert pos_panel.title == "POSITION"


def test_signal_panel_includes_live_feeds_and_next_trigger():
    """Verify Signal panel includes live BTC/ETH tickers and next trigger condition."""
    dashboard = Dashboard(mode="paper")
    dashboard.update(
        account_data={},
        position_data={},
        signal_data={
            "btc_price": 81200.0,
            "eth_price": 2640.0,
            "rankings": "#1 BTCUSD | #2 ETHUSD",
            "15m_bias": "[BTCUSD] BULLISH",
            "5m_bos_choch": "BOS=False",
            "liquidity_sweep": "False",
            "1m_displacement": "False",
            "retest": "False",
            "vwap": "ABOVE",
            "rsi": "48.0",
            "volume": "0.95x",
            "onnx_confidence": "--",
            "signal_score": "RANK #1 BTCUSD (0.50)",
            "next_trigger": "Awaiting 5M BOS/CHoCH + ML >= 65%",
        },
        risk_data={},
        execution_data={},
    )

    sig_panel = dashboard._build_signal_panel()
    assert isinstance(sig_panel, Panel)
    assert sig_panel.title == "SIGNAL"


def test_market_watch_and_position_both_rendered_in_layout_when_position_active():
    """Verify that when a position is active, the layout contains BOTH POSITION and MARKET WATCH."""
    dashboard = Dashboard(mode="live")
    account = AccountManager(is_paper=False)
    rm = RiskManager(RiskConfig())

    active_pos_data = {
        "symbol": "BTCUSD",
        "side": "BUY",
        "entry": 81000.0,
        "current_price": 81250.0,
        "leverage": 150,
        "margin": 2.0,
        "notional": 300.0,
        "sl": 80730.0,
        "tp": 81540.0,
        "unrealized_pnl": 0.25,
        "margin_pnl_pct": 12.5,
        "holding_time": "15s",
    }

    dashboard.update(
        account_data=account.to_dashboard_dict(),
        position_data=active_pos_data,
        signal_data={"btc_price": 81250.0, "eth_price": 2650.0},
        risk_data=rm.get_risk_status(),
        execution_data={"order_id": "delta_999", "order_status": "FILLED"},
        market_watch_data={"assets": {}},
    )

    layout = dashboard.render()
    assert layout.get("position") is not None
    assert layout.get("market_watch") is not None
    assert layout.get("account") is not None


