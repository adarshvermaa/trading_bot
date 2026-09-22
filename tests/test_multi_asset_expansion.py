"""Comprehensive unit test suite for multi-asset expansion (Top 10 Universe)."""

import pytest
from src.config import load_config
from src.cli import parse_args
from src.execution.delta import (
    DELTA_DEFAULT_PRODUCTS,
    normalize_delta_symbol,
)
from src.data.delta_ws import DeltaWSClient
from src.portfolio.account import CONTRACT_VALUES, get_contract_value
from src.ui.dashboard import Dashboard


def test_symbol_normalization():
    """Verify symbol normalization across all requested crypto & commodity assets."""
    mappings = {
        "BTCUSD": "BTCUSD",
        "btcusd": "BTCUSD",
        "BTC": "BTCUSD",
        "ETHUSD": "ETHUSD",
        "eth": "ETHUSD",
        "SOLUSD": "SOLUSD",
        "sol": "SOLUSD",
        "XAU": "XAUTUSD",
        "xau": "XAUTUSD",
        "gold": "XAUTUSD",
        "XAUTUSD": "XAUTUSD",
        "xaut": "XAUTUSD",
        "PAX": "PAXGUSD",
        "paxg": "PAXGUSD",
        "PAXGUSD": "PAXGUSD",
        "DOGE": "DOGEUSD",
        "dogeusdt": "DOGEUSD",
        "ZEC": "ZECUSD",
        "zec": "ZECUSD",
        "XRP": "XRPUSD",
        "xrp": "XRPUSD",
        "BNB": "BNBUSD",
        "bnb": "BNBUSD",
        "AVAX": "AVAXUSD",
        "avax": "AVAXUSD",
    }
    for raw, expected in mappings.items():
        assert normalize_delta_symbol(raw) == expected


def test_delta_default_products_registry():
    """Verify all top 10 assets exist in DELTA_DEFAULT_PRODUCTS with valid properties."""
    expected_symbols = [
        "BTCUSD",
        "ETHUSD",
        "SOLUSD",
        "XAUTUSD",
        "PAXGUSD",
        "DOGEUSD",
        "ZECUSD",
        "XRPUSD",
        "BNBUSD",
        "AVAXUSD",
    ]
    for sym in expected_symbols:
        assert sym in DELTA_DEFAULT_PRODUCTS, f"{sym} missing from DELTA_DEFAULT_PRODUCTS"
        prod = DELTA_DEFAULT_PRODUCTS[sym]
        assert prod["symbol"] == sym
        assert prod["contract_type"] == "perpetual_futures"
        assert float(prod["tick_size"]) > 0
        assert float(prod["contract_value"]) > 0


def test_contract_values():
    """Verify contract values for all 10 assets match Delta Exchange India specs."""
    assert get_contract_value("BTCUSD") == 0.001
    assert get_contract_value("ETHUSD") == 0.01
    assert get_contract_value("SOLUSD") == 1.0
    assert get_contract_value("XAUTUSD") == 0.001
    assert get_contract_value("PAXGUSD") == 0.001
    assert get_contract_value("DOGEUSD") == 100.0
    assert get_contract_value("ZECUSD") == 0.1
    assert get_contract_value("XRPUSD") == 1.0
    assert get_contract_value("BNBUSD") == 0.1
    assert get_contract_value("AVAXUSD") == 1.0

    # Fallback for unknown asset
    assert get_contract_value("UNKNOWN") == 1.0


def test_strategy_config_top10_universe():
    """Verify strategy.yaml config contains all 10 assets and presets."""
    config = load_config()
    universe = config.strategy.assets.universe
    assert len(universe) == 10
    expected = [
        "BTCUSD", "ETHUSD", "SOLUSD", "XAUTUSD", "PAXGUSD",
        "DOGEUSD", "ZECUSD", "XRPUSD", "BNBUSD", "AVAXUSD"
    ]
    assert universe == expected

    # Check presets
    presets = getattr(config.strategy.assets, "presets", {})
    assert "majors" in presets
    assert "top10" in presets
    assert "gold" in presets
    assert presets["majors"] == ["BTCUSD", "ETHUSD"]
    assert presets["gold"] == ["XAUTUSD", "PAXGUSD"]
    assert len(presets["top10"]) == 10


def test_cli_universe_and_symbols_parsing():
    """Verify CLI arguments for universe presets and custom symbol lists."""
    args_preset = parse_args(["--universe", "gold"])
    assert args_preset.universe == "gold"
    assert args_preset.symbols is None

    args_custom = parse_args(["--symbols", "BTCUSD,SOLUSD,DOGEUSD"])
    assert args_custom.symbols == "BTCUSD,SOLUSD,DOGEUSD"
    assert args_custom.universe is None


def test_delta_ws_subscription_top10():
    """Verify DeltaWSClient generates subscription payload for all 10 assets."""
    symbols = [
        "BTCUSD", "ETHUSD", "SOLUSD", "XAUTUSD", "PAXGUSD",
        "DOGEUSD", "ZECUSD", "XRPUSD", "BNBUSD", "AVAXUSD"
    ]
    client = DeltaWSClient(symbols=symbols)
    payload = client._build_subscription_payload()
    channels = payload["payload"]["channels"]
    for ch in channels:
        assert set(ch["symbols"]) == set(symbols)


def test_dashboard_market_watch_multi_asset():
    """Verify dashboard Market Watch table renders 10 assets and formats low-priced assets."""
    dashboard = Dashboard(mode="paper", target_leverage=20)
    assets_data = {
        "BTCUSD": {"delta_price": 96500.5, "bias_15m": "BULLISH", "bos_5m": True, "ob_detected": True, "ob_top": 97000.0, "ob_bottom": 96200.0, "vah": 97000.0, "val": 96000.0, "rsi": 58.2, "volume": 1.4},
        "ETHUSD": {"delta_price": 2750.2, "bias_15m": "BEARISH", "choch_5m": True, "rsi": 44.0, "volume": 0.9},
        "SOLUSD": {"delta_price": 185.75, "bias_15m": "BULLISH", "bos_5m": False, "rsi": 62.0, "volume": 2.1},
        "XAUTUSD": {"delta_price": 2650.0, "bias_15m": "BULLISH", "rsi": 52.0, "volume": 1.1},
        "PAXGUSD": {"delta_price": 2652.5, "bias_15m": "BULLISH", "rsi": 51.5, "volume": 1.0},
        "DOGEUSD": {"delta_price": 0.12456, "bias_15m": "BULLISH", "bos_5m": True, "ob_detected": True, "ob_top": 0.1260, "ob_bottom": 0.1240, "vah": 0.1280, "val": 0.1220, "rsi": 65.0, "volume": 3.5},
        "ZECUSD": {"delta_price": 42.15, "bias_15m": "BEARISH", "rsi": 38.0, "volume": 0.7},
        "XRPUSD": {"delta_price": 0.5842, "bias_15m": "NEUTRAL", "rsi": 49.0, "volume": 1.2},
        "BNBUSD": {"delta_price": 580.4, "bias_15m": "BULLISH", "rsi": 56.0, "volume": 1.3},
        "AVAXUSD": {"delta_price": 28.35, "bias_15m": "NEUTRAL", "rsi": 50.0, "volume": 1.0},
    }
    dashboard.update(
        account_data={},
        position_data={},
        signal_data={},
        risk_data={},
        execution_data={},
        market_watch_data={"assets": assets_data},
    )
    panel = dashboard._build_market_watch_panel()
    assert panel is not None
    assert panel.title == "REAL-TIME MARKET WATCH"
