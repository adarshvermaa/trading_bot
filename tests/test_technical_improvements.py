import pytest
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

from src.strategy.structure import MarketStructure, StructureAnalysis
from src.strategy.signals import SignalGenerator
from src.strategy.regime import evaluate_session, SessionKillZone
from src.risk.risk_manager import RiskManager
from src.execution.delta import DeltaExchangeClient
from src.execution.order_manager import OrderManager, OrderState
from src.config import load_config


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float


# ==============================================================================
# 1. ORDERBOOK IMBALANCE (OBI) FILTER TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_get_orderbook_imbalance_calculation():
    """Verify OBI formula: (bid_vol - ask_vol) / (bid_vol + ask_vol)."""
    delta_client = DeltaExchangeClient(api_key="", api_secret="", live_trading=False)
    
    # Heavy bids (buyers dominating: bid_vol=80, ask_vol=20 => OBI = +0.60)
    delta_client.get_orderbook = AsyncMock(return_value={
        "buy": [{"price": "90000", "size": "50"}, {"price": "89950", "size": "30"}],
        "sell": [{"price": "90050", "size": "10"}, {"price": "90100", "size": "10"}],
    })
    obi_bullish = await delta_client.get_orderbook_imbalance("BTCUSD", depth=5)
    assert pytest.approx(obi_bullish, 0.01) == 0.60

    # Heavy asks (sellers dominating: bid_vol=10, ask_vol=90 => OBI = -0.80)
    delta_client.get_orderbook = AsyncMock(return_value={
        "buy": [{"price": "90000", "size": "10"}],
        "sell": [{"price": "90050", "size": "90"}],
    })
    obi_bearish = await delta_client.get_orderbook_imbalance("BTCUSD", depth=5)
    assert pytest.approx(obi_bearish, 0.01) == -0.80


@pytest.mark.asyncio
async def test_obi_filter_blocks_long_into_heavy_ask_wall():
    """Verify OrderManager blocks LONG orders when OBI < -0.30 (heavy ask resistance)."""
    config = load_config()
    delta_mock = AsyncMock()
    delta_mock.live_trading = True
    delta_mock.get_product.return_value = {
        "id": 1,
        "symbol": "BTCUSD",
        "state": "live",
        "trading_status": "operational",
        "contract_type": "perpetual_futures",
        "max_leverage": 100,
        "contract_value": 0.001,
        "tick_size": 0.5,
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
    }
    delta_mock.get_positions.return_value = []
    delta_mock.get_orderbook_imbalance.return_value = -0.65  # Heavy ask wall

    risk_mock = MagicMock()
    risk_mock.validate_leverage.return_value = (True, 10, "VALID")

    om = OrderManager(delta_mock, risk_mock, config=config.strategy)
    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="LONG",
        price=90000.0,
        size=1.0,
        leverage=10,
        equity=1000.0,
        atr=50.0,
    )
    # Must be blocked by OBI filter
    assert res is None
    delta_mock.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_obi_filter_blocks_short_into_heavy_bid_wall():
    """Verify OrderManager blocks SHORT orders when OBI > +0.30 (heavy bid support)."""
    config = load_config()
    delta_mock = AsyncMock()
    delta_mock.live_trading = True
    delta_mock.get_product.return_value = {
        "id": 1,
        "symbol": "BTCUSD",
        "state": "live",
        "trading_status": "operational",
        "contract_type": "perpetual_futures",
        "max_leverage": 100,
        "contract_value": 0.001,
        "tick_size": 0.5,
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
    }
    delta_mock.get_positions.return_value = []
    delta_mock.get_orderbook_imbalance.return_value = +0.55  # Heavy bid support wall

    risk_mock = MagicMock()
    risk_mock.validate_leverage.return_value = (True, 10, "VALID")

    om = OrderManager(delta_mock, risk_mock, config=config.strategy)
    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="SHORT",
        price=90000.0,
        size=1.0,
        leverage=10,
        equity=1000.0,
        atr=50.0,
    )
    # Must be blocked by OBI filter
    assert res is None
    delta_mock.place_order.assert_not_called()


# ==============================================================================
# 2. FAIR VALUE GAPS (FVG) & STRUCTURAL TAKE PROFIT TESTS
# ==============================================================================

def test_detect_bullish_fair_value_gap():
    """Verify detection of a 3-candle Bullish FVG (Low[i] > High[i-2])."""
    ms = MarketStructure()
    highs = np.array([100.0, 110.0, 115.0])
    lows = np.array([95.0, 99.0, 108.0])
    closes = np.array([98.0, 109.0, 112.0])

    detected, direction, top, bottom, testing = ms.detect_fair_value_gaps(
        highs, lows, closes, atr=5.0
    )
    assert detected is True
    assert direction == "BULLISH"
    assert top == 108.0
    assert bottom == 100.0


def test_detect_bearish_fair_value_gap():
    """Verify detection of a 3-candle Bearish FVG (High[i] < Low[i-2])."""
    ms = MarketStructure()
    highs = np.array([120.0, 111.0, 102.0])
    lows = np.array([110.0, 100.0, 95.0])
    closes = np.array([112.0, 101.0, 97.0])

    detected, direction, top, bottom, testing = ms.detect_fair_value_gaps(
        highs, lows, closes, atr=5.0
    )
    assert detected is True
    assert direction == "BEARISH"
    assert top == 110.0
    assert bottom == 102.0


def test_structural_take_profit_targeting():
    """Verify RiskManager anchors TP at structural swing level when R:R >= 1.5R."""
    config = load_config()
    rm = RiskManager(config.risk)
    
    # Long trade: Entry at 80,000, SL at 79,200 (risk = 800)
    # Swing resistance target at 82,000 (reward = 2,000 => 2,000 / 800 = 2.5R >= 1.5R)
    tp = rm.calculate_take_profit(
        entry_price=80000.0,
        side="LONG",
        margin=100.0,
        leverage=10,
        contract_value=0.001,
        size=1,
        tick_size=0.5,
        structural_target=82000.0,
        sl_price=79200.0,
    )
    assert tp == 82000.0

    # Short trade: Entry at 80,000, SL at 80,800 (risk = 800)
    # Swing support target at 78,000 (reward = 2,000 => 2.5R >= 1.5R)
    tp_short = rm.calculate_take_profit(
        entry_price=80000.0,
        side="SHORT",
        margin=100.0,
        leverage=10,
        contract_value=0.001,
        size=1,
        tick_size=0.5,
        structural_target=78000.0,
        sl_price=80800.0,
    )
    assert tp_short == 78000.0


def test_signal_generator_assigns_structural_target():
    """Verify SignalGenerator populates structural_target_tp from structure swing S/R."""
    ms = MarketStructure(lookback=2)
    sig_gen = SignalGenerator()

    candles_15m = [
        Candle(100, 105, 95, 98, 1000),
        Candle(98, 102, 92, 94, 1000),
        Candle(94, 98, 88, 90, 1000),
        Candle(90, 95, 85, 88, 1000),
        Candle(88, 92, 82, 85, 1000),
        Candle(85, 88, 80, 82, 1000),
    ]

    candles_5m = [
        Candle(85, 87, 83, 84, 500),
        Candle(84, 86, 82, 83, 500),
        Candle(83, 84, 80, 81, 500), # swing low at 80.0
        Candle(81, 84, 81, 83, 500),
        Candle(83, 85, 82, 84, 500),
        Candle(84, 85, 81, 82, 500),
    ]

    candles_1m = [Candle(82.0, 83.0, 81.5, 82.0, 100) for _ in range(25)]
    candles_1m.append(Candle(82.0, 82.5, 80.5, 81.0, 150))
    candles_1m.append(Candle(80.5, 81.5, 78.0, 81.2, 500))

    analysis = ms.analyze(candles_15m, candles_5m, candles_1m, atr_1m=1.5)
    analysis.nearest_resistance = 95.0

    signal = sig_gen.generate(candles_15m, candles_5m, candles_1m, analysis)
    assert signal.direction == "LONG"
    assert signal.structural_target_tp == 95.0


# ==============================================================================
# 3. SESSION KILL ZONES TESTS
# ==============================================================================

def test_evaluate_session_kill_zones():
    """Verify accurate detection of London Open, New York Open, and Asian Dead Zone."""
    # London Open: 08:30 UTC
    t_london = datetime(2026, 9, 20, 8, 30, tzinfo=timezone.utc)
    s_london = evaluate_session(utc_time=t_london)
    assert s_london.zone == SessionKillZone.LONDON_OPEN
    assert s_london.is_prime_kill_zone is True
    assert s_london.is_dead_zone is False
    assert s_london.recommended_min_confidence == 0.65

    # New York Open: 14:00 UTC
    t_ny = datetime(2026, 9, 20, 14, 0, tzinfo=timezone.utc)
    s_ny = evaluate_session(utc_time=t_ny)
    assert s_ny.zone == SessionKillZone.NEW_YORK_OPEN
    assert s_ny.is_prime_kill_zone is True
    assert s_ny.is_dead_zone is False

    # London Close / NY Afternoon: 16:30 UTC
    t_close = datetime(2026, 9, 20, 16, 30, tzinfo=timezone.utc)
    s_close = evaluate_session(utc_time=t_close)
    assert s_close.zone == SessionKillZone.LONDON_CLOSE
    assert s_close.is_prime_kill_zone is False
    assert s_close.is_dead_zone is False

    # Asian Dead Zone: 03:30 UTC
    t_asian = datetime(2026, 9, 20, 3, 30, tzinfo=timezone.utc)
    s_asian = evaluate_session(utc_time=t_asian, dead_zone_confidence=0.75)
    assert s_asian.zone == SessionKillZone.ASIAN_DEAD_ZONE
    assert s_asian.is_dead_zone is True
    assert s_asian.is_prime_kill_zone is False
    assert s_asian.recommended_min_confidence == 0.75

    # Normal Session: 21:00 UTC
    t_normal = datetime(2026, 9, 20, 21, 0, tzinfo=timezone.utc)
    s_normal = evaluate_session(utc_time=t_normal)
    assert s_normal.zone == SessionKillZone.NORMAL_SESSION
    assert s_normal.is_dead_zone is False
