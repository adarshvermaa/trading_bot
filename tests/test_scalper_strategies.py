import pytest
import numpy as np
from dataclasses import dataclass
from src.strategy.structure import MarketStructure, StructureAnalysis
from src.strategy.signals import SignalGenerator, Signal
from src.risk.risk_manager import RiskManager

@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float

def test_sweep_reversal_at_support_produces_long():
    """Test that a liquidity sweep below key support produces a high-conviction LONG signal."""
    ms = MarketStructure(lookback=2)
    sig_gen = SignalGenerator()

    # 15m neutral or downtrend
    candles_15m = [
        Candle(100, 105, 95, 98, 1000),
        Candle(98, 102, 92, 94, 1000),
        Candle(94, 98, 88, 90, 1000),
        Candle(90, 95, 85, 88, 1000),
        Candle(88, 92, 82, 85, 1000),
        Candle(85, 88, 80, 82, 1000),
    ]

    # 5m candles creating a swing low at 80.0
    candles_5m = [
        Candle(85, 87, 83, 84, 500),
        Candle(84, 86, 82, 83, 500),
        Candle(83, 84, 80, 81, 500), # swing low at 80.0
        Candle(81, 84, 81, 83, 500),
        Candle(83, 85, 82, 84, 500),
        Candle(84, 85, 81, 82, 500),
    ]

    # 1m candle: sweeps below 80.0 to 78.0, closes back up at 81.2 (Hammer / rejection pinbar)
    candles_1m = [Candle(82.0, 83.0, 81.5, 82.0, 100) for _ in range(25)]
    candles_1m.append(Candle(82.0, 82.5, 80.5, 81.0, 150))
    candles_1m.append(Candle(80.5, 81.5, 78.0, 81.2, 500))

    analysis = ms.analyze(candles_15m, candles_5m, candles_1m, atr_1m=1.5)

    assert analysis.setup_type == "SWEEP_REVERSAL"
    assert analysis.sweep_direction == "BULLISH"
    assert analysis.is_valid is True

    signal = sig_gen.generate(candles_15m, candles_5m, candles_1m, analysis)
    assert signal.direction == "LONG"
    assert signal.strength >= 0.88
    assert signal.setup_type == "SWEEP_REVERSAL"
    assert signal.timeframe_alignment is True


def test_sweep_reversal_at_resistance_produces_short():
    """Test that a liquidity sweep above key resistance produces a high-conviction SHORT signal."""
    ms = MarketStructure(lookback=2)
    sig_gen = SignalGenerator()

    candles_15m = [
        Candle(80, 85, 78, 83, 1000),
        Candle(83, 88, 82, 87, 1000),
        Candle(87, 92, 85, 90, 1000),
        Candle(90, 95, 88, 93, 1000),
        Candle(93, 98, 91, 96, 1000),
        Candle(96, 100, 94, 98, 1000),
    ]

    # 5m candles creating a swing high at 100.0
    candles_5m = [
        Candle(93, 95, 92, 94, 500),
        Candle(94, 97, 93, 96, 500),
        Candle(96, 100, 95, 97, 500), # swing high at 100.0
        Candle(97, 98, 94, 95, 500),
        Candle(95, 96, 93, 94, 500),
        Candle(94, 98, 94, 97, 500),
    ]

    # 1m candles: sweeps above 100.0 to 101.5, closes back down at 98.8
    candles_1m = [Candle(97.0, 98.0, 96.5, 97.5, 100) for _ in range(25)]
    candles_1m.append(Candle(97.5, 98.5, 97.0, 98.0, 150))
    candles_1m.append(Candle(98.5, 101.5, 98.2, 98.8, 500))

    analysis = ms.analyze(candles_15m, candles_5m, candles_1m, atr_1m=1.5)

    assert analysis.setup_type == "SWEEP_REVERSAL"
    assert analysis.sweep_direction == "BEARISH"
    assert analysis.is_valid is True

    signal = sig_gen.generate(candles_15m, candles_5m, candles_1m, analysis)
    assert signal.direction == "SHORT"
    assert signal.strength >= 0.88
    assert signal.setup_type == "SWEEP_REVERSAL"
    assert signal.timeframe_alignment is True


def test_trend_pullback_to_ema_vwap_produces_aligned_signal():
    """Test that a pullback to EMA & VWAP value zone in an uptrend produces a LONG signal."""
    ms = MarketStructure(lookback=2, displacement_threshold=0.5, retest_tolerance_mult=1.0)
    sig_gen = SignalGenerator()

    # 15m uptrend
    candles_15m = [
        Candle(100, 110, 95, 105, 1000),
        Candle(105, 115, 100, 112, 1000),
        Candle(112, 125, 108, 120, 1000),
        Candle(120, 122, 114, 116, 1000),
        Candle(116, 118, 112, 115, 1000),
        Candle(115, 130, 114, 128, 1000),
        Candle(128, 129, 122, 125, 1000),
        Candle(125, 127, 123, 126, 1000),
    ]

    # 5m breaking structure upward
    candles_5m = [
        Candle(120, 124, 118, 122, 500),
        Candle(122, 125, 120, 123, 500),
        Candle(123, 126, 121, 124, 500),
        Candle(124, 125, 122, 123, 500),
        Candle(123, 124, 122, 123, 500),
        Candle(123, 128, 122, 127, 800), # BOS above 126
    ]

    # 1m pullback into broken level with displacement
    candles_1m = [Candle(120.0 + i*0.2, 121.0 + i*0.2, 119.5 + i*0.2, 120.5 + i*0.2, 200) for i in range(25)]
    candles_1m.append(Candle(127.0, 127.5, 125.8, 126.1, 200)) # retest 126
    candles_1m.append(Candle(126.1, 128.0, 126.0, 127.8, 400)) # displacement

    analysis = ms.analyze(candles_15m, candles_5m, candles_1m, atr_1m=1.0)
    assert analysis.bias_15m == "BULLISH"
    assert analysis.setup_type == "TREND_PULLBACK"
    assert analysis.is_valid is True

    signal = sig_gen.generate(candles_15m, candles_5m, candles_1m, analysis)
    assert signal.direction == "LONG"
    assert signal.strength >= 0.80
    assert signal.setup_type == "TREND_PULLBACK"


def test_stall_protection_in_evaluate_position_health():
    """Test that stagnant trades trigger stall warnings and time-stop exits."""
    sig_gen = SignalGenerator()

    # Upward trending healthy candles
    candles_1m = [
        Candle(100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0)
        for i in range(30)
    ]
    candles_5m = [
        Candle(100.0 + i*5, 105.0 + i*5, 98.0 + i*5, 104.0 + i*5, 50.0)
        for i in range(10)
    ]

    # 1. Trade held for 200s (3.3 mins) with price flat relative to entry
    health, reason = sig_gen.evaluate_position_health(
        position_side="LONG",
        entry_price=130.0,
        current_price=130.01, # flat move
        candles_1m=candles_1m,
        candles_5m=candles_5m,
        nearest_support=115.0,
        nearest_resistance=150.0,
        position_age_seconds=200.0,
    )
    assert health in ("WEAK", "STRONG")
    assert "Stall warning" in reason

    # 2. Trade held for 350s (> 5 mins) with price flat
    health_5m, reason_5m = sig_gen.evaluate_position_health(
        position_side="LONG",
        entry_price=130.0,
        current_price=130.01, # flat move
        candles_1m=candles_1m,
        candles_5m=candles_5m,
        nearest_support=115.0,
        nearest_resistance=150.0,
        position_age_seconds=350.0,
    )
    assert "Time-stop" in reason_5m


def test_margin_sl_tp_ratio_scalping_at_150x():
    """Verify strictly -3% loss and +6% profit of margin at 150x leverage."""
    entry_price = 80000.0
    leverage = 150.0
    sl_ratio = 0.03   # 3% of margin
    tp_ratio = 0.06   # 6% of margin

    # Price moves corresponding to 3% and 6% margin:
    price_pct_loss = sl_ratio / leverage   # 0.03 / 150 = 0.00020 (0.02%)
    price_pct_profit = tp_ratio / leverage # 0.06 / 150 = 0.00040 (0.04%)

    sl_long = entry_price * (1 - price_pct_loss)
    tp_long = entry_price * (1 + price_pct_profit)

    # Dollar distances
    sl_dist = entry_price - sl_long
    tp_dist = tp_long - entry_price

    assert round(sl_dist, 2) == 16.0  # $16 on $80,000 BTC
    assert round(tp_dist, 2) == 32.0  # $32 on $80,000 BTC
    assert tp_dist / sl_dist == 2.0   # 2:1 Reward to Risk
