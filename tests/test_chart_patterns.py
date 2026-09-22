"""
Unit tests for TradingView Chart Pattern & Price Action Engine (src/strategy/chart_patterns.py).
"""
import numpy as np
import pytest
from src.strategy.chart_patterns import (
    ChartPatternEngine,
    ChartPatternType,
    PricingZone,
    calculate_premium_discount_zone,
    detect_double_bottom,
    detect_double_top,
    detect_head_and_shoulders,
    detect_inverse_head_and_shoulders,
    detect_triangles_and_flags,
)


def test_premium_discount_zone():
    range_high = 70000.0
    range_low = 60000.0

    # Price in discount (< 48% => < 64800)
    zone, pct = calculate_premium_discount_zone(62000.0, range_high, range_low)
    assert zone == PricingZone.DISCOUNT.value
    assert pct < 48.0

    # Price in premium (> 52% => > 65200)
    zone, pct = calculate_premium_discount_zone(68000.0, range_high, range_low)
    assert zone == PricingZone.PREMIUM.value
    assert pct > 52.0

    # Price at equilibrium (65000 = 50%)
    zone, pct = calculate_premium_discount_zone(65000.0, range_high, range_low)
    assert zone == PricingZone.EQUILIBRIUM.value
    assert 48.0 <= pct <= 52.0


def test_detect_double_bottom():
    n = 35
    lows = np.full(n, 63000.0)
    highs = np.full(n, 64000.0)
    closes = np.full(n, 63500.0)

    # First bottom at index 8
    lows[8] = 60000.0
    highs[8] = 60500.0
    closes[8] = 60200.0

    # Intermediate peak at index 16
    lows[16] = 61800.0
    highs[16] = 62200.0
    closes[16] = 62000.0

    # Second bottom at index 26
    lows[26] = 60050.0  # within 0.3% of first bottom
    highs[26] = 60600.0
    closes[26] = 60300.0

    # Recent close moving up near or above neckline
    closes[-1] = 62100.0

    res = detect_double_bottom(lows, highs, closes, swing_lows=[8, 26])
    assert res is not None
    assert res["pattern_type"] == ChartPatternType.DOUBLE_BOTTOM.value
    assert res["direction"] == "BULLISH"
    assert res["confidence"] >= 0.70
    assert res["target_price"] > 62000.0


def test_detect_double_top():
    n = 35
    lows = np.full(n, 60800.0)
    highs = np.full(n, 61200.0)
    closes = np.full(n, 61000.0)

    # First top at index 8
    highs[8] = 62000.0
    lows[8] = 61500.0
    closes[8] = 61800.0

    # Intermediate trough at index 16 (neckline at 59800)
    highs[16] = 60200.0
    lows[16] = 59800.0
    closes[16] = 60000.0

    # Second top at index 26
    highs[26] = 61950.0  # within 0.3%
    lows[26] = 61400.0
    closes[26] = 61700.0

    # Recent close moving down near or below neckline
    closes[-1] = 60100.0

    res = detect_double_top(highs, lows, closes, swing_highs=[8, 26])
    assert res is not None
    assert res["pattern_type"] == ChartPatternType.DOUBLE_TOP.value
    assert res["direction"] == "BEARISH"
    assert res["confidence"] >= 0.70
    assert res["target_price"] < 60000.0


def test_detect_head_and_shoulders():
    n = 35
    lows = np.full(n, 60000.0)
    highs = np.full(n, 61000.0)
    closes = np.full(n, 60500.0)

    # Left shoulder: peak at 63000 (idx 6)
    highs[6] = 63000.0
    lows[10] = 61000.0
    # Head: peak at 65000 (idx 16)
    highs[16] = 65000.0
    lows[20] = 61000.0
    # Right shoulder: peak at 63100 (idx 26)
    highs[26] = 63100.0

    closes[-1] = 61500.0

    res = detect_head_and_shoulders(highs, lows, closes, swing_highs=[6, 16, 26], swing_lows=[10, 20])
    assert res is not None
    assert res["pattern_type"] == ChartPatternType.HEAD_AND_SHOULDERS.value
    assert res["direction"] == "BEARISH"
    assert res["confidence"] >= 0.70


def test_detect_inverse_head_and_shoulders():
    n = 35
    lows = np.full(n, 65000.0)
    highs = np.full(n, 66000.0)
    closes = np.full(n, 65500.0)

    # Left shoulder trough at 62000 (idx 6)
    lows[6] = 62000.0
    highs[10] = 64000.0
    # Head trough at 60000 (idx 16)
    lows[16] = 60000.0
    highs[20] = 64000.0
    # Right shoulder trough at 61900 (idx 26)
    lows[26] = 61900.0

    closes[-1] = 63500.0

    res = detect_inverse_head_and_shoulders(highs, lows, closes, swing_highs=[10, 20], swing_lows=[6, 16, 26])
    assert res is not None
    assert res["pattern_type"] == ChartPatternType.INVERSE_HEAD_AND_SHOULDERS.value
    assert res["direction"] == "BULLISH"
    assert res["confidence"] >= 0.70


def test_chart_pattern_engine_analyze():
    engine = ChartPatternEngine()
    
    # Flat data -> None pattern, equilibrium
    n = 30
    highs = np.linspace(60000, 60100, n)
    lows = np.linspace(59900, 60000, n)
    closes = np.linspace(59950, 60050, n)
    volumes = np.full(n, 10.0)

    res = engine.analyze(
        high=highs,
        low=lows,
        close=closes,
        volume=volumes,
        swing_highs=[],
        swing_lows=[],
        range_high=61000.0,
        range_low=59000.0,
    )
    assert res.pricing_zone in [z.value for z in PricingZone]
    assert 0.0 <= res.range_position_pct <= 100.0
