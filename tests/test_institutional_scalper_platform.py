import pytest
import numpy as np
from datetime import datetime, timezone

from src.strategy.structure import (
    MarketStructure,
    StructureAnalysis,
    compute_volume_profile,
    compute_volume_poc,
    detect_equal_highs_lows,
    detect_order_blocks,
    calculate_asian_range,
)
from src.strategy.signals import SignalGenerator, Signal
from src.risk.risk_manager import RiskManager
from src.config import RiskConfig
from src.ui.dashboard import Dashboard


def test_volume_profile_vah_val():
    """Verify Volume Profile computes POC, VAH, and VAL enclosing 70% volume."""
    highs = np.array([60000.0, 60100.0, 60200.0, 60150.0, 60050.0])
    lows = np.array([59800.0, 59900.0, 60000.0, 59950.0, 59850.0])
    closes = np.array([59950.0, 60050.0, 60100.0, 60000.0, 59900.0])
    volumes = np.array([10.0, 50.0, 100.0, 40.0, 15.0])

    poc, vah, val = compute_volume_profile(highs, lows, closes, volumes, num_bins=20, value_area_pct=0.70)
    assert poc > 0.0
    assert val <= poc <= vah
    assert val >= 59800.0
    assert vah <= 60200.0


def test_equal_highs_lows_detection():
    """Verify EQH and EQL detection within tolerance threshold."""
    highs = np.zeros(30)
    lows = np.zeros(30)
    highs[:] = 50000.0
    lows[:] = 49500.0

    # Create two swing highs at 50500.0 and 50505.0 (diff = 0.01% < 0.05%)
    highs[10] = 50500.0
    highs[20] = 50505.0
    swing_highs = [10, 20]

    # Create two swing lows at 49200.0 and 49202.0
    lows[12] = 49200.0
    lows[22] = 49202.0
    swing_lows = [12, 22]

    eqh, eql = detect_equal_highs_lows(highs, lows, swing_highs, swing_lows, tolerance_pct=0.0005)
    assert eqh == pytest.approx(50502.5, rel=1e-4)
    assert eql == pytest.approx(49201.0, rel=1e-4)


def test_order_block_detection():
    """Verify detection of displacement origin candle as institutional Order Block."""
    opens = np.array([50100, 50050, 50020, 50010, 50000, 49800, 50300, 50100, 49950, 50000], dtype=float)
    highs = np.array([50200, 50150, 50100, 50050, 50050, 50450, 50350, 50200, 50050, 50050], dtype=float)
    lows = np.array([50000, 49950, 49900, 49850, 49750, 49780, 50050, 49900, 49850, 49900], dtype=float)
    closes = np.array([50050, 50020, 50010, 50000, 49800, 50400, 50100, 49950, 50000, 50020], dtype=float)

    ob_detected, ob_dir, ob_top, ob_bottom, ob_testing = detect_order_blocks(
        opens, highs, lows, closes, lookback=8, displacement_threshold=0.6
    )
    assert ob_detected is True
    assert ob_dir == "BULLISH"
    assert ob_top == 50050.0
    assert ob_bottom == 49750.0
    assert ob_testing is True


def test_asian_session_range():
    """Verify Asian session (00:00 - 08:00 UTC) range calculation."""
    base_ts = datetime(2026, 9, 21, 2, 0, 0, tzinfo=timezone.utc).timestamp() * 1000.0
    candles = [
        {"open_time": base_ts, "open": 50000, "high": 50800, "low": 49800, "close": 50500, "volume": 100},
        {"open_time": base_ts + 7200000, "open": 50500, "high": 51000, "low": 50200, "close": 50800, "volume": 120},
        {"open_time": base_ts + 14400000, "open": 50800, "high": 50900, "low": 49500, "close": 50000, "volume": 150},
        {"open_time": base_ts + 25200000, "open": 50000, "high": 52000, "low": 49000, "close": 51500, "volume": 200},
    ]
    asian_h, asian_l = calculate_asian_range(candles)
    assert asian_h == 51000.0
    assert asian_l == 49500.0


def test_liquidity_hunt_reversal_signal():
    """Verify LIQUIDITY_HUNT_REVERSAL setup classification and high confidence signal generation."""
    sig_gen = SignalGenerator()
    sa = StructureAnalysis(
        bias_15m="BULLISH",
        bos_5m=False,
        choch_5m=False,
        bos_direction_5m="NONE",
        liquidity_sweep_5m=True,
        displacement_1m=True,
        retest_1m=False,
        is_valid=True,
        invalidation_reason=None,
        setup_type="LIQUIDITY_HUNT_REVERSAL",
        sweep_direction="BULLISH",
        eql=50000.0,
        eqh=51500.0,
        asian_low=50000.0,
        asian_high=51500.0,
        nearest_resistance=51500.0,
    )

    c_1m = [
        {"open": 50100, "high": 50150, "low": 49950, "close": 50120, "volume": 500},
        {"open": 50120, "high": 50150, "low": 49900, "close": 50140, "volume": 800},
    ] * 15

    sig = sig_gen.generate([], [], c_1m, sa)
    assert sig.direction == "LONG"
    assert sig.strength >= 0.94
    assert sig.setup_type == "LIQUIDITY_HUNT_REVERSAL"
    assert sig.structural_target_tp >= 51500.0


def test_order_block_pullback_signal():
    """Verify ORDER_BLOCK_PULLBACK setup classification and signal."""
    sig_gen = SignalGenerator()
    sa = StructureAnalysis(
        bias_15m="BULLISH",
        bos_5m=True,
        choch_5m=False,
        bos_direction_5m="BULLISH",
        liquidity_sweep_5m=False,
        displacement_1m=False,
        retest_1m=True,
        is_valid=True,
        invalidation_reason=None,
        setup_type="ORDER_BLOCK_PULLBACK",
        ob_detected=True,
        ob_direction="BULLISH",
        ob_top=50200.0,
        ob_bottom=49900.0,
        ob_testing=True,
        nearest_resistance=51000.0,
    )
    c_1m = [
        {"open": 50100, "high": 50150, "low": 50050, "close": 50120, "volume": 300},
    ] * 25
    sig = sig_gen.generate([], [], c_1m, sa)
    assert sig.direction == "LONG"
    assert sig.strength >= 0.90
    assert sig.setup_type == "ORDER_BLOCK_PULLBACK"


def test_value_area_breakout_signal():
    """Verify VALUE_AREA_BREAKOUT setup classification and 2.5R+ breakout target."""
    sig_gen = SignalGenerator()
    sa = StructureAnalysis(
        bias_15m="BULLISH",
        bos_5m=True,
        choch_5m=False,
        bos_direction_5m="BULLISH",
        liquidity_sweep_5m=False,
        displacement_1m=True,
        retest_1m=False,
        is_valid=True,
        invalidation_reason=None,
        setup_type="VALUE_AREA_BREAKOUT",
        breakout_direction="BULLISH",
        breakout_level=50500.0,
        vah=50500.0,
        val=49800.0,
        nearest_resistance=52000.0,
    )
    c_1m = [
        {"open": 50400, "high": 50800, "low": 50350, "close": 50750, "volume": 1200},
    ] * 25
    sig = sig_gen.generate([], [], c_1m, sa)
    assert sig.direction == "LONG"
    assert sig.strength >= 0.95
    assert sig.setup_type == "VALUE_AREA_BREAKOUT"
    assert sig.structural_target_tp >= 50750.0 * 1.04


def test_structural_trailing_stop_loss():
    """Verify Structural Swing Trailing Stop ratchets behind confirmed 1m swing points."""
    rm = RiskManager(RiskConfig())
    
    candles_1m = [
        {"high": 50100, "low": 49950},
        {"high": 50200, "low": 50000},
        {"high": 50300, "low": 50150},
        {"high": 50400, "low": 50200},
        {"high": 50300, "low": 50150},
        {"high": 50250, "low": 50100},  # Swing low candidate
        {"high": 50350, "low": 50180},
        {"high": 50450, "low": 50300},
        {"high": 50550, "low": 50400},
        {"high": 50500, "low": 50450},
    ]

    new_sl, updated, msg = rm.calculate_structural_trailing_stop_loss(
        entry_price=50000.0,
        side="LONG",
        current_price=50500.0,
        current_sl=49850.0,
        candles_1m=candles_1m,
        tick_size=0.1,
    )
    assert updated is True
    assert new_sl > 49850.0
    assert new_sl <= 50100.0
    assert "Structural Trailing SL" in msg


def test_dashboard_institutional_display():
    """Verify Dashboard UI correctly renders VAH, VAL, EQH, EQL, Asian H/L and Order Blocks."""
    dashboard = Dashboard(mode="LIVE", target_leverage=25)
    market_watch = {
        "assets": {
            "BTCUSD": {
                "symbol": "BTCUSD",
                "price": 60500.0,
                "delta_price": 60500.0,
                "bias_15m": "BULLISH",
                "bos_5m": True,
                "pdh": 61000.0,
                "pdl": 59000.0,
                "poc": 60000.0,
                "vah": 60800.0,
                "val": 59600.0,
                "eqh": 61200.0,
                "eql": 58800.0,
                "asian_high": 60700.0,
                "asian_low": 59200.0,
                "ob_detected": True,
                "ob_direction": "BULLISH",
                "ob_top": 60200.0,
                "ob_bottom": 59900.0,
                "ob_testing": True,
                "setup_type": "ORDER_BLOCK_PULLBACK",
            }
        }
    }
    dashboard.update(
        account_data={"equity": 10000.0, "available_margin": 8000.0, "daily_pnl": 150.0},
        position_data={},
        signal_data={"btc_price": 60500.0},
        risk_data={"risk_check": "PASS"},
        execution_data={"order_status": "IDLE (SCANNING)", "execution_routing": "HYBRID_OPTIMIZED"},
        market_watch_data=market_watch,
    )
    panel = dashboard._build_market_watch_panel()
    assert panel is not None
    rendered_layout = dashboard.render()
    assert rendered_layout is not None
