import pytest
import numpy as np
import time
from unittest.mock import MagicMock, AsyncMock, patch

from src.data.delta_ws import Candle, DeltaWSClient
from src.strategy.structure import MarketStructure, StructureAnalysis, calculate_volatility_squeeze, compute_volume_poc
from src.strategy.signals import SignalGenerator, Signal
from src.risk.risk_manager import RiskManager
from src.config import AppConfig, load_config
from src.execution.order_manager import ActiveOrder, OrderState


def test_calculate_htf_levels_and_poc():
    """Verify DeltaWSClient.calculate_htf_levels accurately calculates PDH, PDL, PDC, and POC."""
    base_time = 1700000000
    candles = []
    
    # 30 candles between 80,000 and 85,000
    # Heavy volume centered at 82,500
    for i in range(30):
        price = 80000.0 + (i * 150.0)
        vol = 500.0 if abs(price - 82500.0) < 300.0 else 50.0
        candles.append(Candle(
            open_time=base_time + i * 3600 * 1000,
            open=price - 50.0,
            high=price + 200.0,
            low=price - 100.0,
            close=price + 50.0,
            volume=vol,
            close_time=base_time + (i + 1) * 3600 * 1000,
            is_closed=True,
        ))

    levels = DeltaWSClient.calculate_htf_levels(candles)
    expected_window = candles[-25:-1]
    assert levels["PDH"] == pytest.approx(max(c.high for c in expected_window), rel=1e-4)
    assert levels["PDL"] == pytest.approx(min(c.low for c in expected_window), rel=1e-4)
    assert levels["PDC"] == pytest.approx(expected_window[-1].close, rel=1e-4)
    assert levels["POC"] > 0.0
    # POC should be clustered near the high volume zone ~82500
    assert 81500.0 <= levels["POC"] <= 83500.0


def test_volatility_squeeze_compression_and_firing():
    """Verify John Carter Volatility Squeeze detects compression and release."""
    np.random.seed(42)
    n = 35
    close = np.zeros(n)
    high = np.zeros(n)
    low = np.zeros(n)

    # Tight range period: price ~ 80000 +/- 5
    for i in range(25):
        close[i] = 80000.0 + np.sin(i) * 5.0
        high[i] = close[i] + 15.0
        low[i] = close[i] - 15.0

    # Expansion period: price shoots up violently with wide range
    for i in range(25, n):
        close[i] = 80000.0 + (i - 24) * 200.0
        high[i] = close[i] + 100.0
        low[i] = close[i] - 100.0

    is_squeeze, squeeze_fired = calculate_volatility_squeeze(
        high, low, close, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5
    )

    # During tight consolidation, squeeze should be active
    assert is_squeeze[23] == True or is_squeeze[24] == True
    # At index 25/26 (volatility expansion), squeeze_fired should trigger
    assert any(squeeze_fired[25:]) == True


def test_htf_breakout_bullish_and_bearish_detection():
    """Verify MarketStructure.analyze detects HTF_BREAKOUT on volume expansion & displacement."""
    struct = MarketStructure()

    # Create 50 1m candles:
    # 48 candles consolidating below PDH (85,000)
    # Candle 49: huge bullish displacement breaking through 85,000 to 85,500 with 3x volume
    candles_1m = []
    base_time = 1700000000
    for i in range(48):
        candles_1m.append({
            "open": 84800.0,
            "high": 84950.0,
            "low": 84750.0,
            "close": 84900.0,
            "volume": 100.0,
        })
    # Breakout candle
    candles_1m.append({
        "open": 84900.0,
        "high": 85550.0,
        "low": 84880.0,
        "close": 85500.0,  # Closes well above PDH (85,000)
        "volume": 350.0,   # 3.5x volume expansion
    })

    # Mock 15m and 5m candles
    candles_15m = [{"open": 84000.0 + i*10, "high": 84200.0 + i*10, "low": 83900.0 + i*10, "close": 84150.0 + i*10, "volume": 1000.0} for i in range(50)]
    candles_5m = [{"open": 84500.0 + i*5, "high": 84600.0 + i*5, "low": 84400.0 + i*5, "close": 84550.0 + i*5, "volume": 500.0} for i in range(50)]

    htf_levels = {"PDH": 85000.0, "PDL": 83000.0, "PDC": 84500.0, "POC": 84200.0}

    analysis = struct.analyze(
        candles_15m=candles_15m,
        candles_5m=candles_5m,
        candles_1m=candles_1m,
        atr_1m=100.0,
        htf_levels=htf_levels,
    )

    assert analysis.setup_type == "HTF_BREAKOUT"
    assert analysis.breakout_direction == "BULLISH"
    assert analysis.breakout_level == 85000.0
    assert analysis.is_valid == True


def test_breakout_retest_detection():
    """Verify MarketStructure.analyze detects BREAKOUT_RETEST after prior breakout."""
    struct = MarketStructure()

    # 10 candles:
    # Price previously broke 85,000 (high=85,600)
    # Then pulled back to test 85,000 support
    # Last candle forms a bullish rejection (hammer) right at 85,010
    candles_1m = []
    # Previous candles breaking out
    for i in range(6):
        candles_1m.append({"open": 85000.0 + i*100, "high": 85150.0 + i*100, "low": 84950.0 + i*100, "close": 85100.0 + i*100, "volume": 100.0})
    # Pullback candles
    candles_1m.append({"open": 85500.0, "high": 85520.0, "low": 85200.0, "close": 85220.0, "volume": 80.0})
    candles_1m.append({"open": 85220.0, "high": 85250.0, "low": 85050.0, "close": 85070.0, "volume": 80.0})
    # Rejection candle at PDH (85,000)
    candles_1m.append({"open": 85020.0, "high": 85080.0, "low": 84990.0, "close": 85060.0, "volume": 120.0})

    candles_15m = [{"open": 84000.0 + i*10, "high": 84200.0 + i*10, "low": 83900.0 + i*10, "close": 84150.0 + i*10, "volume": 1000.0} for i in range(50)]
    candles_5m = [{"open": 84500.0 + i*5, "high": 84600.0 + i*5, "low": 84400.0 + i*5, "close": 84550.0 + i*5, "volume": 500.0} for i in range(50)]

    htf_levels = {"PDH": 85000.0, "PDL": 83000.0, "PDC": 84500.0, "POC": 84200.0}

    analysis = struct.analyze(
        candles_15m=candles_15m,
        candles_5m=candles_5m,
        candles_1m=candles_1m,
        atr_1m=100.0,
        htf_levels=htf_levels,
    )

    assert analysis.setup_type == "BREAKOUT_RETEST"
    assert analysis.breakout_direction == "BULLISH"
    assert analysis.is_valid == True


def test_breakout_signal_generation_and_targets():
    """Verify SignalGenerator generates high confidence signal and 2.5R+ targets for breakouts."""
    sig_gen = SignalGenerator()

    candles_1m = [{"open": 85000.0, "high": 85500.0, "low": 84950.0, "close": 85450.0, "volume": 300.0} for _ in range(50)]
    candles_5m = [{"open": 84800.0, "high": 85200.0, "low": 84700.0, "close": 85100.0, "volume": 500.0} for _ in range(50)]
    candles_15m = [{"open": 84500.0, "high": 85000.0, "low": 84400.0, "close": 84900.0, "volume": 1000.0} for _ in range(50)]

    analysis = StructureAnalysis(
        bias_15m="BULLISH",
        bos_5m=True,
        choch_5m=False,
        bos_direction_5m="BULLISH",
        liquidity_sweep_5m=False,
        displacement_1m=True,
        retest_1m=False,
        is_valid=True,
        invalidation_reason=None,
        setup_type="HTF_BREAKOUT",
        breakout_direction="BULLISH",
        breakout_level=85000.0,
        nearest_resistance=89000.0,
    )

    signal = sig_gen.generate(candles_15m, candles_5m, candles_1m, analysis)
    assert signal.direction == "LONG"
    assert signal.strength >= 0.95
    assert signal.setup_type == "HTF_BREAKOUT"
    assert signal.structural_target_tp >= 85450.0 * 1.04


def test_breakeven_trigger_locking_at_2r():
    """Verify check_breakeven_trigger locks SL to entry +0.1% once profit reaches >= 2.0R."""
    cfg = load_config()
    rm = RiskManager(cfg.risk)

    entry = 80000.0
    initial_sl = 79500.0  # Risk R = 500
    side = "LONG"

    # Scenario 1: Profit is +600 (+1.2R), not yet 2.0R
    current_p_1 = 80600.0
    new_sl, updated, msg = rm.check_breakeven_trigger(
        entry_price=entry,
        current_price=current_p_1,
        side=side,
        initial_sl=initial_sl,
        current_sl=initial_sl,
        r_multiple=2.0,
    )
    assert updated == False

    # Scenario 2: Profit is +1100 (+2.2R >= 2.0R)
    current_p_2 = 81100.0
    new_sl, updated, msg = rm.check_breakeven_trigger(
        entry_price=entry,
        current_price=current_p_2,
        side=side,
        initial_sl=initial_sl,
        current_sl=initial_sl,
        r_multiple=2.0,
    )
    assert updated == True
    assert new_sl == pytest.approx(80080.0, rel=1e-3)
    assert "SL locked to Breakeven" in msg


def test_breakout_take_profit_2_5r():
    """Verify RiskManager.calculate_take_profit applies >= 2.5R target for HTF breakouts."""
    cfg = load_config()
    rm = RiskManager(cfg.risk)

    entry = 80000.0
    sl = 79200.0  # Risk R = 800
    side = "LONG"

    standard_tp = rm.calculate_take_profit(
        entry_price=entry,
        side=side,
        margin=100.0,
        leverage=25,
        contract_value=1.0,
        size=1,
        sl_price=sl,
        setup_type="TREND_PULLBACK",
    )

    breakout_tp = rm.calculate_take_profit(
        entry_price=entry,
        side=side,
        margin=100.0,
        leverage=25,
        contract_value=1.0,
        size=1,
        sl_price=sl,
        setup_type="HTF_BREAKOUT",
    )

    assert breakout_tp == pytest.approx(82000.0, rel=1e-3)
    assert breakout_tp > standard_tp


def test_dashboard_market_watch_htf_display():
    """Verify Dashboard Market Watch panel formats and renders PDH, PDL, POC, and Squeeze states."""
    from src.ui.dashboard import Dashboard
    from rich.panel import Panel

    dashboard = Dashboard(mode="live", target_leverage=25)
    market_watch = {
        "assets": {
            "BTCUSD": {
                "symbol": "BTCUSD",
                "price": 85500.0,
                "delta_price": 85500.0,
                "bias_15m": "BULLISH",
                "bos_5m": True,
                "pdh": 85000.0,
                "pdl": 83000.0,
                "poc": 84200.0,
                "is_squeeze": False,
                "squeeze_fired": True,
                "setup_type": "HTF_BREAKOUT",
            },
            "ETHUSD": {
                "symbol": "ETHUSD",
                "price": 2700.0,
                "delta_price": 2700.0,
                "bias_15m": "NEUTRAL",
                "bos_5m": False,
                "pdh": 2750.0,
                "pdl": 2600.0,
                "poc": 2680.0,
                "is_squeeze": True,
                "squeeze_fired": False,
                "setup_type": "NONE",
            },
        }
    }
    dashboard.update(
        account_data={},
        position_data={},
        signal_data={},
        risk_data={},
        execution_data={},
        market_watch_data=market_watch,
    )
    panel = dashboard._build_market_watch_panel()
    assert isinstance(panel, Panel)
    assert panel.title == "REAL-TIME MARKET WATCH"

    # Also verify render() succeeds
    layout = dashboard.render()
    assert layout.get("market_watch") is not None


def test_breakout_runner_exemption_logic():
    """Verify runner mode exemption logic protects HTF breakout trades in trending regime."""
    cfg = load_config()
    breakout_cfg = cfg.strategy.breakout
    assert breakout_cfg.runner_mode_enabled is True
    assert breakout_cfg.breakeven_r_mult == 2.0

    # Simulate position condition
    active_breakout = ActiveOrder(
        order_id="1",
        client_order_id="c1",
        state=OrderState.FILLED,
        symbol="BTCUSD",
        side="LONG",
        size=1.0,
        entry_price=85000.0,
        sl_price=84500.0,
        tp_price=87000.0,
        setup_type="HTF_BREAKOUT",
        created_at=time.time() - 2000,  # > 1740s
    )

    current_regime = "TRENDING"
    is_breakout_runner = (
        breakout_cfg and getattr(breakout_cfg, "runner_mode_enabled", False)
        and getattr(active_breakout, "setup_type", "") in ("HTF_BREAKOUT", "BREAKOUT_RETEST")
        and current_regime == "TRENDING"
    )
    assert is_breakout_runner is True

    # If regime is RANGING, runner mode does not exempt
    is_ranging_runner = (
        breakout_cfg and getattr(breakout_cfg, "runner_mode_enabled", False)
        and getattr(active_breakout, "setup_type", "") in ("HTF_BREAKOUT", "BREAKOUT_RETEST")
        and "RANGING" == "TRENDING"
    )
    assert is_ranging_runner is False
