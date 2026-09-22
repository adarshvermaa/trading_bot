"""Unit test suite for Dynamic Leverage Engine based on Market Win-Win & Jev AI System One confidence."""

import pytest
from dataclasses import dataclass
from src.config import DynamicLeverageConfig, DynamicLeverageTiersConfig
from src.risk.dynamic_leverage import DynamicLeverageEngine, DynamicLeverageResult
from src.risk.risk_manager import RiskManager
from src.llm.jev_client import JevClient, JevConfig


@dataclass
class DummyStructure:
    bias_15m: str = "BULLISH"
    bos_5m: bool = True
    bos_direction_5m: str = "BULLISH"
    choch_5m: bool = False
    pricing_zone: str = "DISCOUNT"
    playbook: str = "ICT_JUDAS_SWEEP"
    chart_pattern: str = "DOUBLE_BOTTOM"


@dataclass
class DummySignal:
    direction: str = "LONG"
    relative_volume: float = 2.2
    adx: float = 30.0
    entry_price: float = 65000.0
    sl_price: float = 64500.0  # 0.77% distance


def test_win_win_apex_dynamic_leverage():
    """Verify that an A+ institutional confluence setup achieves WIN_WIN_APEX tier (50x)."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure()
    sig = DummySignal()

    res = engine.evaluate_leverage(
        symbol="BTCUSD",
        direction="LONG",
        entry_price=sig.entry_price,
        sl_price=sig.sl_price,
        structure=struct,
        signal=sig,
        jev_confidence=0.92,
        atr=200.0,
        avg_atr=200.0,
        daily_pnl=0.0,
        exchange_max_leverage=100.0,
        obi=0.25,
    )

    assert res.tier == "WIN_WIN_APEX"
    assert res.effective_confidence >= 0.85
    assert res.leverage == 50
    assert res.volatility_dampener == 1.0
    assert res.drawdown_dampener == 1.0


def test_high_conviction_tier():
    """Verify high conviction setup achieves HIGH_CONVICTION tier (30x)."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure(
        bias_15m="BULLISH",
        bos_5m=True,
        pricing_zone="DISCOUNT",
        playbook="ORDER_BLOCK_FVG_PULLBACK",
        chart_pattern="NONE",
    )
    sig = DummySignal(relative_volume=1.3, adx=22.0)

    res = engine.evaluate_leverage(
        symbol="ETHUSD",
        direction="LONG",
        entry_price=2700.0,
        sl_price=2675.0,  # ~0.92% distance
        structure=struct,
        signal=sig,
        jev_confidence=0.78,
        atr=25.0,
        avg_atr=25.0,
        daily_pnl=0.0,
        exchange_max_leverage=100.0,
    )

    assert res.tier == "HIGH_CONVICTION"
    assert 0.75 <= res.effective_confidence < 0.85
    assert res.leverage == 30


def test_standard_scalp_tier():
    """Verify baseline setup achieves STANDARD_SCALP tier (20x)."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure(
        bias_15m="BULLISH",
        bos_5m=True,
        pricing_zone="EQUILIBRIUM",
        playbook="NONE",
        chart_pattern="NONE",
    )
    sig = DummySignal(relative_volume=1.0, adx=18.0)

    res = engine.evaluate_leverage(
        symbol="SOLUSD",
        direction="LONG",
        entry_price=180.0,
        sl_price=177.0,  # 1.66% distance
        structure=struct,
        signal=sig,
        jev_confidence=0.70,
        atr=3.0,
        avg_atr=3.0,
        daily_pnl=0.0,
        exchange_max_leverage=100.0,
    )

    assert res.tier == "STANDARD_SCALP"
    assert 0.60 <= res.effective_confidence < 0.75
    assert res.leverage == 20


def test_defensive_probe_tier():
    """Verify low confluence setup achieves DEFENSIVE_PROBE tier (10x)."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure(
        bias_15m="BEARISH",  # counter trend!
        bos_5m=False,
        pricing_zone="PREMIUM",  # buying in premium
        playbook="NONE",
        chart_pattern="NONE",
    )
    sig = DummySignal(relative_volume=0.8, adx=12.0)

    res = engine.evaluate_leverage(
        symbol="BTCUSD",
        direction="LONG",
        entry_price=65000.0,
        sl_price=64000.0,
        structure=struct,
        signal=sig,
        jev_confidence=0.45,
        atr=200.0,
        avg_atr=200.0,
        daily_pnl=0.0,
        exchange_max_leverage=100.0,
    )

    assert res.tier == "DEFENSIVE_PROBE"
    assert res.effective_confidence < 0.65
    assert res.leverage == 10


def test_volatility_spike_dampener():
    """Verify that extreme volatility (ATR spike) throttles leverage."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure()
    sig = DummySignal()

    # Normal volatility baseline
    res_normal = engine.evaluate_leverage(
        symbol="BTCUSD",
        direction="LONG",
        entry_price=sig.entry_price,
        sl_price=sig.sl_price,
        structure=struct,
        signal=sig,
        jev_confidence=0.90,
        atr=200.0,
        avg_atr=200.0,
    )

    # Spike volatility (ATR = 2.5x baseline)
    res_spike = engine.evaluate_leverage(
        symbol="BTCUSD",
        direction="LONG",
        entry_price=sig.entry_price,
        sl_price=sig.sl_price,
        structure=struct,
        signal=sig,
        jev_confidence=0.90,
        atr=500.0,
        avg_atr=200.0,
    )

    assert res_spike.volatility_dampener < 1.0
    assert res_spike.effective_confidence < res_normal.effective_confidence
    assert res_spike.leverage < res_normal.leverage


def test_drawdown_dampener():
    """Verify that daily loss throttles leverage to protect remaining capital."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure()
    sig = DummySignal()

    res = engine.evaluate_leverage(
        symbol="BTCUSD",
        direction="LONG",
        entry_price=sig.entry_price,
        sl_price=sig.sl_price,
        structure=struct,
        signal=sig,
        jev_confidence=0.88,
        daily_pnl=-150.0,  # 50% of max daily loss ($300)
        max_daily_loss=300.0,
    )

    assert res.drawdown_dampener == 0.5
    assert res.effective_confidence < 0.85
    assert res.leverage <= 30


def test_liquidation_buffer_constraint():
    """Verify that a wide Stop Loss distance forces leverage lower to preserve liquidation buffer."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure()
    sig = DummySignal(
        entry_price=100.0,
        sl_price=97.0,  # 3.0% Stop Loss!
    )

    # With 3.0% SL and buffer ratio 1.5, max safe leverage = int(1.0 / (0.03 * 1.5)) = 22x
    res = engine.evaluate_leverage(
        symbol="SOLUSD",
        direction="LONG",
        entry_price=100.0,
        sl_price=97.0,
        structure=struct,
        signal=sig,
        jev_confidence=0.95,  # Apex confidence would nominally give 50x!
        exchange_max_leverage=100.0,
    )

    assert res.tier == "WIN_WIN_APEX"
    assert res.max_safe_leverage == 22
    assert res.leverage <= 22  # Clamped strictly by liquidation safety!


def test_asset_exchange_max_leverage_enforcement():
    """Verify that ZECUSD strictly caps leverage at Delta's 20x ceiling even for Win-Win Apex."""
    engine = DynamicLeverageEngine()
    struct = DummyStructure()
    sig = DummySignal(entry_price=40.0, sl_price=39.8)

    res = engine.evaluate_leverage(
        symbol="ZECUSD",
        direction="LONG",
        entry_price=40.0,
        sl_price=39.8,
        structure=struct,
        signal=sig,
        jev_confidence=0.95,
        exchange_max_leverage=20.0,  # Delta strict limit!
    )

    assert res.leverage <= 20


@pytest.mark.asyncio
async def test_jev_client_evaluate_dynamic_leverage_offline():
    """Verify JevClient evaluate_dynamic_leverage fallback when disabled/offline."""
    config = JevConfig(enabled=False)
    client = JevClient(api_key="", config=config)

    res_a_plus = await client.evaluate_dynamic_leverage(
        symbol="BTCUSD",
        setup_grade=3.8,
        dir_conf=0.90,
    )
    assert res_a_plus.recommended_tier == "WIN_WIN_APEX"
    assert res_a_plus.ai_confidence >= 0.85
    assert res_a_plus.leverage_multiplier == 1.0

    res_probe = await client.evaluate_dynamic_leverage(
        symbol="ETHUSD",
        setup_grade=1.8,
        dir_conf=0.60,
    )
    assert res_probe.recommended_tier == "DEFENSIVE_PROBE"
    assert res_probe.leverage_multiplier == 0.25


def test_risk_manager_validate_leverage_with_dynamic_result():
    """Verify RiskManager.validate_leverage accepts DynamicLeverageResult and returns valid tuple."""
    from src.config import load_config
    cfg = load_config()
    rm = RiskManager(cfg.risk)

    dyn_res = DynamicLeverageResult(
        leverage=45,
        tier="WIN_WIN_APEX",
        effective_confidence=0.90,
        market_confidence=0.92,
        ai_confidence=0.88,
        volatility_dampener=1.0,
        drawdown_dampener=1.0,
        liquidation_buffer_ratio=1.5,
        max_safe_leverage=50,
        rationale="Win-win test",
    )

    ok, lev, reason = rm.validate_leverage(dyn_res, exchange_max_leverage=100.0)
    assert ok is True
    assert lev == 45
    assert reason == "VALID"

    # Test when exchange max leverage is lower than dynamic result (e.g. ZEC 20x)
    ok_zec, lev_zec, reason_zec = rm.validate_leverage(dyn_res, exchange_max_leverage=20.0)
    assert ok_zec is True
    assert lev_zec == 20


def test_dashboard_displays_dynamic_leverage():
    """Verify Dashboard renders dynamic leverage in header, market watch, and signal panels."""
    from src.ui.dashboard import Dashboard
    dashboard = Dashboard(mode="paper", target_leverage=25)

    dashboard.update(
        account_data={"equity": 10000.0},
        position_data={},
        signal_data={
            "dynamic_leverage": "45x [WIN_WIN_APEX]",
            "signal_score": "ACTIONABLE (0.92)",
        },
        risk_data={},
        execution_data={},
        market_watch_data={"assets": {}},
    )

    hdr = dashboard._build_header()
    assert hdr is not None
    sig_panel = dashboard._build_signal_panel()
    assert sig_panel is not None
    mw_panel = dashboard._build_market_watch_panel()
    assert mw_panel is not None

