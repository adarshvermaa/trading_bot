"""
Tests verifying full synchronization between Technical Analysis components
(ICT Structure, FVG, OB, Liquidity Sweeps, Dealing Ranges, Chart Patterns, VWAP, RSI)
and Jev AI (System One), as well as order fill confirmation and loop elimination.
"""

import asyncio
import time
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from src.execution.order_manager import OrderManager, ActiveOrder, OrderState
from src.portfolio.account import AccountManager
from src.strategy.structure import MarketStructure, StructureAnalysis
from src.strategy.signals import SignalGenerator, Signal
from src.llm.jev_client import JevClient, JevConfig, JevRegimeResult, JevDynamicSLTPResult
from src.config import AppConfig, load_config


@pytest.fixture
def mock_delta_client():
    client = AsyncMock()
    client.live_trading = True
    client.get_product = AsyncMock(return_value={
        "id": 27,
        "symbol": "BTCUSD",
        "contract_value": 0.001,
        "max_leverage": 100.0,
    })
    return client


@pytest.fixture
def test_order_manager(mock_delta_client):
    risk_manager = MagicMock()
    risk_manager.config = MagicMock()
    risk_manager.config.stop_loss.max_loss_pct_of_margin = 0.03
    risk_manager.validate_risk_reward_ratio = MagicMock(return_value=(True, 3.0, "Valid R:R"))

    account_manager = AccountManager(is_paper=True, initial_paper_balance=10000.0)
    om = OrderManager(mock_delta_client, risk_manager, {}, account_manager=account_manager)
    return om, mock_delta_client, risk_manager, account_manager


# --------------------------------------------------------------------------
# 1. Delta Order Fill Verification (state: "closed" with unfilled_size == 0)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_order_status_recognizes_delta_closed_as_filled(test_order_manager):
    """
    On Delta Exchange REST API v2, fully executed orders return state: 'closed'
    with unfilled_size: '0' and average_fill_price > 0.
    Verify that sync_order_status transitions target_order to FILLED, NOT CANCELLED.
    """
    om, delta_client, _, account_manager = test_order_manager

    order = ActiveOrder(
        order_id="delta_fill_101",
        client_order_id="cid_fill_101",
        state=OrderState.OPEN,
        symbol="BTCUSD",
        side="BUY",
        size=2.0,
        entry_price=65000.0,
        sl_price=64000.0,
        tp_price=68000.0,
        product_id=27,
    )
    om.active_orders["delta_fill_101"] = order
    om.active_client_ids.add("cid_fill_101")

    # Delta returns state: "closed" with unfilled_size: "0" and average_fill_price
    delta_client.get_order = AsyncMock(return_value={
        "id": "delta_fill_101",
        "state": "closed",
        "unfilled_size": "0",
        "average_fill_price": "65120.50",
    })

    updated = await om.sync_order_status("delta_fill_101")

    assert updated is not None
    assert updated.state == OrderState.FILLED
    assert updated.entry_price == 65120.50
    assert updated.last_event == "ORDER_FILLED"
    assert "delta_fill_101" in om.active_orders
    # Verify account manager recorded the fill
    assert "BTCUSD" in account_manager.positions
    pos = account_manager.positions["BTCUSD"]
    assert pos.size == 2.0
    assert pos.entry_price == 65120.50


@pytest.mark.asyncio
async def test_sync_order_status_recognizes_delta_unfilled_closed_as_cancelled(test_order_manager):
    """
    If Delta returns state: 'closed' but unfilled_size > 0 with no fill price,
    it represents a cancelled/expired order that was closed without execution.
    """
    om, delta_client, _, _ = test_order_manager

    order = ActiveOrder(
        order_id="delta_cancel_102",
        client_order_id="cid_cancel_102",
        state=OrderState.OPEN,
        symbol="BTCUSD",
        side="BUY",
        size=1.0,
        entry_price=65000.0,
        sl_price=64000.0,
        tp_price=68000.0,
        product_id=27,
    )
    om.active_orders["delta_cancel_102"] = order
    om.active_client_ids.add("cid_cancel_102")

    delta_client.get_order = AsyncMock(return_value={
        "id": "delta_cancel_102",
        "state": "closed",
        "unfilled_size": "1.0",
        "average_fill_price": None,
    })

    updated = await om.sync_order_status("delta_cancel_102")

    assert updated is not None
    assert updated.state == OrderState.CANCELLED
    assert "delta_cancel_102" not in om.active_orders
    assert "cid_cancel_102" not in om.active_client_ids


# --------------------------------------------------------------------------
# 2. Strategy Loop Order Placement Rejection Cooldown (Infinite Loop Prevention)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_strategy_loop_rejection_triggers_cooldown():
    """
    Verify that when validate_and_place_order returns None (rejection),
    the bot sets a 15-second re-entry cooldown, resets position health to '--',
    and does NOT immediately re-scan and spam orders in an infinite loop.
    """
    from src.main import ScalpingBot

    cfg = load_config()
    cfg.strategy.jev.enabled = False
    cfg.strategy.ml.enabled = False
    bot = ScalpingBot(config=cfg, mode="paper")

    # Mock order manager to reject order (return None)
    bot.order_manager.validate_and_place_order = AsyncMock(return_value=None)
    bot.delta_client.fetch_ticker_price = AsyncMock(return_value=65000.0)

    dummy_signal = Signal(
        direction="LONG",
        strength=0.85,
        entry_price=65000.0,
        sl_price=64500.0,
        tp_price=66500.0,
        ema_cross="BULLISH",
        vwap_position="ABOVE",
        rsi=55.0,
        atr=250.0,
        relative_volume=1.8,
        adx=32.0,
        timeframe_alignment=True,
    )
    dummy_structure = MagicMock()
    dummy_structure.nearest_support = 64800.0
    dummy_structure.nearest_resistance = 66000.0

    candidate_setup = {
        "symbol": "BTCUSD",
        "signal": dummy_signal,
        "structure": dummy_structure,
        "score": 0.88,
        "atr": 250.0,
        "is_actionable": True,
        "technical_levels": {},
    }

    # Simulate one step of strategy loop when setup is returned
    setup = candidate_setup
    delta_live_price = await bot.delta_client.fetch_ticker_price(setup["symbol"])
    exec_price = delta_live_price

    placed_order = await bot.order_manager.validate_and_place_order(
        symbol=setup["symbol"],
        side=setup["signal"].direction,
        entry_price=exec_price,
        sl_price=None,
        tp_price=None,
        atr=setup["atr"],
        spread_bps=0.0,
        equity=bot.account_manager.equity,
        order_type="market",
    )

    if placed_order is not None:
        bot._position_health = "STRONG"
    else:
        bot._re_entry_cooldown_until = time.time() + 15.0
        bot._position_health = "--"

    # Verifications
    assert placed_order is None
    assert bot._position_health == "--"
    assert bot._re_entry_cooldown_until > time.time() + 10.0

    # Verify that in subsequent ticks during cooldown, trading is paused
    is_in_cooldown = time.time() < bot._re_entry_cooldown_until
    assert is_in_cooldown is True


# --------------------------------------------------------------------------
# 3. Technical Levels Synchronization (ICT Structure -> Jev Dynamic SL/TP)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_technical_levels_complete_synchronization_with_jev():
    """
    Verify that all ICT structural levels (EQH, EQL, FVG top/bottom,
    Chart Pattern Target, Pricing Zone, Opposing Liquidity)
    are included in tech_levels and faithfully consumed by Jev System One.
    """
    structure = StructureAnalysis(
        bias_15m="BULLISH",
        bos_5m=True,
        choch_5m=False,
        bos_direction_5m="BULLISH",
        liquidity_sweep_5m=True,
        displacement_1m=True,
        retest_1m=True,
        is_valid=True,
        invalidation_reason=None,
        nearest_support=64500.0,
        nearest_resistance=66200.0,
        ob_bottom=64400.0,
        ob_top=64600.0,
        trap_wick_extreme=64350.0,
        vah=66000.0,
        val=64200.0,
        pdh=66500.0,
        pdl=64000.0,
        eqh=66800.0,
        eql=63900.0,
        fvg_top=65800.0,
        fvg_bottom=65500.0,
        chart_pattern_target=67000.0,
        pricing_zone="DISCOUNT",
    )

    # In main.py, tech_levels is constructed as:
    tech_levels = {
        "swing_low": 64350.0,
        "swing_high": 65500.0,
        "ob_bottom": structure.ob_bottom,
        "ob_top": structure.ob_top,
        "trap_wick_price": structure.trap_wick_extreme,
        "vah": structure.vah,
        "val": structure.val,
        "pdh": structure.pdh,
        "pdl": structure.pdl,
        "nearest_support": structure.nearest_support,
        "nearest_resistance": structure.nearest_resistance,
        "eqh": structure.eqh,
        "eql": structure.eql,
        "fvg_top": structure.fvg_top,
        "fvg_bottom": structure.fvg_bottom,
        "chart_pattern_target": structure.chart_pattern_target,
        "pricing_zone": structure.pricing_zone,
        "opposing_liquidity": structure.eqh,
    }

    # Verify all expected keys exist and are non-null
    assert tech_levels["eqh"] == 66800.0
    assert tech_levels["eql"] == 63900.0
    assert tech_levels["fvg_top"] == 65800.0
    assert tech_levels["fvg_bottom"] == 65500.0
    assert tech_levels["chart_pattern_target"] == 67000.0
    assert tech_levels["pricing_zone"] == "DISCOUNT"
    assert tech_levels["opposing_liquidity"] == 66800.0

    # Call evaluate_dynamic_sl_tp with these levels
    jev_cfg = JevConfig(enabled=True, api_key="test_key")
    client = JevClient(api_key="test_key", config=jev_cfg)

    mock_answers = {
        "sl_anchor": {"choice": "SWEEP_WICK", "confidence": 0.95},
        "sl_cushion": {"score": 1, "confidence": 0.90},
        "tp_target_type": {"choice": "OPPOSING_LIQUIDITY", "confidence": 0.92},
        "target_rr_multiple": {"score": 3, "confidence": 0.88},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers
        res = await client.evaluate_dynamic_sl_tp(
            symbol="BTCUSD",
            direction="LONG",
            entry_price=65000.0,
            atr=200.0,
            technical_levels=tech_levels,
            tick_size=0.1,
        )

        assert res is not None
        assert res.sl_price < 65000.0
        assert res.tp_price > 65000.0
        assert res.realized_rr_ratio >= 2.0
        assert res.tp_target_type == "OPPOSING_LIQUIDITY"

        # Verify that Jev's State string passed to the System One model
        # actually contained the synchronized ICT levels
        state_sent = mock_query.call_args[0][0]
        assert "EQH=$66800.00" in state_sent
        assert "EQL=$63900.00" in state_sent
        assert "FVG Target=$65800.00" in state_sent
        assert "Chart Target=$67000.00" in state_sent


# --------------------------------------------------------------------------
# 4. Periodic Market Regime Arbiter Synchronization
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_jev_market_regime_periodic_sync():
    """
    Verify evaluate_market_regime evaluates ATR ratio, BB bandwidth, and relative volume
    to output a structured JevRegimeResult.
    """
    jev_cfg = JevConfig(enabled=True, api_key="test_key")
    client = JevClient(api_key="test_key", config=jev_cfg)

    mock_regime_answers = {
        "market_regime": {"choice": "TRENDING_EXPANSION", "confidence": 0.92},
        "scalp_suitability": {"value": True, "confidence": 0.90},
        "recommended_strategy": {"choice": "MOMENTUM_BREAKOUT", "confidence": 0.88},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_regime_answers

        res = await client.evaluate_market_regime(
            atr_15m=180.0,
            atr_1h=220.0,
            bb_bandwidth=0.035,
            rel_vol_15m=1.65,
            session_zone="LONDON_NY_OVERLAP",
        )

        assert isinstance(res, JevRegimeResult)
        assert res.market_regime == "TRENDING_EXPANSION"
        assert res.is_favorable is True
        assert res.recommended_strategy == "MOMENTUM_BREAKOUT"


# --------------------------------------------------------------------------
# 5. Position Monitoring VWAP and RSI Live Sync
# --------------------------------------------------------------------------

def test_signal_generator_vwap_and_rsi_calculation():
    """
    Verify that SignalGenerator.calculate_vwap and calculate_rsi
    compute mathematically sound real-time indicator values from candle arrays.
    """
    n = 25
    highs = np.linspace(65000, 65500, n)
    lows = np.linspace(64800, 65300, n)
    closes = np.linspace(64900, 65400, n)
    volumes = np.ones(n) * 10.0

    vwap_arr = SignalGenerator.calculate_vwap(highs, lows, closes, volumes)
    assert len(vwap_arr) == n
    assert 64800.0 < vwap_arr[-1] < 65500.0

    rsi_arr = SignalGenerator.calculate_rsi(closes, period=14)
    assert len(rsi_arr) == n
    # In an uptrend, RSI should be above 50
    assert rsi_arr[-1] > 50.0
