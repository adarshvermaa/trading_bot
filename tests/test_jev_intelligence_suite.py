"""
Unit tests for the Jev AI Decision Intelligence Suite (6 Pillars).

Covers:
1. Pre-Scan Market Regime Arbiter (`evaluate_market_regime`)
2. Smart Execution Routing - Fee Optimizer (`evaluate_execution_routing`)
3. Cross-Asset ICT SMT Divergence Arbiter (`evaluate_cross_asset_smt`)
4. Dynamic Conviction Sizing (`evaluate_conviction_sizing`)
5. Pre-Emptive Scratch Exit (`evaluate_scratch_exit`)
6. Post-Trade Forensic Reflexion & Memory Tagging (`evaluate_post_trade_forensics`)
7. End-to-End integration across RegimeFilter, RiskManager, OrderManager, and PatternMemoryStore.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import RiskConfig, StopLossConfig, TakeProfitConfig, TrailingStopConfig, JevConfig, CapitalConfig
from src.llm.jev_client import (
    JevClient,
    JevRegimeResult,
    JevRoutingResult,
    JevSMTResult,
    JevSizingResult,
    JevScratchExitResult,
    JevForensicsResult,
)
from src.strategy.regime import RegimeFilter, RegimeState
from src.risk.risk_manager import RiskManager
from src.execution.order_manager import OrderManager, ActiveOrder, OrderState
from src.ml.pattern_memory import PatternMemoryStore, MarketPatternFingerprint


@pytest.fixture
def risk_cfg():
    return RiskConfig(
        capital=CapitalConfig(account_balance=10000.0, max_allocation_pct=0.10),
        stop_loss=StopLossConfig(max_loss_pct_of_margin=0.03),
        take_profit=TakeProfitConfig(target_pct_of_margin=0.06),
        trailing_stop=TrailingStopConfig(activation_pct_of_margin=0.02),
    )


@pytest.fixture
def jev_cfg_enabled():
    return JevConfig(
        enabled=True,
        enable_dynamic_sl_tp=True,
        enable_dynamic_trailing=True,
        timeout_ms=1200,
    )


@pytest.fixture
def jev_cfg_disabled():
    return JevConfig(
        enabled=False,
    )


# ---------------------------------------------------------------------------
# Pillar 1: Market Regime Arbiter Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_market_regime_offline_fallback(jev_cfg_disabled):
    client = JevClient(api_key="mock", config=jev_cfg_disabled)
    res = await client.evaluate_market_regime(
        atr_15m=120.0,
        atr_1h=150.0,
        bb_bandwidth=0.03,
        rel_vol_15m=1.2,
    )
    assert isinstance(res, JevRegimeResult)
    assert res.market_regime == "TRENDING_EXPANSION"
    assert res.is_favorable is True


@pytest.mark.asyncio
async def test_evaluate_market_regime_online_mock(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    fake_answers = {
        "market_regime": {"choice": "DEAD_CHOP", "confidence": 0.95},
        "scalp_suitability": {"noul": 0.15},
        "recommended_strategy": {"choice": "SIT_ON_HANDS"},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        res = await client.evaluate_market_regime(
            atr_15m=20.0,
            atr_1h=80.0,
            bb_bandwidth=0.005,
            rel_vol_15m=0.3,
        )

    assert res.market_regime == "DEAD_CHOP"
    assert res.is_favorable is False
    assert res.recommended_strategy == "SIT_ON_HANDS"


@pytest.mark.asyncio
async def test_regime_filter_integration_with_jev(jev_cfg_enabled):
    filter = RegimeFilter()
    client = JevClient(api_key="mock", config=jev_cfg_enabled)

    fake_answers = {
        "market_regime": {"choice": "DEAD_CHOP", "confidence": 0.88},
        "scalp_suitability": {"noul": 0.25},
        "recommended_strategy": {"choice": "SIT_ON_HANDS"},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        jev_res = await client.evaluate_market_regime(
            atr_15m=10.0,
            atr_1h=50.0,
        )

    state = filter.evaluate_with_jev(
        adx=18.0,
        atr=100.0,
        avg_atr=120.0,
        jev_regime=jev_res,
    )

    assert state == RegimeState.RANGING
    assert jev_res is not None
    assert jev_res.market_regime == "DEAD_CHOP"
    assert jev_res.is_favorable is False


# ---------------------------------------------------------------------------
# Pillar 2: Smart Execution Routing Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_execution_routing_maker_preferred(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    fake_answers = {
        "execution_urgency": {"noul": 0.15},
        "order_type": {"choice": "MAKER_POST_ONLY", "confidence": 0.90},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        res = await client.evaluate_execution_routing(
            symbol="BTCUSD",
            side="BUY",
            spread_bps=0.8,
            book_imbalance=1.2,
            tape_velocity_1m=1.0,
        )

    assert res.order_type == "MAKER_POST_ONLY"
    assert res.execution_urgency == 0.15
    assert res.recommended_offset_ticks == 0


@pytest.mark.asyncio
async def test_evaluate_execution_routing_breakout_taker(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    fake_answers = {
        "execution_urgency": {"noul": 0.92},
        "order_type": {"choice": "INSTANT_MARKET_TAKER", "confidence": 0.95},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        res = await client.evaluate_execution_routing(
            symbol="BTCUSD",
            side="BUY",
            spread_bps=2.2,
            book_imbalance=3.5,
            tape_velocity_1m=5.0,
        )

    assert res.order_type == "INSTANT_MARKET_TAKER"
    assert res.execution_urgency == 0.92


# ---------------------------------------------------------------------------
# Pillar 3: Cross-Asset SMT Divergence Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_cross_asset_smt_trap_warning(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    fake_answers = {
        "smt_divergence_detected": {"noul": 0.85},
        "alpha_leader": {"choice": "BTC_LEADS", "confidence": 0.88},
        "favored_asset": {"choice": "AVOID_BOTH"},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        res = await client.evaluate_cross_asset_smt(
            btc_delta_5m=0.015,
            eth_delta_5m=-0.005,
            btc_bias="BULLISH",
            eth_bias="BEARISH",
            btc_bos=True,
            eth_bos=False,
            eth_btc_momentum="ETH_WEAK",
        )

    assert res.is_trap_warning is True
    assert res.favored_asset == "AVOID_BOTH"
    assert res.smt_divergence_detected == 0.85


@pytest.mark.asyncio
async def test_evaluate_cross_asset_smt_offline_fallback(jev_cfg_disabled):
    client = JevClient(api_key="mock", config=jev_cfg_disabled)
    res = await client.evaluate_cross_asset_smt(
        btc_delta_5m=0.01,
        eth_delta_5m=0.008,
        btc_bias="BULLISH",
        eth_bias="BULLISH",
        btc_bos=True,
        eth_bos=True,
    )
    assert res.is_trap_warning is False
    assert res.favored_asset in ("TRADE_BTC", "TRADE_ETH")


# ---------------------------------------------------------------------------
# Pillar 4: Dynamic Conviction Sizing Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_conviction_sizing_clamping(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    
    # Test high conviction capped at 1.5x
    fake_answers_high = {
        "conviction_multiplier": {"score": 2.2},  # Raw score > 1.5
        "risk_tier": {"choice": "AGGRESSIVE_HIGH_CONVICTION", "confidence": 0.95},
    }
    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers_high
        res_high = await client.evaluate_conviction_sizing(
            symbol="BTCUSD",
            setup_grade=3.8,
            onnx_conf=0.92,
        )
    assert res_high.conviction_multiplier == 1.5  # Clamped to 1.5

    # Test low conviction capped at 0.5x
    fake_answers_low = {
        "conviction_multiplier": {"score": 0.1},  # Raw score < 0.5
        "risk_tier": {"choice": "PROBE_HALF_SIZE", "confidence": 0.90},
    }
    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers_low
        res_low = await client.evaluate_conviction_sizing(
            symbol="ETHUSD",
            setup_grade=1.8,
            onnx_conf=0.55,
        )
    assert res_low.conviction_multiplier == 0.5  # Clamped to 0.5


def test_risk_manager_conviction_multiplier_application(risk_cfg):
    risk = RiskManager(risk_cfg)
    
    # Base size (1.0x)
    base_qty, base_margin, base_notional = risk.calculate_position_size(
        equity=10000.0,
        price=60000.0,
        leverage=25,
        contract_value=1.0,
        conviction_multiplier=1.0,
    )
    
    # Sized up (1.5x)
    boosted_qty, boosted_margin, boosted_notional = risk.calculate_position_size(
        equity=10000.0,
        price=60000.0,
        leverage=25,
        contract_value=1.0,
        conviction_multiplier=1.5,
    )
    
    # Sized down (0.5x)
    reduced_qty, reduced_margin, reduced_notional = risk.calculate_position_size(
        equity=10000.0,
        price=60000.0,
        leverage=25,
        contract_value=1.0,
        conviction_multiplier=0.5,
    )

    assert boosted_margin > base_margin
    assert reduced_margin < base_margin
    assert round(boosted_margin / base_margin, 1) == 1.5
    assert round(reduced_margin / base_margin, 1) == 0.5


# ---------------------------------------------------------------------------
# Pillar 5: Pre-Emptive Scratch Exit Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_scratch_exit_trigger(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    fake_answers = {
        "thesis_integrity": {"choice": "THESIS_INVALIDATED"},
        "scratch_action": {"choice": "EMERGENCY_SCRATCH_EXIT", "confidence": 0.95},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        res = await client.evaluate_scratch_exit(
            symbol="BTCUSD",
            side="BUY",
            seconds_held=350.0,
            unrealized_pnl_pct=-0.8,
            delta_absorbed_str="HEAVY_ABSORPTION",
            candle_stall_reason="Repeated upper wicks at supply",
        )

    assert res.should_scratch is True
    assert res.scratch_action == "EMERGENCY_SCRATCH_EXIT"
    assert res.thesis_integrity == "THESIS_INVALIDATED"


@pytest.mark.asyncio
async def test_evaluate_scratch_exit_runner_hold(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    fake_answers = {
        "thesis_integrity": {"choice": "MOMENTUM_EXPANDING"},
        "scratch_action": {"choice": "HOLD", "confidence": 0.92},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        res = await client.evaluate_scratch_exit(
            symbol="BTCUSD",
            side="BUY",
            seconds_held=120.0,
            unrealized_pnl_pct=+1.5,
            delta_absorbed_str="NORMAL",
            candle_stall_reason="NONE",
        )

    assert res.should_scratch is False
    assert res.scratch_action == "HOLD"
    assert res.thesis_integrity == "MOMENTUM_EXPANDING"


# ---------------------------------------------------------------------------
# Pillar 6: Post-Trade Forensic Reflexion & Memory Tagging Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_post_trade_forensics_tagging(jev_cfg_enabled):
    client = JevClient(api_key="mock", config=jev_cfg_enabled)
    
    fake_answers = {
        "root_cause": {"choice": "FAILED_BREAKOUT", "confidence": 0.88},
        "preventable_by_model": {"score": 3.0},
        "pattern_tag": {"choice": "TOXIC_CHOP"},
    }
    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = fake_answers
        res = await client.evaluate_post_trade_forensics(
            symbol="BTCUSD",
            side="SELL",
            entry_price=61000.0,
            exit_price=61350.0,
            pnl_pct=-1.5,
            holding_time_sec=420.0,
            exit_reason="SL_HIT",
        )
    assert res.root_cause == "FAILED_BREAKOUT"
    assert res.pattern_tag == "TOXIC_CHOP"


def test_pattern_memory_store_persists_forensic_tags():
    store = PatternMemoryStore()
    
    fp = MarketPatternFingerprint(
        symbol="BTCUSD",
        direction="LONG",
        entry_price=60000.0,
        pnl=120.0,
        close_reason="TP_HIT",
        root_cause="INSTITUTIONAL_SWEEP_CONFIRMED",
        pattern_tag="GOLDEN_DISPLACEMENT",
    )

    store.record_trade_result(
        fingerprint=fp,
        pnl=120.0,
        close_reason="TP_HIT",
        trade_id="TRD-TEST-1",
        root_cause="INSTITUTIONAL_SWEEP_CONFIRMED",
        pattern_tag="GOLDEN_DISPLACEMENT",
    )

    assert len(store.winning_patterns) > 0
    saved = store.winning_patterns[-1]
    assert saved.pattern_tag == "GOLDEN_DISPLACEMENT"
    assert saved.root_cause == "INSTITUTIONAL_SWEEP_CONFIRMED"


# ---------------------------------------------------------------------------
# End-to-End OrderManager Execution Integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_order_manager_smart_routing_and_conviction_flow(risk_cfg, jev_cfg_disabled):
    mock_delta = MagicMock()
    mock_delta.get_ticker = AsyncMock(return_value={"mark_price": 60000.0, "quotes": {"best_bid": 59995.0, "best_ask": 60005.0}})
    mock_delta.get_product = AsyncMock(return_value={
        "id": 1,
        "symbol": "BTCUSD",
        "state": "live",
        "trading_status": "operational",
        "contract_type": "perpetual_futures",
        "contract_value": "0.001",
        "tick_size": "0.5",
    })
    mock_delta.place_order = AsyncMock(return_value={"id": "delta_12345", "state": "open"})
    mock_delta.get_position = AsyncMock(return_value=None)
    mock_delta.get_product_id = MagicMock(return_value=1)
    mock_delta.set_leverage = AsyncMock(return_value=True)

    risk = RiskManager(risk_cfg)
    client = JevClient(api_key="mock", config=jev_cfg_disabled)
    om = OrderManager(delta_client=mock_delta, risk_manager=risk, config=None, jev_client=client)

    # Place order with 1.25x conviction and LIMIT_MAKER_OPTIMIZED
    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="LONG",
        price=60000.0,
        sl_price=59400.0,
        tp_price=61200.0,
        leverage=25,
        equity=10000.0,
        conviction_multiplier=1.25,
        execution_routing="LIMIT_MAKER_OPTIMIZED",
    )

    assert res is not None
    ao = om.active_order
    assert ao is not None
    assert ao.conviction_multiplier == 1.25
    assert ao.execution_routing == "LIMIT_MAKER_OPTIMIZED"

    # Verify execution dict exposes the smart routing mode
    exec_dict = om.get_execution_dict()
    assert exec_dict["execution_routing"] == "LIMIT_MAKER_OPTIMIZED"
