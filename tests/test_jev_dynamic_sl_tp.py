import os
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from dotenv import load_dotenv

from src.config import RiskConfig, StopLossConfig, TakeProfitConfig, TrailingStopConfig, JevConfig
from src.risk.risk_manager import RiskManager
from src.llm.jev_client import (
    JevClient,
    JevDynamicSLTPResult,
    JevDynamicTrailingResult,
)


@pytest.fixture
def risk_config():
    return RiskConfig(
        stop_loss=StopLossConfig(max_loss_pct_of_margin=0.03, min_stop_distance_atr=0.1),
        take_profit=TakeProfitConfig(target_pct_of_margin=0.06),
        trailing_stop=TrailingStopConfig(activation_pct_of_margin=0.02, max_profit_cap_pct_of_margin=2.0),
    )


@pytest.fixture
def jev_config():
    return JevConfig(
        enabled=True,
        enable_dynamic_sl_tp=True,
        enable_dynamic_trailing=True,
        min_risk_reward_ratio=2.0,
        target_risk_reward_ratio=2.5,
        max_risk_reward_ratio=5.0,
        trailing_interval_seconds=5,
    )


# --------------------------------------------------------------------------
# 1. Tests for validate_risk_reward_ratio
# --------------------------------------------------------------------------

def test_validate_risk_reward_ratio_long(risk_config):
    rm = RiskManager(risk_config)
    entry = 60000.0
    sl = 59000.0  # risk = 1000
    tp_2r = 62000.0  # reward = 2000 (2.0R)
    tp_1_5r = 61500.0  # reward = 1500 (1.5R)

    # Valid 2.0R
    valid, rr, reason = rm.validate_risk_reward_ratio(entry, sl, tp_2r, side="LONG", min_rr=2.0)
    assert valid is True
    assert rr == 2.0
    assert reason == "VALID"

    # Invalid < 2.0R
    valid, rr, reason = rm.validate_risk_reward_ratio(entry, sl, tp_1_5r, side="LONG", min_rr=2.0)
    assert valid is False
    assert rr == 1.5
    assert "below institutional threshold" in reason

    # Invalid SL on wrong side of entry
    valid, rr, reason = rm.validate_risk_reward_ratio(entry, 61000.0, tp_2r, side="LONG", min_rr=2.0)
    assert valid is False
    assert "Invalid risk" in reason


def test_validate_risk_reward_ratio_short(risk_config):
    rm = RiskManager(risk_config)
    entry = 60000.0
    sl = 61000.0  # risk = 1000
    tp_3r = 57000.0  # reward = 3000 (3.0R)
    tp_sub_2r = 58500.0  # reward = 1500 (1.5R)

    # Valid 3.0R
    valid, rr, reason = rm.validate_risk_reward_ratio(entry, sl, tp_3r, side="SHORT", min_rr=2.0)
    assert valid is True
    assert rr == 3.0
    assert reason == "VALID"

    # Invalid < 2.0R
    valid, rr, reason = rm.validate_risk_reward_ratio(entry, sl, tp_sub_2r, side="SHORT", min_rr=2.0)
    assert valid is False
    assert rr == 1.5
    assert "below institutional threshold" in reason


# --------------------------------------------------------------------------
# 2. Tests for calculate_sl_tp_async (Jev AI Dynamic SL/TP)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_calculate_sl_tp_async_order_block_anchor(risk_config, jev_config):
    """Test dynamic Order Block anchor placement with >= 2.5R reward."""
    mock_jev = MagicMock(spec=JevClient)
    mock_jev.enabled = True
    mock_jev.config = jev_config
    mock_jev.evaluate_dynamic_sl_tp = AsyncMock(return_value=JevDynamicSLTPResult(
        sl_price=59850.0,
        tp_price=60400.0,
        sl_anchor="ORDER_BLOCK",
        tp_target_type="LIQUIDITY_POOL",
        target_rr_multiple=2.67,
        sl_cushion_grade=1.0,
        realized_rr_ratio=2.67,
        confidence=0.92,
        latency_ms=180.0,
    ))

    rm = RiskManager(risk_config, jev_client=mock_jev)
    technical_levels = {
        "ob_bottom": 59880.0,
        "ob_top": 59950.0,
        "nearest_resistance": 60500.0,
    }

    sl, tp, meta = await rm.calculate_sl_tp_async(
        side="LONG",
        entry_price=60000.0,
        margin=1000.0,
        leverage=10,
        contract_value=1.0,
        size=1,
        atr=100.0,
        tick_size=0.1,
        technical_levels=technical_levels,
        symbol="BTCUSD",
    )

    assert meta["source"] == "JEV"
    assert meta["sl_anchor"] == "ORDER_BLOCK"
    assert meta["tp_target_type"] == "LIQUIDITY_POOL"
    assert sl == 59850.0
    assert tp == 60400.0
    assert meta["realized_rr"] >= 2.0


@pytest.mark.asyncio
async def test_calculate_sl_tp_async_capital_safety_clamp(risk_config, jev_config):
    """Test that Jev's SL is strictly clamped if it exceeds the 3% max margin loss limit."""
    mock_jev = MagicMock(spec=JevClient)
    mock_jev.enabled = True
    mock_jev.config = jev_config

    # At 10x leverage, 3% margin loss = 0.3% price drop from entry.
    # Entry 60,000 -> Max loss price diff = 60,000 * 0.003 = 180 USD.
    # Maximum safe floor = 59,820 USD.
    # If Jev suggests SL way down at 59,000 USD, it MUST be clamped to 59,820 USD!
    mock_jev.evaluate_dynamic_sl_tp = AsyncMock(return_value=JevDynamicSLTPResult(
        sl_price=59000.0,  # Unsafe wide SL
        tp_price=63000.0,
        sl_anchor="SWING_POINT",
        tp_target_type="MEASURED_EXTENSION",
        target_rr_multiple=3.0,
        sl_cushion_grade=3.0,
        realized_rr_ratio=3.0,
        confidence=0.85,
        latency_ms=190.0,
    ))

    rm = RiskManager(risk_config, jev_client=mock_jev)
    sl, tp, meta = await rm.calculate_sl_tp_async(
        side="LONG",
        entry_price=60000.0,
        margin=1000.0,
        leverage=10,
        contract_value=1.0,
        size=1,
        atr=100.0,
        tick_size=0.1,
        technical_levels={},
        symbol="BTCUSD",
    )

    assert meta["source"] == "JEV"
    # SL must not be 59,000; it must be clamped to max margin loss floor 59,820.0
    assert sl == 59820.0
    # And TP must maintain >= 2.0R based on the clamped risk (180 USD risk -> TP >= 60,360)
    assert tp >= 60000.0 + (2.0 * 180.0)


@pytest.mark.asyncio
async def test_calculate_sl_tp_async_rr_hurdle_enforcement(risk_config, jev_config):
    """Test that if Jev suggests a TP with sub-2.0R, RiskManager expands TP to ensure >= 2.0R."""
    mock_jev = MagicMock(spec=JevClient)
    mock_jev.enabled = True
    mock_jev.config = jev_config

    # Long: Entry 60000, SL 59900 (Risk=100). Jev suggests TP 60120 (Reward=120 -> only 1.2R).
    mock_jev.evaluate_dynamic_sl_tp = AsyncMock(return_value=JevDynamicSLTPResult(
        sl_price=59900.0,
        tp_price=60120.0,  # Only 1.2R
        sl_anchor="SWEEP_WICK",
        tp_target_type="VALUE_AREA_EXTREME",
        target_rr_multiple=2.0,
        sl_cushion_grade=0.0,
        realized_rr_ratio=1.2,
        confidence=0.95,
        latency_ms=140.0,
    ))

    rm = RiskManager(risk_config, jev_client=mock_jev)
    sl, tp, meta = await rm.calculate_sl_tp_async(
        side="LONG",
        entry_price=60000.0,
        margin=1000.0,
        leverage=10,
        contract_value=1.0,
        size=1,
        atr=50.0,
        tick_size=0.1,
        symbol="BTCUSD",
    )

    assert sl == 59900.0
    # TP must be expanded to at least 2.0 * 100 = 60200.0
    assert tp >= 60200.0
    assert meta["realized_rr"] >= 2.0


@pytest.mark.asyncio
async def test_calculate_sl_tp_async_fallback_on_disabled(risk_config):
    """Test graceful institutional fallback when Jev is disabled."""
    rm = RiskManager(risk_config, jev_client=None)

    sl, tp, meta = await rm.calculate_sl_tp_async(
        side="LONG",
        entry_price=60000.0,
        margin=1000.0,
        leverage=10,
        contract_value=1.0,
        size=1,
        atr=100.0,
        tick_size=0.1,
        symbol="BTCUSD",
    )

    assert meta["source"] == "FALLBACK"
    assert sl < 60000.0
    assert tp > 60000.0


# --------------------------------------------------------------------------
# 3. Tests for calculate_dynamic_jev_trailing_stop (Ratchet Invariant)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dynamic_trailing_ratchet_invariant_long(risk_config, jev_config):
    """Test that Trailing Stop for Long can only move UP, never DOWN."""
    mock_jev = MagicMock(spec=JevClient)
    mock_jev.enabled = True
    mock_jev.config = jev_config

    rm = RiskManager(risk_config, jev_client=mock_jev)

    # Initial SL: 59,500. Current price: 60,500 (+1.0R profit).
    # Jev trails to 59,900 (TRAIL_RECENT_SWING).
    mock_jev.evaluate_dynamic_trailing_stop = AsyncMock(return_value=JevDynamicTrailingResult(
        action="TRAIL_RECENT_SWING",
        candidate_sl_price=59900.0,
        buffer_tightness=2.0,
        confidence=0.90,
        reason="Trailed to swing",
        latency_ms=120.0,
    ))

    new_sl, updated, msg = await rm.calculate_dynamic_jev_trailing_stop(
        symbol="BTCUSD",
        side="LONG",
        entry_price=60000.0,
        current_price=60500.0,
        current_sl=59500.0,
        initial_sl=59500.0,
        margin=1000.0,
        contract_value=1.0,
        size=1,
        duration_seconds=30.0,
        atr=100.0,
    )
    assert updated is True
    assert new_sl == 59900.0

    # Next cycle: price pulled back to 60,100. Jev evaluates a lower candidate SL 59,700.
    # The Ratchet Rule MUST reject 59,700 and keep 59,900!
    mock_jev.evaluate_dynamic_trailing_stop = AsyncMock(return_value=JevDynamicTrailingResult(
        action="TRAIL_RECENT_SWING",
        candidate_sl_price=59700.0,  # Lower than current SL 59,900!
        buffer_tightness=2.0,
        confidence=0.88,
        reason="Pullback swing",
        latency_ms=115.0,
    ))

    new_sl, updated, msg = await rm.calculate_dynamic_jev_trailing_stop(
        symbol="BTCUSD",
        side="LONG",
        entry_price=60000.0,
        current_price=60100.0,
        current_sl=59900.0,  # Current active SL
        initial_sl=59500.0,
        margin=1000.0,
        contract_value=1.0,
        size=1,
        duration_seconds=45.0,
        atr=100.0,
    )
    # Ratchet invariant holds: cannot loosen SL!
    assert updated is False
    assert new_sl == 59900.0
    assert "Ratchet held" in msg or "<= current SL" in msg


@pytest.mark.asyncio
async def test_dynamic_trailing_ratchet_invariant_short(risk_config, jev_config):
    """Test that Trailing Stop for Short can only move DOWN, never UP."""
    mock_jev = MagicMock(spec=JevClient)
    mock_jev.enabled = True
    mock_jev.config = jev_config

    rm = RiskManager(risk_config, jev_client=mock_jev)

    # Initial SL: 60,500. Current price: 59,200 (+1.3R profit).
    # Jev trails to 59,800.
    mock_jev.evaluate_dynamic_trailing_stop = AsyncMock(return_value=JevDynamicTrailingResult(
        action="TRAIL_RECENT_SWING",
        candidate_sl_price=59800.0,
        buffer_tightness=2.0,
        confidence=0.91,
        reason="Trailed to swing high",
        latency_ms=130.0,
    ))

    new_sl, updated, msg = await rm.calculate_dynamic_jev_trailing_stop(
        symbol="BTCUSD",
        side="SHORT",
        entry_price=60000.0,
        current_price=59200.0,
        current_sl=60500.0,
        initial_sl=60500.0,
        margin=1000.0,
        contract_value=1.0,
        size=1,
        duration_seconds=40.0,
        atr=100.0,
    )
    assert updated is True
    assert new_sl == 59800.0

    # Next cycle: candidate is 59,950 (higher than 59,800). Ratchet MUST hold 59,800!
    mock_jev.evaluate_dynamic_trailing_stop = AsyncMock(return_value=JevDynamicTrailingResult(
        action="TRAIL_RECENT_SWING",
        candidate_sl_price=59950.0,
        buffer_tightness=2.0,
        confidence=0.85,
        reason="Wider swing",
        latency_ms=120.0,
    ))

    new_sl, updated, msg = await rm.calculate_dynamic_jev_trailing_stop(
        symbol="BTCUSD",
        side="SHORT",
        entry_price=60000.0,
        current_price=59500.0,
        current_sl=59800.0,
        initial_sl=60500.0,
        margin=1000.0,
        contract_value=1.0,
        size=1,
        duration_seconds=55.0,
        atr=100.0,
    )
    assert updated is False
    assert new_sl == 59800.0
    assert "Ratchet held" in msg or ">= current SL" in msg


# --------------------------------------------------------------------------
# 4. End-to-End Order Placement with Jev Metadata
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_order_manager_captures_jev_metadata():
    """Verify that OrderManager records dynamic SL/TP and R:R metadata onto ActiveOrder."""
    from src.execution.order_manager import OrderManager, OrderState
    from src.execution.delta import DeltaExchangeClient

    delta_mock = MagicMock(spec=DeltaExchangeClient)
    delta_mock.live_trading = False
    delta_mock.get_product = AsyncMock(return_value={
        "id": 1,
        "symbol": "BTCUSD",
        "contract_type": "perpetual_futures",
        "state": "live",
        "trading_status": "operational",
        "tick_size": "0.1",
        "contract_value": "1.0",
        "max_leverage": "100",
    })
    delta_mock.set_leverage = AsyncMock(return_value=True)
    delta_mock.has_active_position = AsyncMock(return_value=False)
    delta_mock.get_positions = AsyncMock(return_value=[])
    delta_mock.place_order = AsyncMock(return_value={
        "id": "ord-12345",
        "state": "filled",
        "average_fill_price": "60000.0",
    })

    risk_mock = MagicMock(spec=RiskManager)
    risk_mock.validate_leverage.return_value = (True, 10, "VALID")
    risk_mock.calculate_position_size.return_value = (1, 1000.0, 10000.0)
    risk_mock.validate_trade.return_value = (True, "APPROVED")

    # Real calculate_sl_tp_async call simulation:
    risk_mock.calculate_sl_tp_async = AsyncMock(return_value=(
        59500.0,
        61250.0,
        {
            "source": "JEV",
            "sl_anchor": "ORDER_BLOCK",
            "tp_target_type": "LIQUIDITY_POOL",
            "target_rr": 2.5,
            "realized_rr": 2.5,
            "cushion_grade": 1.0,
            "confidence": 0.94,
        }
    ))

    om = OrderManager(delta_mock, risk_mock, config=None)

    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="LONG",
        price=60000.0,
        equity=10000.0,
        atr=100.0,
        technical_levels={"ob_bottom": 59550.0},
    )

    assert res is not None
    active = om.active_order
    assert active is not None
    assert active.sl_price == 59500.0
    assert active.tp_price == 61250.0
    assert active.sl_anchor == "ORDER_BLOCK"
    assert active.tp_target_type == "LIQUIDITY_POOL"
    assert active.target_rr == 2.5
    assert active.realized_rr == 2.5
    assert active.jev_metadata["source"] == "JEV"


# --------------------------------------------------------------------------
# 5. Live Jev System One Handshake & Payload Validation
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_jev_dynamic_sl_tp_and_trailing():
    """Live verification of dynamic SL/TP and trailing against TypeSafe AI if JEV_LLM key is in .env."""
    load_dotenv()
    key = os.getenv("JEV_LLM", os.getenv("TYPESAFE_API_KEY", ""))
    if not key:
        pytest.skip("No live JEV_LLM key present in environment.")

    cfg = JevConfig(
        enabled=True,
        model="jev-latest",
        enable_dynamic_sl_tp=True,
        enable_dynamic_trailing=True,
        min_risk_reward_ratio=2.0,
        target_risk_reward_ratio=2.5,
        max_risk_reward_ratio=5.0,
        timeout_ms=2500,
    )
    client = JevClient(api_key=key, config=cfg)
    risk_cfg = RiskConfig(
        stop_loss=StopLossConfig(max_loss_pct_of_margin=0.03, min_stop_distance_atr=0.1),
        take_profit=TakeProfitConfig(target_pct_of_margin=0.06),
        trailing_stop=TrailingStopConfig(activation_pct_of_margin=0.02, max_profit_cap_pct_of_margin=2.0),
    )
    rm = RiskManager(risk_cfg, jev_client=client)

    try:
        # 1. Live Dynamic SL/TP Resolution
        tech_levels = {
            "swing_low": 59800.0,
            "swing_high": 60500.0,
            "ob_bottom": 59850.0,
            "ob_top": 59950.0,
            "trap_wick_price": 59780.0,
            "nearest_resistance": 61500.0,
        }
        sl, tp, meta = await rm.calculate_sl_tp_async(
            side="LONG",
            entry_price=60000.0,
            margin=1000.0,
            leverage=10,
            contract_value=1.0,
            size=1,
            atr=100.0,
            tick_size=0.1,
            technical_levels=tech_levels,
            symbol="BTCUSD",
        )

        assert meta["source"] in ("JEV", "FALLBACK")
        assert sl < 60000.0
        assert tp > 60000.0
        # Check strict 2.0R constraint
        risk = 60000.0 - sl
        reward = tp - 60000.0
        assert reward / risk >= 1.99

        # 2. Live Dynamic Trailing Stop Resolution
        new_sl, updated, msg = await rm.calculate_dynamic_jev_trailing_stop(
            symbol="BTCUSD",
            side="LONG",
            entry_price=60000.0,
            current_price=60800.0,
            current_sl=59850.0,
            initial_sl=59850.0,
            margin=1000.0,
            contract_value=1.0,
            size=1,
            duration_seconds=45.0,
            atr=100.0,
            technical_levels={"swing_low": 60400.0, "rsi": 62.0},
            tick_size=0.1,
        )
        assert isinstance(updated, bool)
        assert new_sl >= 59850.0  # Ratchet invariant: never lower than current SL
        assert len(msg) > 0

    finally:
        await client.close()
