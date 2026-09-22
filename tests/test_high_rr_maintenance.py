"""
Unit tests for High Risk:Reward Maintenance (>= 2.5R) and Trailing Ratchets.
Verifies:
1. Dynamic SL/TP engine guarantees institutional R:R >= 2.5R up to 5.0R.
2. Dynamic trailing stop ratchets to Breakeven (+0.1R fee buffer) at >= +1.2R.
3. Dynamic trailing stop locks runner profit (+1.0R) at >= +2.0R.
"""
from unittest.mock import AsyncMock, patch
import pytest

from src.llm.jev_client import JevClient, JevConfig


@pytest.mark.asyncio
async def test_evaluate_dynamic_sl_tp_guarantees_minimum_2_5_rr():
    jev_cfg = JevConfig(api_key="test-key", model="system-one", enabled=True, min_risk_reward_ratio=2.5)
    client = JevClient(api_key="test-key", config=jev_cfg)

    # Synthetic long setup at $60,000, swing low at $59,500 (risk = $500)
    # Opposing liquidity target at $61,500 (3.0R)
    mock_answers = {
        "sl_anchor": {"choice": "SWING_LOW", "confidence": 0.88},
        "sl_cushion": {"score": 2.0, "confidence": 0.85},
        "tp_target_type": {"choice": "OPPOSING_LIQUIDITY", "confidence": 0.90},
        "target_rr_multiple": {"score": 3.0, "confidence": 0.85},
    }

    tech_levels = {
        "swing_low": 59500.0,
        "swing_high": 60500.0,
        "eqh": 61500.0,
        "opposing_liquidity": 61500.0,
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        res = await client.evaluate_dynamic_sl_tp(
            symbol="BTCUSD",
            direction="LONG",
            entry_price=60000.0,
            atr=200.0,
            technical_levels=tech_levels,
            tick_size=0.1,
        )

        assert res is not None
        assert res.realized_rr_ratio >= 2.5
        assert res.sl_price < 60000.0
        assert res.tp_price > 60000.0
        # Target distance >= 2.5 * risk distance
        risk_dist = 60000.0 - res.sl_price
        reward_dist = res.tp_price - 60000.0
        assert reward_dist >= (2.5 * risk_dist) - 0.5


@pytest.mark.asyncio
async def test_dynamic_trailing_breakeven_ratchet_at_1_2r():
    jev_cfg = JevConfig(api_key="test-key", model="system-one", enabled=True)
    client = JevClient(api_key="test-key", config=jev_cfg)

    # Entry = 60000, initial_sl = 59000 (risk = 1000)
    # Current price = 61250 (+1.25R profit)
    # Current SL = 59000
    mock_answers = {
        "trailing_action": {"choice": "HOLD_INITIAL", "confidence": 0.60},
        "buffer_tightness": {"score": 2.0},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        res = await client.evaluate_dynamic_trailing_stop(
            symbol="BTCUSD",
            side="LONG",
            entry_price=60000.0,
            current_price=61250.0,
            current_sl=59000.0,
            initial_sl=59000.0,
            unrealized_pnl=125.0,
            duration_seconds=120.0,
            atr=150.0,
            technical_levels={"swing_low": 60500.0},
            tick_size=0.1,
        )

        assert res is not None
        # Must be ratcheted to at least entry + 0.1*risk = 60000 + 100 = 60100
        assert res.candidate_sl_price >= 60100.0


@pytest.mark.asyncio
async def test_dynamic_trailing_runner_lock_ratchet_at_2_0r():
    jev_cfg = JevConfig(api_key="test-key", model="system-one", enabled=True)
    client = JevClient(api_key="test-key", config=jev_cfg)

    # Entry = 60000, initial_sl = 59000 (risk = 1000)
    # Current price = 62100 (+2.1R profit)
    # Current SL = 60100 (breakeven held)
    mock_answers = {
        "trailing_action": {"choice": "HOLD_INITIAL", "confidence": 0.50},
        "buffer_tightness": {"score": 1.0},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        res = await client.evaluate_dynamic_trailing_stop(
            symbol="BTCUSD",
            side="LONG",
            entry_price=60000.0,
            current_price=62100.0,
            current_sl=60100.0,
            initial_sl=59000.0,
            unrealized_pnl=210.0,
            duration_seconds=300.0,
            atr=150.0,
            technical_levels={"swing_low": 60800.0},
            tick_size=0.1,
        )

        assert res is not None
        # Must be ratcheted to at least entry + 1.0*risk = 60000 + 1000 = 61000
        assert res.candidate_sl_price >= 61000.0
