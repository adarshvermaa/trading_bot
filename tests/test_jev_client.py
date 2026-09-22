import asyncio
import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.config import JevConfig
from src.llm.jev_client import (
    JevClient,
    JevPreTradeAudit,
    JevActiveTradeAudit,
    JevMistakeForensics,
)


def test_jev_client_disabled_when_no_key():
    cfg = JevConfig(enabled=True)
    client = JevClient(api_key="", config=cfg)
    assert not client.enabled
    assert client.get_status() == "DISABLED"


def test_jev_client_disabled_when_config_disabled():
    cfg = JevConfig(enabled=False)
    client = JevClient(api_key="valid-key", config=cfg)
    assert not client.enabled
    assert client.get_status() == "DISABLED"


def test_jev_client_enabled_and_status():
    cfg = JevConfig(enabled=True, model="jev-latest", timeout_ms=800)
    client = JevClient(api_key="valid-key", config=cfg)
    assert client.enabled
    assert "ACTIVE [jev-latest" in client.get_status()


@pytest.mark.asyncio
async def test_evaluate_pre_trade_setup_trap_veto():
    cfg = JevConfig(enabled=True, max_trap_probability=0.35)
    client = JevClient(api_key="test-key", config=cfg)

    # Mock response returning a trap probability of 0.45
    mock_answers = {
        "is_trap": {"type": "noul", "noul": 0.45},
        "direction_bias": {
            "type": "choice",
            "choice": "LONG",
            "confidence": 0.85,
            "probabilities": {"LONG": 0.85, "SHORT": 0.05, "NEUTRAL": 0.10},
        },
        "setup_grade": {"type": "score", "score": 3.0, "confidence": 0.80},
        "cost_headwind": {"type": "noul", "noul": 0.10},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        indicators = {"rsi": 65.0, "adx": 28.0, "relative_volume": 1.8, "vwap_position": "ABOVE", "atr": 120.0}
        structure = MagicMock(bias_15m="BULLISH", bos_5m=True, choch_5m=False, liquidity_sweep_5m=False, displacement_1m=True, retest_1m=True)
        fingerprint = MagicMock(is_bull_trap=0.0, is_bear_trap=0.0, is_judas_swing=0.0, spread_bps=0.8)

        audit = await client.evaluate_pre_trade_setup(
            symbol="BTCUSD",
            direction="LONG",
            indicators=indicators,
            structure=structure,
            fingerprint=fingerprint,
        )

        assert audit is not None
        assert audit.is_vetoed is True
        assert "Trap risk too high" in audit.veto_reason
        assert audit.is_boosted is False
        assert client.total_vetoes == 1


@pytest.mark.asyncio
async def test_evaluate_pre_trade_setup_direction_conflict_veto():
    cfg = JevConfig(enabled=True, min_confidence=0.70)
    client = JevClient(api_key="test-key", config=cfg)

    # Candidate is LONG, but Jev identifies institutional SHORT with 90% confidence
    mock_answers = {
        "is_trap": {"type": "noul", "noul": 0.15},
        "direction_bias": {
            "type": "choice",
            "choice": "SHORT",
            "confidence": 0.90,
            "probabilities": {"LONG": 0.05, "SHORT": 0.90, "NEUTRAL": 0.05},
        },
        "setup_grade": {"type": "score", "score": 2.8, "confidence": 0.85},
        "cost_headwind": {"type": "noul", "noul": 0.10},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        indicators = {"rsi": 55.0, "adx": 25.0, "relative_volume": 1.2, "vwap_position": "ABOVE", "atr": 100.0}
        structure = MagicMock(bias_15m="BULLISH", bos_5m=True)
        fingerprint = MagicMock(spread_bps=0.8)

        audit = await client.evaluate_pre_trade_setup(
            symbol="BTCUSD",
            direction="LONG",
            indicators=indicators,
            structure=structure,
            fingerprint=fingerprint,
        )

        assert audit is not None
        assert audit.is_vetoed is True
        assert "Direction conflict" in audit.veto_reason
        assert audit.is_boosted is False


@pytest.mark.asyncio
async def test_evaluate_pre_trade_setup_boost():
    cfg = JevConfig(enabled=True, min_confidence=0.70, min_setup_grade=2.5)
    client = JevClient(api_key="test-key", config=cfg)

    # Jev confirms direction LONG, low trap, Grade 3.5 setup
    mock_answers = {
        "is_trap": {"type": "noul", "noul": 0.08},
        "direction_bias": {
            "type": "choice",
            "choice": "LONG",
            "confidence": 0.95,
            "probabilities": {"LONG": 0.95, "SHORT": 0.02, "NEUTRAL": 0.03},
        },
        "setup_grade": {"type": "score", "score": 3.6, "confidence": 0.90},
        "cost_headwind": {"type": "noul", "noul": 0.12},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        indicators = {"rsi": 62.0, "adx": 30.0, "relative_volume": 2.0, "vwap_position": "ABOVE", "atr": 150.0}
        structure = MagicMock(bias_15m="BULLISH", bos_5m=True)
        fingerprint = MagicMock(spread_bps=0.8)

        audit = await client.evaluate_pre_trade_setup(
            symbol="BTCUSD",
            direction="LONG",
            indicators=indicators,
            structure=structure,
            fingerprint=fingerprint,
        )

        assert audit is not None
        assert audit.is_vetoed is False
        assert audit.is_boosted is True
        assert audit.boost_amount > 0.10
        assert client.total_boosts == 1


@pytest.mark.asyncio
async def test_evaluate_active_trade_reversal_exit():
    cfg = JevConfig(enabled=True, enable_active_monitoring=True)
    client = JevClient(api_key="test-key", config=cfg)

    mock_answers = {
        "momentum_state": {
            "type": "choice",
            "choice": "REVERSAL",
            "confidence": 0.92,
            "probabilities": {"STRONG": 0.02, "FADING": 0.06, "REVERSAL": 0.92},
        },
        "should_exit_early": {"type": "noul", "noul": 0.85},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        audit = await client.evaluate_active_trade(
            symbol="BTCUSD",
            side="LONG",
            entry_price=65000.0,
            current_price=64850.0,
            pnl=-75.0,
            duration_seconds=120.0,
            rsi=38.0,
            vwap_dist_pct=-0.003,
            structure_health_notes="Support broke on high bear volume",
        )

        assert audit is not None
        assert audit.momentum_state == "REVERSAL"
        assert audit.should_exit_early >= 0.80
        assert audit.momentum_confidence == 0.92


@pytest.mark.asyncio
async def test_diagnose_stopped_out_trade():
    cfg = JevConfig(enabled=True, enable_mistake_forensics=True)
    client = JevClient(api_key="test-key", config=cfg)

    mock_answers = {
        "root_cause": {
            "type": "choice",
            "choice": "LIQUIDITY_SWEEP",
            "confidence": 0.98,
            "probabilities": {"CHOP_TRAP": 0.01, "LIQUIDITY_SWEEP": 0.98, "COUNTER_TREND": 0.01},
        },
        "was_preventable": {"type": "noul", "noul": 0.70},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        forensics = await client.diagnose_stopped_out_trade(
            symbol="ETHUSD",
            side="LONG",
            entry_price=3500.0,
            exit_price=3470.0,
            pnl=-150.0,
            close_reason="SL_HIT",
            market_context="Wicked down through support before immediate 50pt rally",
        )

        assert forensics is not None
        assert forensics.root_cause == "LIQUIDITY_SWEEP"
        assert forensics.confidence >= 0.90
        assert forensics.was_preventable >= 0.60


@pytest.mark.asyncio
async def test_timeout_fallback_does_not_block():
    cfg = JevConfig(enabled=True, timeout_ms=50)
    client = JevClient(api_key="test-key", config=cfg)

    # Simulate query timing out
    with patch("asyncio.timeout", side_effect=asyncio.TimeoutError):
        audit = await client.evaluate_pre_trade_setup(
            symbol="BTCUSD",
            direction="LONG",
            indicators={},
            structure=MagicMock(),
            fingerprint=MagicMock(),
        )
        assert audit is None
        assert client.total_fallbacks == 1


@pytest.mark.asyncio
async def test_live_handshake_if_api_key_present():
    """Live verification against TypeSafe AI if JEV_LLM is configured in environment."""
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("JEV_LLM", os.getenv("TYPESAFE_API_KEY", ""))
    if not key:
        pytest.skip("No live JEV_LLM key present in environment.")

    cfg = JevConfig(enabled=True, model="jev-latest", timeout_ms=3500)
    client = JevClient(api_key=key, config=cfg)

    try:
        indicators = {"rsi": 60.0, "adx": 30.0, "relative_volume": 1.5, "vwap_position": "ABOVE", "atr": 100.0}
        structure = MagicMock(bias_15m="BULLISH", bos_5m=True, choch_5m=False, liquidity_sweep_5m=False, displacement_1m=True, retest_1m=True)
        fingerprint = MagicMock(is_bull_trap=0.0, is_bear_trap=0.0, is_judas_swing=0.0, spread_bps=0.8)

        audit = await client.evaluate_pre_trade_setup(
            symbol="BTCUSD",
            direction="LONG",
            indicators=indicators,
            structure=structure,
            fingerprint=fingerprint,
        )
        assert audit is not None
        assert audit.direction_bias in ("LONG", "SHORT", "NEUTRAL")
        assert 0.0 <= audit.is_trap <= 1.0
        assert 0.0 <= audit.setup_grade <= 4.0
        assert audit.latency_ms > 0
    finally:
        await client.close()
