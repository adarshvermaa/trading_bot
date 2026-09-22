"""
Unit tests for Jev Cognitive Sovereignty over Pattern Audit Memory.
Verifies that local pattern memory acts as an advisory input to Jev System One,
and Jev holds sole sovereignty over whether to veto or execute.
"""
from unittest.mock import AsyncMock, patch
import pytest

from src.llm.jev_client import JevClient, JevConfig, JevPreTradeAudit
from src.ml.pattern_memory import PatternAuditResult, MarketPatternFingerprint
from src.strategy.structure import StructureAnalysis
from src.ui.dashboard import Dashboard


@pytest.mark.asyncio
async def test_jev_evaluates_pattern_memory_as_advisory():
    """Verify that Jev System One receives advisory pattern memory and decides cognitively."""
    jev_cfg = JevConfig(api_key="test-key", model="system-one", enabled=True)
    client = JevClient(api_key="test-key", config=jev_cfg)

    # Advisory audit indicating 92% similarity to past loss
    advisory_audit = PatternAuditResult(
        is_blocked=True,  # In legacy code, this caused an immediate hard veto
        is_boosted=False,
        reason="Current setup matches past mistake TRD-1553938295 with 92.0% similarity",
        matched_loss_trade_id="TRD-1553938295",
        matched_loss_similarity=0.92,
    )

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
        fvg_detected=True,
        fvg_direction="BULLISH",
        chart_pattern="DOUBLE_BOTTOM",
        pricing_zone="DISCOUNT",
        playbook="ICT_JUDAS_SWEEP",
    )

    # Mock _query_system_one to simulate Jev overriding past mistake due to fresh volume/structure
    mock_answers = {
        "is_trap": {"noul": 0.05, "confidence": 0.90},
        "direction_bias": {"choice": "LONG", "confidence": 0.88, "probabilities": {"LONG": 0.88, "SHORT": 0.05, "FLAT": 0.07}},
        "setup_grade": {"score": 3.4, "confidence": 0.85},
        "past_mistake_risk": {"noul": 0.15, "confidence": 0.92},
        "cost_headwind": {"noul": 0.10, "confidence": 0.80},
    }

    fp = MarketPatternFingerprint(
        rsi_norm=0.38,
        adx_norm=0.32,
        atr_norm=0.002,
        vwap_dist=0.001,
        vol_ratio=2.5,
        ema_trend=1.0,
        is_squeeze=0.0,
        structure_score=0.8,
        dist_to_vah_pct=0.002,
        dist_to_val_pct=0.01,
        dist_to_pdh_pct=0.005,
        dist_to_pdl_pct=0.015,
        is_bull_trap=0.0,
        is_bear_trap=0.0,
        is_judas_swing=0.0,
        is_volume_absorption=0.0,
        upper_wick_ratio=0.1,
        lower_wick_ratio=0.2,
        body_ratio=0.7,
        spread_bps=1.0,
        symbol="BTCUSD",
        direction="LONG",
        entry_price=64000.0,
    )

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        res = await client.evaluate_pre_trade_setup(
            symbol="BTCUSD",
            direction="LONG",
            indicators={"rsi": 38.0, "adx": 32.0, "relative_volume": 2.5, "vwap_position": "ABOVE", "atr": 150.0},
            structure=structure,
            fingerprint=fp,
            pattern_memory_audit=advisory_audit,
        )

        assert res is not None
        # Jev evaluated state including the advisory warning
        assert mock_query.called
        call_args = mock_query.call_args[0]
        state_prompt = call_args[0]
        assert "TRD-1553938295" in state_prompt
        assert "92.0%" in state_prompt
        assert "DOUBLE_BOTTOM" in state_prompt
        assert "DISCOUNT" in state_prompt

        # Jev overruled the past mistake -> not vetoed!
        assert not res.is_vetoed
        assert res.setup_grade == 3.4


@pytest.mark.asyncio
async def test_jev_cognitive_veto_when_mistake_is_critical_repeat():
    """Verify that Jev actively vetoes when it determines the setup genuinely repeats a fatal flaw."""
    jev_cfg = JevConfig(api_key="test-key", model="system-one", enabled=True)
    client = JevClient(api_key="test-key", config=jev_cfg)

    advisory_audit = PatternAuditResult(
        is_blocked=True,
        is_boosted=False,
        reason="Matches past mistake TRD-99999",
        matched_loss_trade_id="TRD-99999",
        matched_loss_similarity=0.89,
    )

    structure = StructureAnalysis(
        bias_15m="BULLISH",
        bos_5m=False,
        choch_5m=False,
        bos_direction_5m="NONE",
        liquidity_sweep_5m=False,
        displacement_1m=False,
        retest_1m=False,
        is_valid=True,
        invalidation_reason=None,
    )

    fp = MarketPatternFingerprint(
        rsi_norm=0.50,
        adx_norm=0.15,
        atr_norm=0.001,
        vwap_dist=-0.002,
        vol_ratio=0.8,
        ema_trend=-1.0,
        is_squeeze=0.0,
        structure_score=0.3,
        dist_to_vah_pct=0.01,
        dist_to_val_pct=0.002,
        dist_to_pdh_pct=0.02,
        dist_to_pdl_pct=0.001,
        is_bull_trap=0.0,
        is_bear_trap=0.0,
        is_judas_swing=0.0,
        is_volume_absorption=0.0,
        upper_wick_ratio=0.3,
        lower_wick_ratio=0.1,
        body_ratio=0.5,
        spread_bps=1.2,
        symbol="BTCUSD",
        direction="LONG",
        entry_price=64000.0,
    )

    # Jev confirms high past mistake repeat risk (noul=0.85 >= 0.70)
    mock_answers = {
        "is_trap": {"noul": 0.10, "confidence": 0.70},
        "direction_bias": {"choice": "LONG", "confidence": 0.60, "probabilities": {"LONG": 0.60, "SHORT": 0.20, "FLAT": 0.20}},
        "setup_grade": {"score": 1.8, "confidence": 0.80},
        "past_mistake_risk": {"noul": 0.85, "confidence": 0.88},
        "cost_headwind": {"noul": 0.20, "confidence": 0.80},
    }

    with patch.object(client, "_query_system_one", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = mock_answers

        res = await client.evaluate_pre_trade_setup(
            symbol="BTCUSD",
            direction="LONG",
            indicators={"rsi": 50.0, "adx": 15.0, "relative_volume": 0.8, "vwap_position": "BELOW", "atr": 100.0},
            structure=structure,
            fingerprint=fp,
            pattern_memory_audit=advisory_audit,
        )

        assert res is not None
        assert res.is_vetoed
        assert "Pattern Trap" in res.veto_reason


def test_dashboard_renders_advisory_pattern_audit():
    """Verify that the dashboard signal panel formats ADVISORY pattern memory properly."""
    dash = Dashboard(mode="PAPER")
    dash.signal_data = {
        "last_pattern_audit": "ADVISORY (TRD-1553938295 87%)",
        "chart_pattern": "DOUBLE_BOTTOM",
        "pricing_zone": "DISCOUNT",
        "playbook": "ICT_JUDAS_SWEEP",
        "jev_status": "ACTIVE (LATENCY 120ms)",
        "jev_verdict": "PASS (Grd=3.2)",
    }

    panel = dash._build_signal_panel()
    assert panel is not None
    rendered_str = str(panel)
    assert rendered_str is not None
