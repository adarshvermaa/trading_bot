import os
import time
import shutil
import tempfile
import numpy as np
import pytest

from src.ml.pattern_memory import (
    MarketPatternFingerprint,
    PatternMemoryStore,
    PatternAuditResult,
    NUM_FEATURES,
)
from src.ml.onnx_model import ONNXScalperModel


@pytest.fixture
def temp_pattern_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_fingerprint_vector_shape():
    fp = MarketPatternFingerprint(
        rsi_norm=0.55,
        adx_norm=0.30,
        atr_norm=0.002,
        vwap_dist=0.001,
        vol_ratio=1.5,
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
    vec = fp.to_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (NUM_FEATURES,)
    assert vec.dtype == np.float32


def test_pattern_memory_preseed(temp_pattern_dir):
    store = PatternMemoryStore(storage_dir=temp_pattern_dir)
    # Preseeded patterns should exist on fresh start
    assert len(store.winning_patterns) >= 3
    assert len(store.losing_patterns) >= 3
    assert store._win_matrix is not None
    assert store._loss_matrix is not None


def test_mistake_pattern_veto(temp_pattern_dir):
    store = PatternMemoryStore(storage_dir=temp_pattern_dir, loss_similarity_threshold=0.85)

    # 1. Create a known mistake: Long into PDH resistance with weak volume
    mistake = MarketPatternFingerprint(
        rsi_norm=0.75,
        adx_norm=0.15,
        atr_norm=0.002,
        vwap_dist=0.009,
        vol_ratio=0.50,
        ema_trend=1.0,
        is_squeeze=0.0,
        structure_score=0.30,
        dist_to_vah_pct=0.001,
        dist_to_val_pct=0.015,
        dist_to_pdh_pct=-0.0005,
        dist_to_pdl_pct=0.02,
        is_bull_trap=1.0,
        is_bear_trap=0.0,
        is_judas_swing=0.0,
        is_volume_absorption=0.0,
        upper_wick_ratio=0.45,
        lower_wick_ratio=0.1,
        body_ratio=0.45,
        spread_bps=1.2,
        symbol="BTCUSD",
        direction="LONG",
        pnl=-2.50,
        close_reason="BULL_TRAP_STOPPED_OUT",
        trade_id="TEST-MISTAKE-001",
    )
    store.record_trade_result(mistake, pnl=-2.50, close_reason="BULL_TRAP_STOPPED_OUT", trade_id="TEST-MISTAKE-001")

    # 2. Present an identical / near-identical candidate LONG setup
    candidate = MarketPatternFingerprint(
        rsi_norm=0.74,
        adx_norm=0.16,
        atr_norm=0.002,
        vwap_dist=0.0088,
        vol_ratio=0.52,
        ema_trend=1.0,
        is_squeeze=0.0,
        structure_score=0.31,
        dist_to_vah_pct=0.001,
        dist_to_val_pct=0.015,
        dist_to_pdh_pct=-0.0004,
        dist_to_pdl_pct=0.02,
        is_bull_trap=1.0,
        is_bear_trap=0.0,
        is_judas_swing=0.0,
        is_volume_absorption=0.0,
        upper_wick_ratio=0.44,
        lower_wick_ratio=0.1,
        body_ratio=0.46,
        spread_bps=1.2,
        symbol="BTCUSD",
        direction="LONG",
    )

    audit = store.check_pattern(candidate)
    assert audit.is_blocked is True
    assert audit.matched_loss_trade_id == "TEST-MISTAKE-001"
    assert audit.matched_loss_similarity >= 0.85
    assert "VETO" in audit.reason


def test_winning_pattern_boost(temp_pattern_dir):
    store = PatternMemoryStore(storage_dir=temp_pattern_dir, win_similarity_threshold=0.80)

    # 1. Create a known high-profit Bear Trap Reversal
    winner = MarketPatternFingerprint(
        rsi_norm=0.30,
        adx_norm=0.32,
        atr_norm=0.0025,
        vwap_dist=-0.006,
        vol_ratio=2.6,
        ema_trend=-1.0,
        is_squeeze=0.0,
        structure_score=0.88,
        dist_to_vah_pct=-0.015,
        dist_to_val_pct=-0.001,
        dist_to_pdh_pct=-0.02,
        dist_to_pdl_pct=0.0002,
        is_bull_trap=0.0,
        is_bear_trap=1.0,
        is_judas_swing=0.0,
        is_volume_absorption=0.0,
        upper_wick_ratio=0.1,
        lower_wick_ratio=0.58,
        body_ratio=0.32,
        spread_bps=0.8,
        symbol="BTCUSD",
        direction="LONG",
        pnl=6.50,
        close_reason="TP_HIT",
        trade_id="TEST-WIN-001",
    )
    store.record_trade_result(winner, pnl=6.50, close_reason="TP_HIT", trade_id="TEST-WIN-001")

    # 2. Present very similar setup
    candidate = MarketPatternFingerprint(
        rsi_norm=0.31,
        adx_norm=0.31,
        atr_norm=0.0025,
        vwap_dist=-0.0058,
        vol_ratio=2.5,
        ema_trend=-1.0,
        is_squeeze=0.0,
        structure_score=0.86,
        dist_to_vah_pct=-0.014,
        dist_to_val_pct=-0.001,
        dist_to_pdh_pct=-0.02,
        dist_to_pdl_pct=0.0003,
        is_bull_trap=0.0,
        is_bear_trap=1.0,
        is_judas_swing=0.0,
        is_volume_absorption=0.0,
        upper_wick_ratio=0.1,
        lower_wick_ratio=0.56,
        body_ratio=0.34,
        spread_bps=0.8,
        symbol="BTCUSD",
        direction="LONG",
    )

    audit = store.check_pattern(candidate)
    assert audit.is_blocked is False
    assert audit.is_boosted is True
    assert audit.confidence_adjustment > 0
    assert audit.matched_win_trade_id in ("TEST-WIN-001", "PRESEED-WIN-BEAR-TRAP-LONG")


def test_direction_isolation(temp_pattern_dir):
    store = PatternMemoryStore(storage_dir=temp_pattern_dir)

    # A short mistake should NEVER veto a long candidate
    short_mistake = MarketPatternFingerprint(
        rsi_norm=0.20,
        direction="SHORT",
        pnl=-2.0,
        close_reason="SHORT_STOPPED",
        trade_id="TEST-SHORT-LOSS",
    )
    store.record_trade_result(short_mistake, pnl=-2.0, close_reason="SHORT_STOPPED", trade_id="TEST-SHORT-LOSS")

    long_candidate = MarketPatternFingerprint(
        rsi_norm=0.20,
        direction="LONG",
    )

    audit = store.check_pattern(long_candidate)
    assert audit.is_blocked is False


def test_sub_millisecond_speed_benchmark(temp_pattern_dir):
    store = PatternMemoryStore(storage_dir=temp_pattern_dir)

    # Populate 100 patterns
    for i in range(100):
        fp = MarketPatternFingerprint(
            rsi_norm=np.random.uniform(0.2, 0.8),
            vol_ratio=np.random.uniform(0.5, 3.0),
            direction="LONG" if i % 2 == 0 else "SHORT",
        )
        if i % 2 == 0:
            store.winning_patterns.append(fp)
        else:
            store.losing_patterns.append(fp)
    store._rebuild_matrices()

    candidate = MarketPatternFingerprint(rsi_norm=0.50, vol_ratio=1.2, direction="LONG")

    start = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        store.check_pattern(candidate)
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / iterations) * 1000.0
    print(f"\nAverage pattern check time: {avg_time_ms:.4f} ms")
    assert avg_time_ms < 0.5  # Must be under 0.5 milliseconds!


def test_onnx_20_features_and_legacy_compatibility():
    model_path = "models/scalper_model.onnx"
    assert os.path.exists(model_path)

    model = ONNXScalperModel(model_path)
    assert model.is_available

    # 1. 20-feature inference via MarketPatternFingerprint
    fp = MarketPatternFingerprint(
        rsi_norm=0.55,
        adx_norm=0.30,
        vol_ratio=1.5,
        ema_trend=1.0,
        structure_score=0.8,
        direction="LONG",
    )
    feats_20 = model.prepare_pattern_features(fp)
    direction, conf, _ = model.predict(feats_20)
    assert direction in ("LONG", "SHORT", "NONE")
    assert 0.0 <= conf <= 1.0

    # 2. Legacy 7-feature inference backwards compatibility
    feats_7 = model.prepare_features("BULLISH", 55.0, 0.002, 0.001, 1.5, 30.0, 0.8)
    dir_7, conf_7, _ = model.predict(feats_7)
    assert dir_7 in ("LONG", "SHORT", "NONE")
    assert 0.0 <= conf_7 <= 1.0
