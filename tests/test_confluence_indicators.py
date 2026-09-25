import numpy as np
import pytest
from src.strategy.signals import SignalGenerator


def test_calculate_cvd_bullish_and_bearish():
    # 5 candles where price closes at the high (pure buyer dominance)
    highs = np.array([105.0, 106.0, 107.0, 108.0, 109.0])
    lows = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    closes = np.array([105.0, 106.0, 107.0, 108.0, 109.0])  # close == high -> buy_ratio = 1.0 -> delta_vol = volume
    volumes = np.array([10.0, 10.0, 10.0, 10.0, 10.0])

    cvd = SignalGenerator.calculate_cvd(highs, lows, closes, volumes)
    assert len(cvd) == 5
    assert np.all(cvd == np.array([10.0, 20.0, 30.0, 40.0, 50.0]))

    # 5 candles where price closes at the low (pure seller dominance)
    closes_bear = np.array([100.0, 101.0, 102.0, 103.0, 104.0])  # close == low -> buy_ratio = 0.0 -> delta_vol = -volume
    cvd_bear = SignalGenerator.calculate_cvd(highs, lows, closes_bear, volumes)
    assert np.all(cvd_bear == np.array([-10.0, -20.0, -30.0, -40.0, -50.0]))


def test_detect_cvd_divergence():
    # Bullish divergence: price makes lower low, but CVD makes higher low
    # Lookback = 5
    close_bull = np.array([100.0, 98.0, 95.0, 96.0, 93.0])  # 93 is lower than min([100, 98, 95, 96]) = 95
    cvd_bull = np.array([10.0, 5.0, -20.0, -10.0, -5.0])    # -5 is higher than min([10, 5, -20, -10]) = -20
    assert SignalGenerator.detect_cvd_divergence(close_bull, cvd_bull, lookback=5) == "BULLISH_DIVERGENCE"

    # Bearish divergence: price makes higher high, but CVD makes lower high
    close_bear = np.array([100.0, 102.0, 105.0, 104.0, 107.0])  # 107 is higher than max([100, 102, 105, 104]) = 105
    cvd_bear = np.array([10.0, 20.0, 50.0, 40.0, 35.0])         # 35 is lower than max([10, 20, 50, 40]) = 50
    assert SignalGenerator.detect_cvd_divergence(close_bear, cvd_bear, lookback=5) == "BEARISH_DIVERGENCE"

    # Neutral when aligned
    close_neut = np.array([100.0, 102.0, 105.0, 104.0, 108.0])
    cvd_neut = np.array([10.0, 20.0, 50.0, 40.0, 60.0])
    assert SignalGenerator.detect_cvd_divergence(close_neut, cvd_neut, lookback=5) == "NEUTRAL"


def test_calculate_vwap_bands():
    n = 30
    np.random.seed(42)
    closes = np.linspace(100.0, 110.0, n)
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n, 100.0)

    bands = SignalGenerator.calculate_vwap_bands(highs, lows, closes, volumes, std_mults=(1.0, 2.0, 3.0))
    assert "vwap" in bands
    assert "upper_1.0" in bands
    assert "lower_1.0" in bands
    assert "upper_2.0" in bands
    assert "lower_2.0" in bands
    assert "upper_3.0" in bands
    assert "lower_3.0" in bands

    last_idx = -1
    assert bands["upper_3.0"][last_idx] > bands["upper_2.0"][last_idx]
    assert bands["upper_2.0"][last_idx] > bands["upper_1.0"][last_idx]
    assert bands["upper_1.0"][last_idx] >= bands["vwap"][last_idx]
    assert bands["vwap"][last_idx] >= bands["lower_1.0"][last_idx]
    assert bands["lower_1.0"][last_idx] > bands["lower_2.0"][last_idx]
    assert bands["lower_2.0"][last_idx] > bands["lower_3.0"][last_idx]


def test_calculate_squeeze_momentum():
    n = 50
    # Simulate tight consolidation for BB squeeze: constant price with tiny fluctuation
    closes = np.full(n, 100.0) + np.sin(np.linspace(0, 10, n)) * 0.1
    highs = closes + 0.2
    lows = closes - 0.2

    is_squeeze, squeeze_fired, momentum = SignalGenerator.calculate_squeeze_momentum(
        highs, lows, closes, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5
    )

    assert len(is_squeeze) == n
    assert len(squeeze_fired) == n
    assert len(momentum) == n
    # During tight consolidation, squeeze should be True in later periods
    assert np.any(is_squeeze[25:])


def test_calculate_order_flow_imbalance():
    # Bid volume dominates -> positive OFI
    bids = [[100.0, 10.0], [99.0, 5.0]]
    asks = [[101.0, 2.0], [102.0, 3.0]]
    ofi_positive = SignalGenerator.calculate_order_flow_imbalance(bids, asks, depth=5)
    assert ofi_positive > 0.0

    # Ask volume dominates -> negative OFI
    bids_low = [[100.0, 2.0], [99.0, 1.0]]
    asks_high = [[101.0, 15.0], [102.0, 10.0]]
    ofi_negative = SignalGenerator.calculate_order_flow_imbalance(bids_low, asks_high, depth=5)
    assert ofi_negative < 0.0

    # Dict format support
    bids_dict = [{"price": 100.0, "size": 10.0}]
    asks_dict = [{"price": 101.0, "size": 2.0}]
    ofi_dict = SignalGenerator.calculate_order_flow_imbalance(bids_dict, asks_dict)
    assert ofi_dict > 0.0
