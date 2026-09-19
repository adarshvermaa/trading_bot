import numpy as np
import pytest
from src.strategy.signals import SignalGenerator

def test_calculate_ema():
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    ema = SignalGenerator.calculate_ema(prices, period=3)
    assert len(ema) == len(prices)
    assert ema[0] == 10.0
    # Values should trend upwards
    assert np.all(np.diff(ema) > 0)

def test_calculate_vwap():
    high = np.array([10.5, 11.5, 12.5])
    low = np.array([9.5, 10.5, 11.5])
    close = np.array([10.0, 11.0, 12.0])
    volume = np.array([100.0, 200.0, 300.0])
    vwap = SignalGenerator.calculate_vwap(high, low, close, volume)
    assert len(vwap) == len(close)
    # Typical prices are 10, 11, 12
    # VWAP[0] = 10, VWAP[1] = (1000 + 2200)/300 = 10.667
    assert abs(vwap[0] - 10.0) < 1e-4
    assert vwap[1] < vwap[2]

def test_calculate_rsi():
    # 20 points ascending
    prices = np.linspace(100.0, 200.0, 30)
    rsi = SignalGenerator.calculate_rsi(prices, period=14)
    assert len(rsi) == len(prices)
    # With pure gains, RSI should be near 100
    assert rsi[-1] > 90.0

def test_calculate_relative_volume_zero_division():
    """Test that all-zero volume arrays do not raise division by zero or produce NaNs."""
    volume = np.zeros(30, dtype=np.float64)
    rel_vol = SignalGenerator.calculate_relative_volume(volume, period=20)
    assert len(rel_vol) == 30
    assert not np.isnan(rel_vol).any()
    assert not np.isinf(rel_vol).any()
    assert (rel_vol == 1.0).any()

