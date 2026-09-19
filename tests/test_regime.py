import pytest
from src.strategy.regime import RegimeFilter, RegimeState

def test_regime_trending():
    rf = RegimeFilter()
    state = rf.evaluate(adx=28.0, atr=1.0, avg_atr=1.0)
    assert state == RegimeState.TRENDING
    advice = rf.get_position_sizing_advice(adx=28.0, atr=1.0, avg_atr=1.0)
    assert advice.position_size_multiplier == 1.0

def test_regime_ranging():
    rf = RegimeFilter()
    state = rf.evaluate(adx=15.0, atr=1.0, avg_atr=1.0)
    assert state == RegimeState.RANGING
    advice = rf.get_position_sizing_advice(adx=15.0, atr=1.0, avg_atr=1.0)
    assert advice.position_size_multiplier == 0.8

def test_regime_volatile():
    rf = RegimeFilter()
    state = rf.evaluate(adx=30.0, atr=3.0, avg_atr=1.0)
    assert state == RegimeState.VOLATILE
    advice = rf.get_position_sizing_advice(adx=30.0, atr=3.0, avg_atr=1.0)
    assert advice.position_size_multiplier == 0.5
