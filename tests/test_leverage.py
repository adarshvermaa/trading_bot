import pytest
from src.risk.risk_manager import RiskManager

def test_btc_eth_high_leverage(risk_manager: RiskManager):
    valid, lev, reason = risk_manager.validate_leverage("BTCUSD", 200)
    assert valid
    assert lev == 150
    
    valid, lev, reason = risk_manager.validate_leverage("ETHUSD", 200)
    assert valid
    assert lev == 150

def test_sol_default_leverage(risk_manager: RiskManager):
    valid, lev, reason = risk_manager.validate_leverage("SOLUSD", 200)
    assert valid
    assert lev == 75

def test_leverage_capped_by_exchange(risk_manager: RiskManager):
    risk_manager.config.leverage.reject_on_leverage_fail = False
    valid, lev, reason = risk_manager.validate_leverage("BTCUSD", 100)
    assert valid
    assert lev == 20  # Fallbacks to safe_fallback_leverage if exchange max < desired

def test_reject_on_leverage_fail(risk_manager: RiskManager):
    risk_manager.config.leverage.reject_on_leverage_fail = True
    valid, lev, reason = risk_manager.validate_leverage("BTCUSD", 100)
    assert not valid
    assert reason == "Exchange max leverage too low and reject_on_leverage_fail is True"

def test_safe_fallback_leverage(risk_manager: RiskManager):
    risk_manager.config.leverage.reject_on_leverage_fail = False
    valid, lev, reason = risk_manager.validate_leverage("BTCUSD", 10)
    assert not valid
    assert reason == "Exchange max leverage below safe fallback"
