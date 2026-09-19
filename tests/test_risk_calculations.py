import pytest
from src.risk.risk_manager import RiskManager

def test_calculate_position_size(risk_manager: RiskManager):
    equity = 10000.0
    price = 50000.0
    leverage = 10
    contract_value = 1.0
    
    size, margin, notional = risk_manager.calculate_position_size(equity, price, leverage, contract_value)
    
    assert margin == equity * 0.80
    assert notional == margin * leverage
    assert size == int(notional / (contract_value * price))

def test_calculate_fees(risk_manager: RiskManager):
    notional = 100000.0
    
    fees_taker = risk_manager.calculate_fees(notional, is_taker=True)
    expected_taker = notional * 0.0005 + notional * 0.001
    assert fees_taker == expected_taker
    
    fees_maker = risk_manager.calculate_fees(notional, is_taker=False)
    expected_maker = notional * 0.0002 + notional * 0.001
    assert fees_maker == expected_maker
