import pytest
from src.risk.risk_manager import RiskManager

def test_capital_allocation(risk_manager: RiskManager):
    equity = 10000.0
    price = 50000.0
    leverage = 10
    contract_value = 1.0
    
    size, margin, notional = risk_manager.calculate_position_size(equity, price, leverage, contract_value)
    
    assert margin <= equity * 0.80
    assert (equity - margin) >= equity * 0.20 # 20% reserve
