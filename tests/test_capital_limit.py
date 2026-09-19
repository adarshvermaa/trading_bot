import pytest
from src.risk.risk_manager import RiskManager

def test_capital_limit(risk_manager: RiskManager):
    equity = 10000.0
    margin = 9000.0 # 90% allocation > 80% allowed
    
    valid, reason = risk_manager.validate_trade(
        equity=equity, margin=margin, notional=90000.0, leverage=10, 
        sl_price=49000.0, tp_price=52000.0, entry_price=50000.0, side='LONG', 
        atr=500.0, spread_bps=5.0, has_position=False
    )
    assert not valid
    assert reason == "INSUFFICIENT_EQUITY"
