import pytest
from src.risk.risk_manager import RiskManager

def test_one_position_rule(risk_manager: RiskManager):
    valid, reason = risk_manager.validate_trade(
        equity=10000.0, margin=1000.0, notional=10000.0, leverage=10, 
        sl_price=49000.0, tp_price=52000.0, entry_price=50000.0, side='LONG', 
        atr=500.0, spread_bps=5.0, has_position=True
    )
    assert not valid
    assert reason == "MAX_POSITIONS_REACHED"

    valid, reason = risk_manager.validate_trade(
        equity=10000.0, margin=1000.0, notional=10000.0, leverage=10, 
        sl_price=49000.0, tp_price=52000.0, entry_price=50000.0, side='LONG', 
        atr=500.0, spread_bps=5.0, has_position=False
    )
    assert valid
    assert reason == "APPROVED"
