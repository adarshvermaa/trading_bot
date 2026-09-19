import pytest
import time
from src.risk.risk_manager import RiskManager

def test_daily_loss_limit(risk_manager: RiskManager):
    risk_manager.daily_start_equity = 10000.0
    risk_manager.record_trade_result(-400.0) # 4% loss, limit is 3%
    
    valid, reason = risk_manager.validate_trade(
        equity=9600.0, margin=1000.0, notional=10000.0, leverage=10, 
        sl_price=49000.0, tp_price=52000.0, entry_price=50000.0, side='LONG', 
        atr=500.0, spread_bps=5.0, has_position=False
    )
    assert not valid
    assert reason == "DAILY_LOSS_LIMIT_REACHED"

def test_consecutive_loss_limit(risk_manager: RiskManager):
    risk_manager.daily_start_equity = 10000.0
    for _ in range(5):
        risk_manager.record_trade_result(-10.0)
        risk_manager.last_loss_time = 0 # skip cooldown
        
    valid, reason = risk_manager.validate_trade(
        equity=9950.0, margin=1000.0, notional=10000.0, leverage=10, 
        sl_price=49000.0, tp_price=52000.0, entry_price=50000.0, side='LONG', 
        atr=500.0, spread_bps=5.0, has_position=False
    )
    assert not valid
    assert reason == "MAX_CONSECUTIVE_LOSSES_REACHED"

def test_cooldown_after_loss(risk_manager: RiskManager):
    risk_manager.daily_start_equity = 10000.0
    risk_manager.record_trade_result(-10.0)
    # Cooldown should be active (last_loss_time is approx time.time())
    
    valid, reason = risk_manager.validate_trade(
        equity=9990.0, margin=1000.0, notional=10000.0, leverage=10, 
        sl_price=49000.0, tp_price=52000.0, entry_price=50000.0, side='LONG', 
        atr=500.0, spread_bps=5.0, has_position=False
    )
    assert not valid
    assert reason == "COOLDOWN_ACTIVE"
