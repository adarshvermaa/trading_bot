import pytest
from src.risk.risk_manager import RiskManager

def test_sl_tp_long(risk_manager: RiskManager):
    entry_price = 50000.0
    margin = 1000.0
    leverage = 10
    contract_value = 0.001
    size = int(margin * leverage / (entry_price * contract_value))
    atr = 50.0
    
    sl_price, valid, reason = risk_manager.calculate_stop_loss(entry_price, 'LONG', margin, leverage, contract_value, size, atr)
    assert valid
    assert sl_price < entry_price
    
    tp_price = risk_manager.calculate_take_profit(entry_price, 'LONG', margin, leverage, contract_value, size)
    assert tp_price > entry_price

def test_sl_tp_short(risk_manager: RiskManager):
    entry_price = 50000.0
    margin = 1000.0
    leverage = 10
    contract_value = 0.001
    size = int(margin * leverage / (entry_price * contract_value))
    atr = 50.0
    
    sl_price, valid, reason = risk_manager.calculate_stop_loss(entry_price, 'SHORT', margin, leverage, contract_value, size, atr)
    assert valid
    assert sl_price > entry_price
    
    tp_price = risk_manager.calculate_take_profit(entry_price, 'SHORT', margin, leverage, contract_value, size)
    assert tp_price < entry_price

def test_sl_rejected_when_too_close(risk_manager: RiskManager):
    entry_price = 50000.0
    margin = 10.0 # tiny margin -> tiny loss -> tiny price diff
    leverage = 10
    contract_value = 1.0
    size = 1
    atr = 500.0
    
    sl_price, valid, reason = risk_manager.calculate_stop_loss(entry_price, 'LONG', margin, leverage, contract_value, size, atr)
    assert not valid
    assert "Stop loss too close" in reason

def test_sl_tp_high_leverage_btc_150x(risk_manager: RiskManager):
    """Test that 150x leverage on BTC does not produce negative or distorted stop loss prices."""
    entry_price = 64250.0
    margin = 500.0
    leverage = 150
    contract_value = 0.001
    size = int(margin * leverage / (entry_price * contract_value)) # ~1167 contracts
    atr = 10.0
    
    sl_price, valid, reason = risk_manager.calculate_stop_loss(
        entry_price, 'LONG', margin, leverage, contract_value, size, atr, tick_size=0.5
    )
    assert valid, f"SL rejected: {reason}"
    assert sl_price > 0.0, f"Stop loss price {sl_price} must be strictly positive"
    assert sl_price < entry_price
    # Stop distance should reflect ~5% margin risk at 150x: ~21.4 USD
    assert 10.0 <= (entry_price - sl_price) <= 50.0

    tp_price = risk_manager.calculate_take_profit(
        entry_price, 'LONG', margin, leverage, contract_value, size, tick_size=0.5
    )
    assert tp_price > entry_price
    # TP distance should be ~10% margin risk (1:2 R:R): ~42.8 USD
    assert (tp_price - entry_price) > (entry_price - sl_price)

