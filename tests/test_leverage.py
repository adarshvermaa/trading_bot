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


def test_100x_leverage_production_config():
    """Verify production config loads 100x high leverage for BTC and ETH."""
    from src.config import load_config
    config = load_config()
    assert config.risk.leverage.high_leverage_value == 100
    assert config.risk.leverage.default_leverage_value == 50
    assert config.risk.leverage.safe_fallback_leverage == 25
    assert "BTCUSD" in config.risk.leverage.high_leverage_assets
    assert "ETHUSD" in config.risk.leverage.high_leverage_assets

    rm = RiskManager(config.risk)
    valid, lev, reason = rm.validate_leverage("BTCUSD", 100)
    assert valid is True
    assert lev == 100

    valid_eth, lev_eth, _ = rm.validate_leverage("ETHUSD", 100)
    assert valid_eth is True
    assert lev_eth == 100


def test_100x_micro_balance_trade_validation():
    """Verify that a micro account ($1.00 balance) with 100x leverage passes validation for 1 minimum contract."""
    from src.config import load_config
    config = load_config()
    rm = RiskManager(config.risk)

    # 1 BTC contract at $85,000 with 100x leverage = $85 notional, $0.85 margin
    equity = 1.00
    leverage = 100
    price = 85000.0
    contract_val = 0.001
    notional = 1.0 * contract_val * price # $85.00
    margin = notional / leverage           # $0.85

    valid, reason = rm.validate_trade(
        equity=equity,
        margin=margin,
        notional=notional,
        leverage=leverage,
        sl_price=84974.5,
        tp_price=86000.0,
        entry_price=price,
        side="LONG",
        atr=50.0,
        is_min_contract=True,
    )
    assert valid is True
    assert reason == "APPROVED"
