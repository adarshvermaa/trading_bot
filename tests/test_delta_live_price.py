import pytest
from unittest.mock import AsyncMock

from src.execution.delta import DeltaExchangeClient
from src.risk.risk_manager import RiskManager
from src.config import load_config

def test_delta_sl_tp_ratios():
    """Verify that Stop Loss is -3.0% of margin and Trailing Stop is enabled up to +200% ceiling."""
    config = load_config()
    risk_manager = RiskManager(config.risk)

    assert config.risk.stop_loss.max_loss_pct_of_margin == 0.03
    assert config.risk.take_profit.target_pct_of_margin == 2.00
    assert config.risk.trailing_stop.enabled is True
    assert config.risk.trailing_stop.activation_pct_of_margin == 0.02
    assert config.risk.trailing_stop.buffer_pct_of_margin == 0.01

    entry_price = 81300.0
    margin = 100.0
    leverage = 150
    contract_value = 0.001
    size = int(margin * leverage / (entry_price * contract_value)) # ~184 contracts
    atr = 25.0

    sl_price, valid, reason = risk_manager.calculate_stop_loss(
        entry_price, 'LONG', margin, leverage, contract_value, size, atr, tick_size=0.5
    )
    assert valid, f"SL invalid: {reason}"
    assert sl_price < entry_price

    tp_price = risk_manager.calculate_take_profit(
        entry_price, 'LONG', margin, leverage, contract_value, size, tick_size=0.5
    )
    assert tp_price > entry_price

    price_diff_sl = entry_price - sl_price
    price_diff_tp = tp_price - entry_price

    # Loss at SL must not exceed 3% of margin
    loss_at_sl = price_diff_sl * (size * contract_value)
    assert pytest.approx(loss_at_sl, rel=0.05) == margin * 0.03

    # Profit at max TP ceiling equals 200% of margin
    profit_at_tp = price_diff_tp * (size * contract_value)
    assert pytest.approx(profit_at_tp, rel=0.05) == margin * 2.00

def test_delta_client_latest_price_cache():
    """Verify DeltaExchangeClient maintains _latest_prices and get_latest_price()."""
    client = DeltaExchangeClient(api_key="", api_secret="", live_trading=False)
    client._latest_prices["BTCUSD"] = 81270.5
    client._latest_mark_prices["BTCUSD"] = 81269.8

    assert client.get_latest_price("BTCUSD") == 81270.5
    assert client.get_latest_price("btcusd") == 81270.5
    assert client.get_latest_mark_price("BTCUSD") == 81269.8
    assert client.get_latest_price("ETHUSD") is None
