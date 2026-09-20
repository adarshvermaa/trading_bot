import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from src.execution.order_manager import OrderManager, OrderState
from src.execution.delta import DeltaExchangeClient
from src.risk.risk_manager import RiskManager
from src.strategy.signals import Signal
from src.strategy.structure import StructureAnalysis
from src.config import load_config

@pytest.mark.asyncio
async def test_end_to_end_order_placement_margin_sl_tp():
    """Verify that when a scalp signal triggers, OrderManager uses RiskManager to enforce 150x leverage and exact margin-based SL/TP."""
    config = load_config()
    delta_client = DeltaExchangeClient(api_key="", api_secret="", live_trading=False)
    delta_client.set_leverage = AsyncMock(return_value={"leverage": 150})
    risk_manager = RiskManager(config.risk)
    order_manager = OrderManager(delta_client, risk_manager, config=config.strategy)

    # Setup mock product for BTCUSD with 150x max leverage
    btc_product = {
        "id": 1,
        "symbol": "BTCUSD",
        "contract_type": "perpetual_futures",
        "state": "live",
        "trading_status": "operational",
        "contract_value": "0.001",
        "tick_size": "0.5",
        "max_leverage": "150",
        "taker_commission_rate": "0.0005",
        "maker_commission_rate": "0.0002",
    }
    delta_client.products_cache["BTCUSD"] = btc_product

    # Mock order placement on delta_client
    delta_client.place_order = AsyncMock(return_value={
        "id": 101,
        "product_id": 1,
        "symbol": "BTCUSD",
        "side": "buy",
        "size": 150,
        "order_type": "market_order",
        "state": "filled",
        "price": "80000.0",
        "average_fill_price": "80000.0",
    })

    # Validate and place order with sl_price=None, tp_price=None
    res = await order_manager.validate_and_place_order(
        symbol="BTCUSD",
        side="LONG",
        entry_price=80000.0,
        sl_price=None,
        tp_price=None,
        atr=25.0,
        spread_bps=1.0,
        equity=100.0,
    )

    assert res is not None
    active = order_manager.active_order
    assert active is not None
    assert active.symbol == "BTCUSD"
    assert active.side == "LONG"
    assert active.entry_price == 80000.0

    # Leverage must be set to configured high leverage (25x)
    delta_client.set_leverage.assert_called_once_with(1, config.risk.leverage.high_leverage_value)

    # Stop loss must be below entry for LONG and take profit above entry
    assert active.sl_price < 80000.0
    assert active.tp_price > 80000.0

    # Verify place_order received the calculated bracket_stop_loss_price and bracket_take_profit_price
    delta_client.place_order.assert_called_once()
    call_kwargs = delta_client.place_order.call_args[1]
    assert call_kwargs["bracket_stop_loss_price"] == active.sl_price
    assert call_kwargs["bracket_take_profit_price"] == active.tp_price

    price_diff_sl = 80000.0 - active.sl_price
    price_diff_tp = active.tp_price - 80000.0

    assert price_diff_sl > 0
    assert price_diff_tp > 0
    expected_ratio = config.risk.take_profit.target_pct_of_margin / config.risk.stop_loss.max_loss_pct_of_margin
    assert pytest.approx(price_diff_tp / price_diff_sl, rel=0.1) == expected_ratio
