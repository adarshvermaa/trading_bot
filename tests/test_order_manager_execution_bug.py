import pytest
from unittest.mock import AsyncMock, MagicMock
from src.execution.order_manager import OrderManager, OrderState, ActiveOrder

@pytest.mark.asyncio
async def test_order_manager_maps_long_to_buy():
    delta_mock = AsyncMock()
    delta_mock.live_trading = True
    delta_mock.get_product.return_value = {
        "id": 1,
        "symbol": "BTCUSD",
        "state": "live",
        "trading_status": "operational",
        "contract_type": "perpetual_futures",
        "max_leverage": 100,
        "contract_value": 0.001,
        "tick_size": 0.5,
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
    }
    delta_mock.place_order.return_value = {"id": "delta_101", "state": "open"}
    delta_mock.get_positions.return_value = []
    delta_mock.set_leverage.return_value = {"success": True}

    risk_mock = MagicMock()
    risk_mock.validate_leverage.return_value = (True, 10, "VALID")
    risk_mock.calculate_sl_tp.return_value = (49500.0, 51000.0)
    risk_mock.validate_trade.return_value = (True, "APPROVED")

    om = OrderManager(delta_mock, risk_mock, {})
    
    # Place order with side="LONG"
    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="LONG",
        price=50000.0,
        size=10.0,
        leverage=10,
        equity=10000.0,
        atr=50.0,
    )

    assert res is not None
    # Verify Delta client was called with side="buy", NOT "long"
    delta_mock.place_order.assert_called_once()
    call_kwargs = delta_mock.place_order.call_args[1]
    assert call_kwargs["side"] == "buy"

@pytest.mark.asyncio
async def test_order_manager_maps_short_to_sell():
    delta_mock = AsyncMock()
    delta_mock.live_trading = True
    delta_mock.get_product.return_value = {
        "id": 1,
        "symbol": "BTCUSD",
        "state": "live",
        "trading_status": "operational",
        "contract_type": "perpetual_futures",
        "max_leverage": 100,
        "contract_value": 0.001,
        "tick_size": 0.5,
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
    }
    delta_mock.place_order.return_value = {"id": "delta_102", "state": "open"}
    delta_mock.get_positions.return_value = []
    delta_mock.set_leverage.return_value = {"success": True}

    risk_mock = MagicMock()
    risk_mock.validate_leverage.return_value = (True, 10, "VALID")
    risk_mock.calculate_sl_tp.return_value = (50500.0, 49000.0)
    risk_mock.validate_trade.return_value = (True, "APPROVED")

    om = OrderManager(delta_mock, risk_mock, {})
    
    # Place order with side="SHORT"
    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="SHORT",
        price=50000.0,
        size=10.0,
        leverage=10,
        equity=10000.0,
        atr=50.0,
    )

    assert res is not None
    # Verify Delta client was called with side="sell", NOT "short"
    delta_mock.place_order.assert_called_once()
    call_kwargs = delta_mock.place_order.call_args[1]
    assert call_kwargs["side"] == "sell"
