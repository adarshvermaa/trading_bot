import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.execution.order_manager import OrderManager, OrderState, ActiveOrder

@pytest.mark.asyncio
async def test_cancel_pending_order():
    delta_mock = AsyncMock()
    risk_mock = MagicMock()
    om = OrderManager(delta_mock, risk_mock, {})
    
    om.active_orders["order_1"] = ActiveOrder("order_1", "client_1", OrderState.PENDING, "BTCUSD", "buy", 1.0, 50000, 49000, 52000)
    
    await om.on_structure_invalidated(is_filled=False, order_id="order_1", product_id=1, side="buy", size=1.0)
    
    delta_mock.cancel_order.assert_called_once_with("order_1", 1)
    assert om.active_orders["order_1"].state == OrderState.CANCELLED
