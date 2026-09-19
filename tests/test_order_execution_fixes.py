import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.execution.delta import DeltaExchangeClient
from src.execution.order_manager import OrderManager, ActiveOrder, OrderState
from src.risk.risk_manager import RiskManager
from src.config import load_config


@pytest.fixture
def test_setup():
    config = load_config()
    delta_client = DeltaExchangeClient(live_trading=False)
    risk_manager = RiskManager(config.risk)
    account_manager = MagicMock()
    account_manager.equity = 10000.0
    account_manager.positions = {}
    order_manager = OrderManager(
        delta_client=delta_client,
        risk_manager=risk_manager,
        config=config,
        account_manager=account_manager,
    )
    return config, delta_client, risk_manager, account_manager, order_manager


@pytest.mark.asyncio
async def test_market_order_payload_structure(test_setup):
    config, delta_client, risk_manager, account_manager, order_manager = test_setup

    delta_client.live_trading = True
    delta_client._request = AsyncMock(return_value={"id": 12345, "state": "filled", "fill_price": "2650.00"})

    product = {
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_type": "perpetual_futures",
        "contract_value": 0.01,
        "tick_size": 0.05,
        "max_leverage": 150.0,
        "state": "live",
        "trading_status": "operational",
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
    }
    delta_client.get_product = AsyncMock(return_value=product)
    delta_client.set_leverage = AsyncMock(return_value={"success": True})

    res = await order_manager.validate_and_place_order(
        symbol="ETHUSD",
        side="LONG",
        price=2650.0,
        order_type="market",
        equity=100.0,
    )

    assert res is not None
    assert delta_client._request.called
    method, path = delta_client._request.call_args[0][:2]
    kwargs = delta_client._request.call_args[1]
    body = kwargs.get("body", {})

    assert method == "POST"
    assert path == "/v2/orders"
    assert body.get("order_type") == "market_order"
    assert "limit_price" not in body  # Market order must omit limit_price
    assert "bracket_stop_loss_price" in body
    assert body.get("stop_trigger_method") == "last_traded_price"

    # Verify active order was immediately FILLED in live mode
    ao = order_manager.active_order
    assert ao is not None
    assert ao.state == OrderState.FILLED
    assert ao.entry_price == 2650.0


@pytest.mark.asyncio
async def test_orderbook_crossing_price_for_limit_orders(test_setup):
    config, delta_client, risk_manager, account_manager, order_manager = test_setup

    # Mock orderbook with best bid and ask
    delta_client.get_orderbook = AsyncMock(return_value={
        "buy": [{"price": "2648.10", "size": 10}],
        "sell": [{"price": "2652.50", "size": 10}],
    })

    best_bid, best_ask = await delta_client.get_best_bid_ask("ETHUSD")
    assert best_bid == 2648.10
    assert best_ask == 2652.50

    delta_client.live_trading = False
    product = {
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_type": "perpetual_futures",
        "contract_value": 0.01,
        "tick_size": 0.05,
        "max_leverage": 150.0,
        "state": "live",
        "trading_status": "operational",
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
    }
    delta_client.get_product = AsyncMock(return_value=product)
    delta_client.set_leverage = AsyncMock(return_value={"success": True})

    # When placing limit order for BUY with use_orderbook_pricing=True, it should price at best_ask (2652.50)
    await order_manager.validate_and_place_order(
        symbol="ETHUSD",
        side="LONG",
        price=2645.0,  # Old/stale price
        order_type="limit",
        equity=100.0,
    )

    ao = order_manager.active_order
    assert ao is not None
    assert ao.entry_price == 2652.50  # Crossed spread to match ask


@pytest.mark.asyncio
async def test_sync_order_status_transitions_open_to_filled(test_setup):
    config, delta_client, risk_manager, account_manager, order_manager = test_setup
    delta_client.live_trading = True

    # Register an OPEN order
    order = ActiveOrder(
        order_id="test_ord_123",
        client_order_id="cid_123",
        state=OrderState.OPEN,
        symbol="ETHUSD",
        side="BUY",
        size=1.0,
        entry_price=2648.15,
        sl_price=2640.0,
        tp_price=2670.0,
        product_id=3136,
        created_at=time.time(),
    )
    order_manager.active_orders["test_ord_123"] = order

    delta_client.get_order = AsyncMock(return_value={
        "id": "test_ord_123",
        "state": "filled",
        "average_fill_price": "2650.25",
    })
    delta_client.get_product = AsyncMock(return_value={
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_value": 0.01,
        "max_leverage": 150.0,
    })

    updated = await order_manager.sync_order_status("test_ord_123")
    assert updated is not None
    assert updated.state == OrderState.FILLED
    assert updated.entry_price == 2650.25
    assert updated.last_event == "ORDER_FILLED"
    assert account_manager.update_paper_position.called


@pytest.mark.asyncio
async def test_check_unfilled_timeouts_cancels_and_releases_lock(test_setup):
    config, delta_client, risk_manager, account_manager, order_manager = test_setup
    delta_client.live_trading = True
    delta_client.cancel_order = AsyncMock(return_value={"success": True})

    # Register an OPEN order created 15 seconds ago
    order = ActiveOrder(
        order_id="stale_ord_456",
        client_order_id="cid_stale",
        state=OrderState.OPEN,
        symbol="ETHUSD",
        side="BUY",
        size=1.0,
        entry_price=2648.15,
        sl_price=2640.0,
        tp_price=2670.0,
        product_id=3136,
        created_at=time.time() - 15.0,
    )
    order_manager.active_orders["stale_ord_456"] = order
    order_manager.active_client_ids.add("cid_stale")

    # Has active position is True before timeout
    assert await order_manager.has_active_position(3136) is True

    # Check timeout with 10s threshold
    cancelled = await order_manager.check_unfilled_timeouts(timeout_seconds=10.0)
    assert "stale_ord_456" in cancelled
    assert delta_client.cancel_order.called
    assert "cid_stale" not in order_manager.active_client_ids
    assert "stale_ord_456" not in order_manager.active_orders

    # Has active position is now False (lock released)
    delta_client.get_position = AsyncMock(return_value=[])
    assert await order_manager.has_active_position(3136) is False


@pytest.mark.asyncio
async def test_sync_from_exchange_position_transitions_open_order(test_setup):
    config, delta_client, risk_manager, account_manager, order_manager = test_setup

    order = ActiveOrder(
        order_id="ord_789",
        client_order_id="cid_789",
        state=OrderState.OPEN,
        symbol="ETHUSD",
        side="BUY",
        size=2.0,
        entry_price=2648.0,
        sl_price=2640.0,
        tp_price=2670.0,
        product_id=3136,
        created_at=time.time(),
    )
    order_manager.active_orders["ord_789"] = order

    exchange_pos = {
        "product_id": 3136,
        "symbol": "ETHUSD",
        "size": "2.0",
        "entry_price": "2649.50",
    }

    synced = order_manager.sync_from_exchange_position(exchange_pos, "ETHUSD", 3136)
    assert synced is True
    assert order.state == OrderState.FILLED
    assert order.entry_price == 2649.50
    assert order.last_event == "EXCHANGE_POSITION_CONFIRMED"


@pytest.mark.asyncio
async def test_paper_market_order_executes_at_live_delta_price_not_1000():
    """Verify that paper market orders execute at Delta live market price and NEVER default to 1000.0."""
    client = DeltaExchangeClient({"live_trading": False})
    client.products_cache["ETHUSD"] = {
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_value": 0.01,
        "tick_size": 0.05,
    }
    client.products_by_id_cache[3136] = client.products_cache["ETHUSD"]
    client._latest_prices["ETHUSD"] = 2648.75
    client.get_best_bid_ask = AsyncMock(return_value=(None, None))

    res = await client.place_order(
        product_id=3136,
        side="buy",
        size=10.0,
        order_type="market",
        symbol="ETHUSD",
        market_price=2648.75,
        tick_size=0.05,
    )

    assert res["status"] == "filled"
    assert res["fill_price"] == 2648.75
    assert res["average_fill_price"] == 2648.75
    assert res["fill_price"] != 1000.0

    pos = client.paper_account.positions[3136]
    assert pos.entry_price == 2648.75
    assert pos.contract_value == 0.01
    assert pos.symbol == "ETHUSD"
    await client.close()


@pytest.mark.asyncio
async def test_paper_market_order_executes_at_orderbook_ask():
    """Verify that paper market buy orders cross the spread to Delta's Best Ask."""
    client = DeltaExchangeClient({"live_trading": False})
    client.products_cache["ETHUSD"] = {
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_value": 0.01,
        "tick_size": 0.05,
    }
    client.products_by_id_cache[3136] = client.products_cache["ETHUSD"]
    client.get_best_bid_ask = AsyncMock(return_value=(2648.70, 2648.85))

    res = await client.place_order(
        product_id=3136,
        side="buy",
        size=5.0,
        order_type="market",
        symbol="ETHUSD",
        tick_size=0.05,
    )

    assert res["fill_price"] == 2648.85  # Filled at best ask
    await client.close()


@pytest.mark.asyncio
async def test_close_paper_position_applies_contract_value_multiplier():
    """Verify closing a paper position uses contract_value so P&L is not 100x/1000x magnified."""
    client = DeltaExchangeClient({"live_trading": False, "paper_balance": 10000.0})
    client.products_cache["ETHUSD"] = {
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_value": 0.01,
        "tick_size": 0.05,
    }
    client.products_by_id_cache[3136] = client.products_cache["ETHUSD"]
    client._latest_prices["ETHUSD"] = 2648.75
    client.get_best_bid_ask = AsyncMock(return_value=(None, None))

    # Open LONG 100 contracts (1.0 ETH) at $2,648.75
    await client.place_order(
        product_id=3136,
        side="buy",
        size=100.0,
        order_type="market",
        symbol="ETHUSD",
        market_price=2648.75,
        tick_size=0.05,
    )

    # Close at $2,658.75 (+$10.00 price move on 1.0 ETH = +$10.00 P&L)
    pnl = client.close_paper_position(3136, close_price=2658.75)
    assert pnl == 10.0  # 100 contracts * 0.01 * $10 = $10.00 (NOT $1,000.00!)
    assert client.paper_account.equity == 10010.0
    await client.close()


@pytest.mark.asyncio
async def test_validate_and_place_order_paper_market_flow(test_setup):
    """Verify validate_and_place_order creates active order with exact Delta market fill price."""
    config, delta_client, risk_manager, account_manager, order_manager = test_setup
    delta_client.products_cache["ETHUSD"] = {
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_value": 0.01,
        "tick_size": 0.05,
        "contract_type": "perpetual_futures",
        "max_leverage": 150.0,
        "state": "live",
        "trading_status": "operational",
    }
    delta_client.products_by_id_cache[3136] = delta_client.products_cache["ETHUSD"]
    delta_client.get_product = AsyncMock(return_value=delta_client.products_cache["ETHUSD"])
    delta_client.set_leverage = AsyncMock(return_value={"success": True})
    delta_client._latest_prices["ETHUSD"] = 2648.75
    delta_client.get_best_bid_ask = AsyncMock(return_value=(None, None))

    res = await order_manager.validate_and_place_order(
        symbol="ETHUSD",
        side="LONG",
        price=2648.75,
        order_type="market",
        equity=100.0,
    )

    assert res is not None
    ao = order_manager.active_order
    assert ao is not None
    assert ao.entry_price == 2648.75
    assert ao.entry_price != 1000.0
    assert ao.state == OrderState.FILLED
    assert account_manager.update_paper_position.called
    call_kwargs = account_manager.update_paper_position.call_args.kwargs
    assert call_kwargs["price"] == 2648.75
