import pytest
from unittest.mock import AsyncMock, patch
from src.execution.delta import DeltaExchangeClient, normalize_delta_symbol, DELTA_DEFAULT_PRODUCTS
from src.execution.order_manager import OrderManager, OrderState
from src.config import load_config
from src.risk.risk_manager import RiskManager
from src.portfolio.account import AccountManager


@pytest.mark.asyncio
async def test_symbol_normalization():
    """Verify symbol normalization maps BTC and ETH variants to official Delta symbols."""
    assert normalize_delta_symbol("BTC") == "BTCUSD"
    assert normalize_delta_symbol("BTCUSDT") == "BTCUSD"
    assert normalize_delta_symbol("btcusdt") == "BTCUSD"
    assert normalize_delta_symbol("BTCUSD") == "BTCUSD"
    assert normalize_delta_symbol("ETH") == "ETHUSD"
    assert normalize_delta_symbol("ETHUSDT") == "ETHUSD"
    assert normalize_delta_symbol("ethusd") == "ETHUSD"
    # Unrelated symbol preserved
    assert normalize_delta_symbol("SOLUSD") == "SOLUSD"


@pytest.mark.asyncio
async def test_fetch_target_products_btc_eth_only():
    """Verify fetch_target_products retrieves only BTC and ETH with official Delta specs."""
    client = DeltaExchangeClient({"live_trading": True})

    prods = await client.fetch_target_products(["BTCUSD", "ETHUSD"])
    assert "BTCUSD" in prods
    assert "ETHUSD" in prods
    assert len(prods) == 2

    btc = prods["BTCUSD"]
    assert btc["id"] == 27
    assert btc["symbol"] == "BTCUSD"
    assert btc["contract_value"] == 0.001
    assert btc["tick_size"] == 0.5
    assert btc["contract_type"] == "perpetual_futures"

    eth = prods["ETHUSD"]
    assert eth["id"] == 3136
    assert eth["symbol"] == "ETHUSD"
    assert eth["contract_value"] == 0.01
    assert eth["tick_size"] == 0.05
    assert eth["contract_type"] == "perpetual_futures"

    await client.close()


@pytest.mark.asyncio
async def test_order_placement_payload_delta_docs_compliance():
    """Verify live order placement payload conforms exactly to official Delta v2 API documentation."""
    client = DeltaExchangeClient({
        "live_trading": True,
        "delta_api_key": "test_key",
        "delta_api_secret": "test_secret"
    })

    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = {"id": "delta_ord_123", "status": "open"}

        res = await client.place_order(
            product_id=27,
            side="buy",
            size=15.0,
            order_type="market",
            bracket_stop_loss_price=84500.5,
            bracket_take_profit_price=89000.0,
            client_order_id="my_scalp_order_001",
            tick_size=0.5,
            symbol="BTCUSD"
        )

        mock_req.assert_called_once()
        call_method, call_path = mock_req.call_args[0][0], mock_req.call_args[0][1]
        call_body = mock_req.call_args[1].get("body", {})

        assert call_method == "POST"
        assert call_path == "/v2/orders"

        # Check required fields from official Delta v2 order documentation
        assert call_body["product_id"] == 27
        assert call_body["product_symbol"] == "BTCUSD"
        assert call_body["side"] == "buy"
        assert call_body["size"] == 15
        assert isinstance(call_body["size"], int)
        assert call_body["order_type"] == "market_order"
        assert call_body["time_in_force"] == "gtc"
        assert call_body["reduce_only"] is False
        assert call_body["client_order_id"] == "my_scalp_order_001"
        assert call_body["bracket_stop_loss_price"] == "84500.5"
        assert call_body["bracket_take_profit_price"] == "89000.0"
        assert call_body["stop_trigger_method"] == "last_traded_price"
        assert call_body["bracket_stop_trigger_method"] == "last_traded_price"
        # Market order must omit limit_price
        assert "limit_price" not in call_body

    await client.close()


@pytest.mark.asyncio
async def test_update_bracket_stop_loss_delta_docs_compliance():
    """Verify PUT /v2/orders/bracket matches Delta v2 specifications."""
    client = DeltaExchangeClient({
        "live_trading": True,
        "delta_api_key": "test_key",
        "delta_api_secret": "test_secret"
    })

    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = {"success": True}

        res = await client.update_bracket_stop_loss(
            product_id=27,
            new_sl=85123.5,
            tick_size=0.5,
            order_id="998877"
        )

        assert res["success"] is True
        # First call is GET /v2/orders (check resting stop orders), second is PUT /v2/orders/bracket fallback
        put_call = next(c for c in mock_req.call_args_list if c[0][0] == "PUT")
        call_method, call_path = put_call[0][0], put_call[0][1]
        call_body = put_call[1].get("body", {})

        assert call_method == "PUT"
        assert call_path == "/v2/orders/bracket"
        assert call_body["product_id"] == 27
        assert call_body["product_symbol"] == "BTCUSD"
        assert call_body["bracket_stop_loss_price"] == "85123.5"
        assert call_body["bracket_stop_trigger_method"] == "last_traded_price"
        assert call_body["id"] == 998877

    await client.close()


@pytest.mark.asyncio
async def test_signal_to_delta_order_manager_execution():
    """Verify analysis signal flows cleanly into Delta execution with correct product sizing."""
    config = load_config()
    delta_client = DeltaExchangeClient({
        "live_trading": False,
        "paper_balance": 10000.0
    })
    delta_client.get_orderbook_imbalance = AsyncMock(return_value=0.0)
    risk_manager = RiskManager(config.risk)
    account_manager = AccountManager(is_paper=True, paper_balance=10000.0)
    order_manager = OrderManager(
        delta_client=delta_client,
        risk_manager=risk_manager,
        config=config,
        account_manager=account_manager
    )

    # Signal generated from Binance for BTC
    order_res = await order_manager.validate_and_place_order(
        symbol="BTCUSD",
        side="LONG",
        price=85000.0,
        equity=10000.0,
        order_type="market"
    )

    assert order_res is not None
    assert order_res["status"] == "filled"
    assert order_res["product_id"] == 27
    assert order_res["side"] == "buy"

    # Active order tracked
    ao = order_manager.active_order
    assert ao is not None
    assert ao.symbol == "BTCUSD"
    assert ao.product_id == 27
    assert ao.state == OrderState.FILLED

    # Check portfolio position has correct contract value multiplier (0.001)
    pos = account_manager.positions.get("BTCUSD")
    assert pos is not None
    assert pos.leverage == config.risk.leverage.high_leverage_value

    await delta_client.close()


@pytest.mark.asyncio
async def test_market_order_enforced_even_if_limit_requested():
    """Verify that Delta API strictly enforces market_order and omits limit_price even if limit is passed."""
    client = DeltaExchangeClient({
        "live_trading": True,
        "delta_api_key": "test_key",
        "delta_api_secret": "test_secret"
    })

    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = {"id": "delta_ord_456", "status": "open"}

        res = await client.place_order(
            product_id=27,
            side="buy",
            size=10.0,
            order_type="limit",  # Even if caller passes limit
            limit_price=85000.0,
            bracket_stop_loss_price=82450.0,
            bracket_take_profit_price=87550.0,
            client_order_id="limit_override_test",
            tick_size=0.5,
            symbol="BTCUSD"
        )

        mock_req.assert_called_once()
        call_body = mock_req.call_args[1].get("body", {})

        # Must strictly be market_order with no limit_price
        assert call_body["order_type"] == "market_order"
        assert "limit_price" not in call_body
        assert call_body["product_symbol"] == "BTCUSD"
        assert call_body["product_id"] == 27

    await client.close()
