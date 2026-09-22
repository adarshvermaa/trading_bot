import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.execution.delta import DeltaExchangeClient, DeltaAPIError
from src.execution.order_manager import OrderManager, OrderState
from src.data.delta_ws import DeltaWSClient
from src.config import load_config


@pytest.mark.asyncio
async def test_delta_get_products_filters_futures_only():
    """Verify get_products only returns and caches perpetual_futures and futures contracts."""
    client = DeltaExchangeClient({"live_trading": True})
    
    mock_products = [
        {"id": 1, "symbol": "BTCUSD", "contract_type": "perpetual_futures", "state": "live"},
        {"id": 2, "symbol": "ETHUSD", "contract_type": "perpetual_futures", "state": "live"},
        {"id": 3, "symbol": "BTC-28MAR25-100000-C", "contract_type": "call_options", "state": "live"},
        {"id": 4, "symbol": "ETH-28MAR25-3000-P", "contract_type": "put_options", "state": "live"},
        {"id": 5, "symbol": "BTC_USDT", "contract_type": "spot", "state": "live"},
        {"id": 6, "symbol": "BTCUSD_FUT", "contract_type": "futures", "state": "live"},
    ]
    
    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_products
        result = await client.get_products()
        
        # Verify API request parameters enforced futures contract types
        mock_req.assert_called_once_with(
            "GET", 
            "/v2/products", 
            params={"contract_types": "perpetual_futures,futures", "states": "live"}, 
            weight=3
        )
        
        # Verify output only includes futures contracts
        assert len(result) == 3
        returned_types = {p["contract_type"] for p in result}
        assert returned_types == {"perpetual_futures", "futures"}
        
        # Verify cache contains only futures
        assert "BTCUSD" in client.products_cache
        assert "ETHUSD" in client.products_cache
        assert "BTCUSD_FUT" in client.products_cache
        assert "BTC-28MAR25-100000-C" not in client.products_cache
        assert "BTC_USDT" not in client.products_cache
    
    await client.close()


@pytest.mark.asyncio
async def test_delta_get_product_rejects_non_futures():
    """Verify get_product raises DeltaAPIError for non-futures contracts."""
    client = DeltaExchangeClient({"live_trading": True})
    
    option_product = {
        "id": 99,
        "symbol": "BTC-OPTION",
        "contract_type": "call_options",
        "state": "live"
    }
    
    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        # First call for get_products returns empty, second call fetches symbol directly
        mock_req.side_effect = [[], option_product]
        
        with pytest.raises(DeltaAPIError, match="Only futures products are supported"):
            await client.get_product("BTC-OPTION")
            
    await client.close()


@pytest.mark.asyncio
async def test_delta_get_product_accepts_perpetual_futures():
    """Verify get_product returns properly structured product dict for perpetual_futures."""
    client = DeltaExchangeClient({"live_trading": True})
    
    futures_product = {
        "id": 27,
        "symbol": "BTCUSD",
        "contract_type": "perpetual_futures",
        "contract_value": "0.001",
        "tick_size": "0.5",
        "initial_margin": "0.005",
        "taker_commission_rate": "0.0005",
        "maker_commission_rate": "0.0002",
        "state": "live",
        "trading_status": "operational"
    }
    
    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = [futures_product]
        prod = await client.get_product("BTCUSD")
        
        assert prod["id"] == 27
        assert prod["symbol"] == "BTCUSD"
        assert prod["contract_type"] == "perpetual_futures"
        assert prod["contract_value"] == 0.001
        assert prod["tick_size"] == 0.5
        assert prod["max_leverage"] == 200.0
        assert prod["state"] == "live"
        assert prod["trading_status"] == "operational"
        
    await client.close()


@pytest.mark.asyncio
async def test_order_manager_rejects_non_futures():
    """Verify OrderManager rejects order validation if product is not a futures contract."""
    delta_mock = AsyncMock()
    risk_mock = MagicMock()
    om = OrderManager(delta_mock, risk_mock, {})
    
    # Return a spot product
    delta_mock.get_product.return_value = {
        "id": 100,
        "symbol": "BTC_USDT",
        "contract_type": "spot",
        "contract_value": 1.0,
        "tick_size": 0.01,
        "max_leverage": 1.0,
        "taker_commission_rate": 0.001,
        "maker_commission_rate": 0.001,
        "state": "live",
        "trading_status": "operational"
    }
    delta_mock.get_position.return_value = []
    
    res = await om.validate_and_place_order(
        symbol="BTC_USDT",
        side="buy",
        size=1.0,
        price=50000.0,
        leverage=1,
        client_order_id="test_client_spot"
    )
    
    assert res is None
    delta_mock.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_order_manager_accepts_futures():
    """Verify OrderManager accepts and places order for perpetual_futures contract."""
    delta_mock = AsyncMock()
    risk_mock = MagicMock()
    om = OrderManager(delta_mock, risk_mock, {})
    
    delta_mock.get_product.return_value = {
        "id": 27,
        "symbol": "BTCUSD",
        "contract_type": "perpetual_futures",
        "contract_value": 0.001,
        "tick_size": 0.5,
        "max_leverage": 200.0,
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
        "state": "live",
        "trading_status": "operational"
    }
    delta_mock.get_position.return_value = []
    delta_mock.get_wallet_balances.return_value = {"equity": 10000.0, "available_balance": 10000.0}
    delta_mock.place_order.return_value = {"id": "ord_123", "state": "open"}
    
    risk_mock.validate_leverage.return_value = True
    risk_mock.calculate_sl_tp.return_value = (49500.0, 51000.0)
    risk_mock.validate_trade.return_value = (True, "APPROVED")
    
    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="buy",
        size=10.0,
        price=50000.0,
        leverage=50,
        client_order_id="test_client_futures"
    )
    
    assert res is not None
    assert res["id"] == "ord_123"
    delta_mock.place_order.assert_called_once()


def test_delta_ws_subscription_payload():
    """Verify DeltaWSClient builds correct subscription payload for BTCUSD & ETHUSD."""
    client = DeltaWSClient(["BTCUSD", "ETHUSD"])
    payload = client._build_subscription_payload()
    assert payload["type"] == "subscribe"
    channels = payload["payload"]["channels"]
    ch_names = {c["name"] for c in channels}
    assert "candlestick_1m" in ch_names
    assert "candlestick_5m" in ch_names
    assert "candlestick_15m" in ch_names
    assert "v2/ticker" in ch_names
    for c in channels:
        assert c["symbols"] == ["BTCUSD", "ETHUSD"]


def test_strategy_universe_assets():
    """Verify strategy.yaml config contains BTCUSD, ETHUSD, and expanded universe assets."""
    config = load_config()
    universe = config.strategy.assets.universe
    assert "BTCUSD" in universe
    assert "ETHUSD" in universe
    assert len(universe) >= 2
