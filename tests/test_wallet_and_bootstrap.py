import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from src.data.delta_ws import DeltaWSClient, Candle
from src.execution.delta import DeltaExchangeClient
from src.portfolio.account import AccountManager


@pytest.mark.asyncio
async def test_delta_bootstrap_historical_candles():
    """Verify bootstrap_historical_candles fetches and stores candles from Delta for all timeframes."""
    symbols = ["BTCUSD"]
    client = DeltaWSClient(symbols=symbols)

    # Mock _fetch_delta_klines to simulate REST response
    fake_candle_count = 100
    with patch.object(client, "_fetch_delta_klines", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = fake_candle_count
        count = await client.bootstrap_historical_candles(limit=100)
        
        # 1 symbol * 3 timeframes (1m, 5m, 15m) = 3 calls
        assert mock_fetch.call_count == 3
        assert count == 300
        assert not client.is_stale
        assert client.new_candle_event.is_set()


@pytest.mark.asyncio
async def test_delta_get_wallet_balances_multi_asset():
    """Verify DeltaExchangeClient aggregates multi-asset wallet balances (USD, USDT, etc.)."""
    client = DeltaExchangeClient({"live_trading": True})
    
    mock_balances = [
        {
            "asset_symbol": "USD",
            "balance": "0.150678615",
            "available_balance": "0.150678615",
            "position_margin": "0",
            "order_margin": "0"
        },
        {
            "asset_symbol": "USDT",
            "balance": "50.0",
            "available_balance": "40.0",
            "position_margin": "10.0",
            "order_margin": "0"
        },
        {
            "asset_symbol": "BTC",
            "balance": "0",
            "available_balance": "0",
            "position_margin": "0",
            "order_margin": "0"
        }
    ]
    
    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_balances
        bals = await client.get_wallet_balances()
        
        assert abs(bals["equity"] - 50.150678615) < 1e-4
        assert abs(bals["available_balance"] - 40.150678615) < 1e-4
        assert abs(bals["position_margin"] - 10.0) < 1e-4
        assert bals["available_margin"] == bals["available_balance"]


@pytest.mark.asyncio
async def test_account_manager_update_from_exchange():
    """Verify AccountManager correctly maps available_balance and calculates reserve."""
    account = AccountManager(is_paper=False)
    assert account.equity == 0.0
    assert account.available_margin == 0.0

    balances = {
        "equity": 100.0,
        "available_balance": 80.0,
        "position_margin": 20.0,
    }
    positions = [
        {
            "product_symbol": "BTCUSD",
            "size": 1,
            "entry_price": 80000.0,
            "leverage": 20,
            "margin": 20.0,
            "notional": 400.0,
            "unrealized_pnl": 5.0,
        }
    ]

    account.update_from_exchange(balances, positions)

    assert account.equity == 100.0
    assert account.available_margin == 80.0
    assert account.used_margin == 20.0
    assert account.reserve == 20.0  # 20% of 100
    assert "BTCUSD" in account.positions
    pos = account.positions["BTCUSD"]
    assert pos.side == "LONG"
    assert pos.entry_price == 80000.0
    assert pos.unrealized_pnl == 5.0
