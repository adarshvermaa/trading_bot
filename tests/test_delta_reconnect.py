import pytest
import asyncio
import time
from unittest.mock import patch, AsyncMock
from src.execution.delta import DeltaExchangeClient, DeltaAPIError

@pytest.mark.asyncio
async def test_delta_paper_mode():
    client = DeltaExchangeClient({"live_trading": False})
    
    # In paper mode, it should not use HTTP requests
    res = await client.get_products()
    assert isinstance(res, list)
    
    order = await client.place_order(1, "buy", 1.0, "limit", 50000.0)
    assert order["status"] == "filled"
    
    pos = await client.get_position(1)
    assert len(pos) == 1
    assert pos[0]["size"] == 1.0
    assert pos[0]["entry_price"] == 50000.0
    await client.close()

@pytest.mark.asyncio
async def test_delta_rate_limiter():
    client = DeltaExchangeClient({"live_trading": True})
    
    # Initial budget is 20000, wait if it goes below 100 or weight
    client.rate_limiter.budget = 5
    
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        client.rate_limiter.last_reset = time.time() - 299  # Need to wait 1 second
        await client.rate_limiter.consume(10)
        mock_sleep.assert_called_once()
        assert client.rate_limiter.budget == 19990  # Reset and consumed 10
