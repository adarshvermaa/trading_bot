import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from src.data.binance_ws import BinanceWSClient, Candle

@pytest.mark.asyncio
async def test_binance_ws_duplicate_filtering():
    client = BinanceWSClient(["btcusdt"], {"btcusdt": "BTCUSD"})
    
    msg = {
        "data": {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 1000,
                "x": True,
                "i": "1m",
                "o": "100", "h": "110", "l": "90", "c": "105", "v": "10", "T": 1059
            }
        }
    }
    
    await client._handle_message(json.dumps(msg))
    assert len(client.store.get_candles("BTCUSD", "1m")) == 1
    
    # Process duplicate message
    await client._handle_message(json.dumps(msg))
    assert len(client.store.get_candles("BTCUSD", "1m")) == 1  # Still 1
    
def test_binance_ws_stale_detection():
    client = BinanceWSClient(["btcusdt"], {"btcusdt": "BTCUSD"}, stale_data_seconds=1.0)
    
    assert client.is_stale # No messages yet
    
    client._last_message_time = time.time()
    assert not client.is_stale
    
    client._last_message_time = time.time() - 2.0
    assert client.is_stale
