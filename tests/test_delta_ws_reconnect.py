import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
from src.data.delta_ws import DeltaWSClient, Candle


@pytest.mark.asyncio
async def test_delta_ws_duplicate_filtering_and_candle_closure():
    client = DeltaWSClient(["BTCUSD"])

    # First candle forming at t=1000
    msg1 = {
        "type": "candlestick_1m",
        "symbol": "BTCUSD",
        "resolution": "1m",
        "candle_start_time": 1000 * 1_000_000,
        "open": 81000.0,
        "high": 81100.0,
        "low": 80900.0,
        "close": 81050.0,
        "volume": 10.0,
    }
    await client._handle_message(json.dumps(msg1))
    assert len(client.store.get_candles("BTCUSD", "1m")) == 0  # Candle is still forming

    # Next minute arrives at t=1060 -> closes preceding candle at t=1000
    msg2 = {
        "type": "candlestick_1m",
        "symbol": "BTCUSD",
        "resolution": "1m",
        "candle_start_time": 1060 * 1_000_000,
        "open": 81050.0,
        "high": 81200.0,
        "low": 81000.0,
        "close": 81150.0,
        "volume": 15.0,
    }
    await client._handle_message(json.dumps(msg2))
    candles = client.store.get_candles("BTCUSD", "1m")
    assert len(candles) == 1
    assert candles[0].open == 81000.0
    assert candles[0].close == 81050.0
    assert candles[0].is_closed is True

    # Duplicate message with same start time t=1060 should not duplicate closed store
    await client._handle_message(json.dumps(msg2))
    assert len(client.store.get_candles("BTCUSD", "1m")) == 1


def test_delta_ws_stale_detection():
    client = DeltaWSClient(["BTCUSD"], stale_data_seconds=1.0)

    assert client.is_stale  # No messages yet

    client._last_message_time = time.time()
    assert not client.is_stale

    client._last_message_time = time.time() - 2.0
    assert client.is_stale
