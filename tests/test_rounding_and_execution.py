import pytest
import math
from unittest.mock import AsyncMock, patch
from src.risk.risk_manager import RiskManager, round_to_tick, format_price
from src.execution.delta import DeltaExchangeClient
from src.config import RiskConfig


def test_round_to_tick_btc():
    tick_size = 0.5
    # Nearest
    assert round_to_tick(64250.24, tick_size, 'NEAREST') == 64250.0
    assert round_to_tick(64250.26, tick_size, 'NEAREST') == 64250.5
    # Directional
    assert round_to_tick(64250.26, tick_size, 'DOWN') == 64250.0
    assert round_to_tick(64250.24, tick_size, 'UP') == 64250.5


def test_round_to_tick_eth():
    tick_size = 0.05
    assert round_to_tick(3450.123, tick_size, 'NEAREST') == 3450.10
    assert round_to_tick(3450.129, tick_size, 'NEAREST') == 3450.15
    assert round_to_tick(3450.14, tick_size, 'DOWN') == 3450.10
    assert round_to_tick(3450.11, tick_size, 'UP') == 3450.15


def test_round_to_tick_micro_assets():
    # DOGEUSD tick_size = 0.000001
    doge_tick = 0.000001
    assert round_to_tick(0.1234567, doge_tick, 'NEAREST') == 0.123457
    assert round_to_tick(0.1234564, doge_tick, 'DOWN') == 0.123456
    assert round_to_tick(0.1234561, doge_tick, 'UP') == 0.123457

    # SOLUSD tick_size = 0.0001
    sol_tick = 0.0001
    assert round_to_tick(182.45678, sol_tick, 'NEAREST') == 182.4568

    # TRXUSD tick_size = 0.00001
    trx_tick = 0.00001
    assert round_to_tick(0.154321, trx_tick, 'NEAREST') == 0.15432


def test_format_price_no_scientific_notation():
    assert format_price(64250.5, 0.5) == "64250.5"
    assert format_price(3450.1, 0.05) == "3450.10"
    assert format_price(0.123456, 0.000001) == "0.123456"
    assert format_price(0.000005, 0.000001) == "0.000005"


def test_calculate_sl_tp_with_rounding_long():
    rm = RiskManager(RiskConfig())
    entry = 64250.33
    tick_size = 0.5
    
    sl_price, tp_price = rm.calculate_sl_tp(
        side='LONG',
        entry_price=entry,
        margin=1000.0,
        leverage=50,
        contract_value=0.001,
        size=10,
        atr=50.0,
        tick_size=tick_size
    )
    
    assert sl_price < entry
    assert tp_price > entry
    # Must be exact multiple of tick_size
    assert math.isclose(round_to_tick(sl_price, tick_size), sl_price)
    assert math.isclose(round_to_tick(tp_price, tick_size), tp_price)
    # Check that decimal is .0 or .5
    assert (sl_price * 10) % 5 == 0
    assert (tp_price * 10) % 5 == 0


def test_calculate_sl_tp_with_rounding_short():
    rm = RiskManager(RiskConfig())
    entry = 3450.123
    tick_size = 0.05
    
    sl_price, tp_price = rm.calculate_sl_tp(
        side='SHORT',
        entry_price=entry,
        margin=1000.0,
        leverage=50,
        contract_value=0.01,
        size=10,
        atr=20.0,
        tick_size=tick_size
    )
    
    assert sl_price > entry
    assert tp_price < entry
    assert math.isclose(round_to_tick(sl_price, tick_size), sl_price)
    assert math.isclose(round_to_tick(tp_price, tick_size), tp_price)


@pytest.mark.asyncio
async def test_place_order_formats_bracket_prices_with_tick_size():
    client = DeltaExchangeClient({"live_trading": True})
    
    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = {"id": "ord_999", "state": "open"}
        
        res = await client.place_order(
            product_id=27,
            side="buy",
            size=5.0,
            order_type="limit",
            limit_price=64250.33,
            bracket_stop_loss_price=64000.22,
            bracket_take_profit_price=64750.77,
            client_order_id="test_bracket_rounding",
            tick_size=0.5
        )
        
        mock_req.assert_called_once()
        call_args = mock_req.call_args
        body = call_args.kwargs["body"]
        
        # Prices must be string-formatted multiples of tick_size (0.5), and market_order omits limit_price
        assert body["order_type"] == "market_order"
        assert "limit_price" not in body
        assert body["bracket_stop_loss_price"] == "64000.0"
        assert body["bracket_take_profit_price"] == "64751.0"
        assert body["product_id"] == 27
        assert body["side"] == "buy"
        
    await client.close()
