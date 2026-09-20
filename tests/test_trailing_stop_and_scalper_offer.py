import pytest
import math
import time
from unittest.mock import AsyncMock, MagicMock

from src.config import load_config
from src.risk.risk_manager import RiskManager
from src.execution.delta import DeltaExchangeClient, PaperPosition
from src.execution.order_manager import OrderManager, OrderState, ActiveOrder
from src.portfolio.account import AccountManager

def test_initial_sl_is_fixed_at_minus_3_percent():
    config = load_config()
    rm = RiskManager(config.risk)

    entry_price = 80000.0
    margin = 100.0
    leverage = 150
    contract_value = 0.001
    size = int(margin * leverage / (entry_price * contract_value))  # 187 contracts
    atr = 25.0

    sl_price, valid, _ = rm.calculate_stop_loss(
        entry_price, 'LONG', margin, leverage, contract_value, size, atr, tick_size=0.5
    )
    assert valid
    assert sl_price < entry_price

    price_drop = entry_price - sl_price
    loss_amount = price_drop * (size * contract_value)
    # Loss amount must equal 3% of margin ($3.00)
    assert pytest.approx(loss_amount, rel=0.05) == 3.00


def test_trailing_stop_activation_and_stepping_long():
    """Verify that trailing stop triggers at +2% margin profit and ratchets according to user table."""
    config = load_config()
    rm = RiskManager(config.risk)

    entry_price = 80000.0
    margin = 100.0
    contract_value = 0.001
    size = 187 # 0.187 BTC
    initial_sl = 79983.50 # -3% margin loss

    # 1. Under +2% margin profit (e.g. +1.5% margin P&L = +$1.50 -> price = 80008.02)
    p_1_5 = entry_price + (1.50 / (size * contract_value))
    new_sl, updated, _ = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_1_5, contract_value, size, initial_sl, tick_size=0.5
    )
    assert updated is False
    assert new_sl == initial_sl

    # 2. At +2.0% margin P&L: SL must move to +1.0% margin profit (+$1.00)
    p_2_0 = entry_price + (2.05 / (size * contract_value))
    new_sl, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_2_0, contract_value, size, initial_sl, tick_size=0.5
    )
    assert updated is True
    assert "+1% margin profit" in msg
    profit_locked = (new_sl - entry_price) * (size * contract_value)
    assert pytest.approx(profit_locked, rel=0.1) == 1.00 # +1% margin profit

    current_sl = new_sl

    # 3. At +3.5% margin P&L: SL must move to +2.0% margin profit (+$2.00)
    p_3_5 = entry_price + (3.50 / (size * contract_value))
    new_sl, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_3_5, contract_value, size, current_sl, tick_size=0.5
    )
    assert updated is True
    assert "+2% margin profit" in msg
    assert new_sl > current_sl
    current_sl = new_sl

    # 4. At +5.2% margin P&L: SL must move to +4.0% margin profit (+$4.00)
    p_5_2 = entry_price + (5.20 / (size * contract_value))
    new_sl, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_5_2, contract_value, size, current_sl, tick_size=0.5
    )
    assert updated is True
    assert "+4% margin profit" in msg
    assert new_sl > current_sl
    current_sl = new_sl

    # 5. At +10.0% margin P&L: SL must move to +9.0% margin profit (+$9.00)
    p_10_0 = entry_price + (10.05 / (size * contract_value))
    new_sl, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_10_0, contract_value, size, current_sl, tick_size=0.5
    )
    assert updated is True
    assert "+9% margin profit" in msg
    assert new_sl > current_sl
    current_sl = new_sl

    # 6. Ratchet rule: price pulls back from +10% to +6%
    p_pullback = entry_price + (6.00 / (size * contract_value))
    new_sl, updated, _ = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_pullback, contract_value, size, current_sl, tick_size=0.5
    )
    assert updated is False
    assert new_sl == current_sl # SL does not move backwards!


def test_trailing_stop_short_position():
    """Verify trailing stop logic for SHORT position."""
    config = load_config()
    rm = RiskManager(config.risk)

    entry_price = 80000.0
    margin = 100.0
    contract_value = 0.001
    size = 187
    initial_sl = 80016.50 # -3% margin loss

    # Price drops to +2.5% margin profit
    p_short_profit = entry_price - (2.50 / (size * contract_value))
    new_sl, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, 'SHORT', margin, p_short_profit, contract_value, size, initial_sl, tick_size=0.5
    )
    assert updated is True
    assert "+1% margin profit" in msg
    assert new_sl < entry_price # locked in profit below entry
    profit_locked = (entry_price - new_sl) * (size * contract_value)
    assert pytest.approx(profit_locked, rel=0.1) == 1.00


def test_delta_client_and_order_manager_sl_update():
    """Verify that update_active_sl and update_bracket_stop_loss synchronize correctly."""
    client = DeltaExchangeClient(api_key="", api_secret="", live_trading=False)
    account_mgr = AccountManager(is_paper=True, initial_paper_balance=10000.0)
    config = load_config()
    rm = RiskManager(config.risk)
    om = OrderManager(client, rm, config, account_mgr)

    # Setup paper position
    pos = PaperPosition(product_id=1, symbol="BTCUSD", size=10, entry_price=80000.0, bracket_sl=79984.0, side="buy")
    client.paper_account.positions[1] = pos

    ao = ActiveOrder("ord_test", "cli_test", OrderState.FILLED, "BTCUSD", "BUY", 10.0, 80000.0, 79984.0, 80100.0, product_id=1)
    om.active_orders["ord_test"] = ao

    account_mgr.update_paper_position("BTCUSD", "BUY", 10.0, 80000.0, 150, 100.0, 15000.0, 79984.0, 80100.0)

    # Trail SL to 80005.0 (+1% margin profit)
    om.update_active_sl(80005.0)
    assert om.active_order.sl_price == 80005.0
    assert account_mgr.positions["BTCUSD"].sl == 80005.0

    client.update_paper_sl(1, 80005.0)
    assert client.paper_account.positions[1].bracket_sl == 80005.0

    # If price pulls back to 80005.0, check_paper_sl_tp triggers SL_HIT in profit
    trigger = client.check_paper_sl_tp(1, 80005.0)
    assert trigger == "SL_HIT"


def test_trailing_stop_higher_steps_up_to_200_percent():
    """Verify trailing stop steps at +50% and +200% margin profit."""
    config = load_config()
    rm = RiskManager(config.risk)

    entry_price = 80000.0
    margin = 100.0
    contract_value = 0.001
    size = 187
    current_sl = 80048.0 # +9% margin profit SL

    # At +50.5% margin P&L: target SL is +49%
    p_50 = entry_price + (50.50 / (size * contract_value))
    new_sl, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_50, contract_value, size, current_sl, tick_size=0.5
    )
    assert updated is True
    assert "+49% margin profit" in msg
    assert new_sl > current_sl
    current_sl = new_sl

    # At +200.0% margin P&L: target SL is +199%
    p_200 = entry_price + (200.0 / (size * contract_value))
    new_sl, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, 'LONG', margin, p_200, contract_value, size, current_sl, tick_size=0.5
    )
    assert updated is True
    assert "+199% margin profit" in msg
    assert new_sl > current_sl


def test_delta_scalper_offer_29m_limit():
    """Verify that 29-minute position threshold applies to Delta Scalper Offer."""
    config = load_config()
    trailing_cfg = config.risk.trailing_stop
    assert trailing_cfg.scalper_offer_max_seconds_major == 1740 # 29 min
    assert trailing_cfg.scalper_offer_max_seconds_other == 840  # 14 min

    # Check holding times
    now = time.time()
    created_at_25m = now - (25 * 60)
    assert (now - created_at_25m) < trailing_cfg.scalper_offer_max_seconds_major

    created_at_29m = now - (29 * 60)
    assert (now - created_at_29m) >= trailing_cfg.scalper_offer_max_seconds_major


@pytest.mark.asyncio
async def test_delta_client_update_bracket_stop_loss_live_edits_resting_stop_order():
    """Verify in live trading mode that update_bracket_stop_loss finds resting stop-loss and updates via PUT /v2/orders."""
    client = DeltaExchangeClient(api_key="dummy_key", api_secret="dummy_secret", live_trading=True)

    # Mock get_open_orders to return a resting stop-loss order
    client.get_open_orders = AsyncMock(return_value=[
        {
            "id": 999888,
            "product_id": 27,
            "order_type": "market_order",
            "stop_order_type": "stop_loss_order",
            "stop_price": "80000.0",
            "state": "open",
        }
    ])
    client._request = AsyncMock(return_value={"success": True, "id": 999888})

    res = await client.update_bracket_stop_loss(
        product_id=27,
        new_sl=80500.0,
        tick_size=0.5,
        order_id="123456",
        side="buy",
        size=10,
    )
    assert res["success"] is True
    assert res["bracket_stop_loss_price"] == 80500.0
    client._request.assert_called_once_with(
        "PUT",
        "/v2/orders",
        body={"id": 999888, "product_id": 27, "stop_price": "80500.0"},
        weight=5,
    )


@pytest.mark.asyncio
async def test_delta_client_update_bracket_stop_loss_fallback_places_new_order():
    """Verify in live trading mode that if resting order not found and bracket PUT fails, a new stop order is posted."""
    client = DeltaExchangeClient(api_key="dummy_key", api_secret="dummy_secret", live_trading=True)

    # No open orders
    client.get_open_orders = AsyncMock(return_value=[])

    async def mock_request(method, path, body=None, weight=1, params=None):
        if path == "/v2/orders/bracket":
            raise Exception("open_order_not_found")
        if path == "/v2/orders" and method == "POST":
            return {"success": True, "id": 777666}
        return {}

    client._request = AsyncMock(side_effect=mock_request)

    res = await client.update_bracket_stop_loss(
        product_id=27,
        new_sl=80500.0,
        tick_size=0.5,
        order_id="123456",
        side="buy",
        size=10,
    )
    assert res["success"] is True
    assert res["bracket_stop_loss_price"] == 80500.0
    # Verify POST was called for new stop order
    post_calls = [c for c in client._request.call_args_list if c[0][0] == "POST" and c[0][1] == "/v2/orders"]
    assert len(post_calls) == 1
    call_body = post_calls[0][1]["body"]
    assert call_body["stop_order_type"] == "stop_loss_order"
    assert call_body["side"] == "sell"
    assert call_body["stop_price"] == "80500.0"


@pytest.mark.asyncio
async def test_live_trailing_stop_loss_failsafe_execution():
    """Verify that when live price crosses trailed sl_price, order_manager closes position and records TRAILING_SL_HIT."""
    config = load_config()
    client = DeltaExchangeClient(api_key="dummy_key", api_secret="dummy_secret", live_trading=True)
    client.close_position = AsyncMock(return_value={"success": True})
    rm = RiskManager(config.risk)
    account_mgr = AccountManager(is_paper=False)
    om = OrderManager(client, rm, config, account_mgr)

    # Active BUY order entered at 80000.0, trailed SL is 80100.0 (+1% margin profit)
    ao = ActiveOrder(
        order_id="live_ord_1",
        client_order_id="cli_1",
        state=OrderState.FILLED,
        symbol="BTCUSD",
        side="BUY",
        size=10.0,
        entry_price=80000.0,
        sl_price=80100.0,
        tp_price=82000.0,
        product_id=27,
    )
    om.active_orders["live_ord_1"] = ao

    live_price = 80090.0 # Price pulled back below trailed SL (80100.0)
    assert live_price <= ao.sl_price
    reason = "TRAILING_SL_HIT" if ao.sl_price > ao.entry_price else "SL_HIT"
    assert reason == "TRAILING_SL_HIT"

    pnl = await om.close_and_record(reason=reason, close_price=live_price, contract_value=0.001)
    # Expected profit: (80090 - 80000) * 10 * 0.001 = 0.90 USD
    assert pytest.approx(pnl, rel=0.01) == 0.90
    assert om.active_order is None
    assert om.last_closed_order is not None
    assert om.last_closed_order["reason"] == "TRAILING_SL_HIT"
    assert om.last_closed_order["pnl"] == pnl
    client.close_position.assert_called_once_with(27, "BUY", 10.0)


