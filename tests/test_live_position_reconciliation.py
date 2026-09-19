import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from src.execution.order_manager import OrderManager, OrderState, ActiveOrder
from src.portfolio.account import AccountManager
from src.ui.dashboard import Dashboard


@pytest.mark.asyncio
async def test_sync_exchange_positions_detects_closed_live_position():
    """Verify that when Delta reports empty positions, a FILLED order transitions to CLOSED."""
    delta_mock = AsyncMock()
    delta_mock.live_trading = True
    risk_mock = MagicMock()
    account_mgr = AccountManager(is_paper=False, initial_paper_balance=10000.0)

    om = OrderManager(delta_mock, risk_mock, {}, account_manager=account_mgr)

    # Setup an active filled order
    ao = ActiveOrder(
        order_id="delta_999",
        client_order_id="c_999",
        state=OrderState.FILLED,
        symbol="BTCUSD",
        side="BUY",
        size=1.0,
        entry_price=60000.0,
        sl_price=59500.0,
        tp_price=61000.0,
        product_id=27,
    )
    om.active_orders["delta_999"] = ao
    om.active_client_ids.add("c_999")

    # Before reconciliation: active order is present
    assert om.active_order == ao

    # Delta reports empty positions (e.g. TP hit on Delta Exchange)
    reconciled = om.sync_exchange_positions([], current_price=61050.0)

    assert len(reconciled) == 1
    assert reconciled[0]["order_id"] == "delta_999"
    assert reconciled[0]["reason"] == "TP_HIT"
    assert reconciled[0]["pnl"] > 0
    risk_mock.record_trade_result.assert_called_once()

    # Active order tracking must now be completely clear
    assert om.active_order is None
    assert "delta_999" not in om.active_orders
    assert "c_999" not in om.active_client_ids

    # Last closed order is recorded
    assert om.last_closed_order is not None
    assert om.last_closed_order["order_id"] == "delta_999"
    assert om.last_closed_order["reason"] == "TP_HIT"


@pytest.mark.asyncio
async def test_has_active_position_releases_lock_on_exchange_closure():
    """Verify has_active_position releases immediately when Delta exchange positions list is empty."""
    delta_mock = AsyncMock()
    delta_mock.live_trading = True
    risk_mock = MagicMock()
    account_mgr = AccountManager(is_paper=False, initial_paper_balance=10000.0)

    om = OrderManager(delta_mock, risk_mock, {}, account_manager=account_mgr)

    ao = ActiveOrder(
        order_id="delta_888",
        client_order_id="c_888",
        state=OrderState.FILLED,
        symbol="ETHUSD",
        side="SELL",
        size=2.0,
        entry_price=3000.0,
        sl_price=3050.0,
        tp_price=2900.0,
        product_id=12,
    )
    om.active_orders["delta_888"] = ao
    om.active_client_ids.add("c_888")

    # Delta returns empty positions
    delta_mock.get_positions.return_value = []

    has_pos = await om.has_active_position()

    # Lock must be released!
    assert has_pos is False
    assert om.active_order is None
    assert om.last_closed_order is not None
    assert om.last_closed_order["order_id"] == "delta_888"


def test_get_execution_dict_formats_closed_and_idle_states():
    """Verify get_execution_dict returns proper status and styling indicators."""
    delta_mock = AsyncMock()
    risk_mock = MagicMock()
    om = OrderManager(delta_mock, risk_mock, {})

    # 1. Idle state (no orders ever)
    d_idle = om.get_execution_dict()
    assert d_idle["order_status"] == "IDLE (SCANNING)"
    assert d_idle["order_id"] == "--"

    # 2. Active filled state
    ao = ActiveOrder("ord_1", "c1", OrderState.FILLED, "BTCUSD", "BUY", 1.0, 65000.0, 64000.0, 67000.0)
    om.active_orders["ord_1"] = ao
    d_active = om.get_execution_dict()
    assert d_active["order_status"] == "FILLED"
    assert d_active["order_id"] == "ord_1"
    assert "$65,000.00" in d_active["fill_price"]

    # 3. Closed state (after trade finishes)
    om.active_orders.clear()
    om.last_closed_order = {
        "order_id": "ord_1",
        "symbol": "BTCUSD",
        "entry_price": 65000.0,
        "exit_price": 66500.0,
        "pnl": 1.50,
        "reason": "TP_HIT",
    }
    d_closed = om.get_execution_dict()
    assert d_closed["order_status"] == "CLOSED (IDLE)"
    assert d_closed["order_id"] == "ord_1"
    assert "TP_HIT" in d_closed["last_event"]
    assert "+$1.50" in d_closed["last_event"]


def test_dashboard_build_execution_panel_renders_cleanly():
    """Verify Dashboard._build_execution_panel renders CLOSED (IDLE) without errors."""
    dash = Dashboard(mode="LIVE")

    dash.update(
        account_data={},
        position_data={},
        signal_data={},
        risk_data={},
        execution_data={
            "order_id": "delta_777",
            "order_status": "CLOSED (IDLE)",
            "fill_price": "$65,000.00 -> $65,500.00",
            "fees": "--",
            "last_event": "TP_HIT (+$2.50)",
        },
    )

    panel = dash._build_execution_panel()
    assert panel.title == "EXECUTION"
    assert panel.border_style == "cyan"


@pytest.mark.asyncio
async def test_close_and_record_clears_active_tracking_and_sets_closed():
    """Verify close_and_record clears active tracking and sets last_closed_order."""
    delta_mock = AsyncMock()
    delta_mock.live_trading = False
    delta_mock.close_paper_position.return_value = 5.0
    risk_mock = MagicMock()
    account_mgr = AccountManager(is_paper=True, initial_paper_balance=10000.0)

    om = OrderManager(delta_mock, risk_mock, {}, account_manager=account_mgr)
    ao = ActiveOrder("paper_1", "c1", OrderState.FILLED, "BTCUSD", "BUY", 1.0, 60000.0, 59000.0, 62000.0, product_id=27)
    om.active_orders["paper_1"] = ao
    om.active_client_ids.add("c1")

    pnl = await om.close_and_record(reason="TP_HIT", close_price=62000.0, contract_value=0.001)

    assert pnl == 5.0
    assert om.active_order is None
    assert "paper_1" not in om.active_orders
    assert "c1" not in om.active_client_ids
    assert om.last_closed_order is not None
    assert om.last_closed_order["reason"] == "TP_HIT"
