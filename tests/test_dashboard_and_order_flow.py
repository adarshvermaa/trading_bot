import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.execution.order_manager import OrderManager, OrderState, ActiveOrder
from src.portfolio.account import AccountManager
from src.ui.dashboard import Dashboard
from src.risk.risk_manager import RiskManager
from src.config import RiskConfig


@pytest.mark.asyncio
async def test_order_manager_active_order_property():
    """Verify active_order property returns the latest active order."""
    delta_mock = AsyncMock()
    risk_mock = MagicMock()
    om = OrderManager(delta_mock, risk_mock, {})

    assert om.active_order is None

    ao1 = ActiveOrder("ord_1", "c1", OrderState.FILLED, "BTCUSD", "BUY", 1.0, 60000.0, 59000.0, 62000.0)
    om.active_orders["ord_1"] = ao1
    assert om.active_order == ao1

    ao2 = ActiveOrder("ord_2", "c2", OrderState.CANCELLED, "ETHUSD", "BUY", 1.0, 3000.0, 2900.0, 3200.0)
    om.active_orders["ord_2"] = ao2
    # Should skip CANCELLED and return ord_1
    assert om.active_order == ao1


@pytest.mark.asyncio
async def test_order_manager_has_active_position_no_args():
    """Verify has_active_position works without arguments."""
    delta_mock = AsyncMock()
    risk_mock = MagicMock()
    om = OrderManager(delta_mock, risk_mock, {})

    delta_mock.get_positions.return_value = []
    has_pos = await om.has_active_position()
    assert has_pos is False

    delta_mock.get_positions.return_value = [{"product_id": 27, "size": 1.0, "entry_price": 60000.0}]
    has_pos = await om.has_active_position()
    assert has_pos is True


@pytest.mark.asyncio
async def test_order_manager_validate_and_place_order_kwargs_flow():
    """Verify validate_and_place_order handles keyword arguments from main.py."""
    delta_mock = AsyncMock()
    risk_mock = MagicMock()
    account_mgr = AccountManager(is_paper=True, initial_paper_balance=10000.0)
    om = OrderManager(delta_mock, risk_mock, {}, account_manager=account_mgr)

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
        "trading_status": "operational",
    }
    delta_mock.get_position.return_value = []
    delta_mock.get_positions.return_value = []
    delta_mock.get_wallet_balances.return_value = {"equity": 10000.0, "available_balance": 10000.0}
    delta_mock.live_trading = False
    delta_mock.place_order.return_value = {"id": "paper_123", "status": "filled"}

    risk_mock.validate_leverage.return_value = (True, 20, "VALID")
    risk_mock.calculate_position_size.return_value = (1, 50.0, 1000.0)
    risk_mock.calculate_sl_tp.return_value = (59500.0, 61000.0)
    risk_mock.validate_trade.return_value = (True, "APPROVED")

    # Call with main.py signature
    res = await om.validate_and_place_order(
        symbol="BTCUSD",
        side="LONG",
        entry_price=60000.0,
        sl_price=59500.0,
        tp_price=61000.0,
        atr=500.0,
        spread_bps=0.0,
        equity=account_mgr.equity,
    )

    assert res is not None
    assert res["id"] == "paper_123"
    assert om.active_order is not None
    assert om.active_order.order_id == "paper_123"
    assert "BTCUSD" in account_mgr.positions


def test_paper_account_manager_dashboard_dict():
    """Verify paper account dashboard metrics start populated."""
    account = AccountManager(is_paper=True, initial_paper_balance=10000.0)
    d = account.to_dashboard_dict()
    assert d["equity"] == 10000.0
    assert d["available_margin"] == 10000.0
    assert d["reserve_pct"] == 20.0
    assert d["daily_loss_limit"] == -300.0


def test_dashboard_renders_with_initial_data():
    """Verify Dashboard renders without error with populated data."""
    dashboard = Dashboard(mode="paper")
    account = AccountManager(is_paper=True, initial_paper_balance=10000.0)
    rm = RiskManager(RiskConfig())

    dashboard.update(
        account_data=account.to_dashboard_dict(),
        position_data={},
        signal_data={
            "15m_bias": "[BTCUSD] BULLISH",
            "5m_bos_choch": "BOS=False CHoCH=False",
            "liquidity_sweep": "False",
            "1m_displacement": "False",
            "retest": "False",
            "vwap": "ABOVE",
            "rsi": "54.2",
            "volume": "1.20x",
            "onnx_confidence": "--",
            "llm_status": "Disabled",
            "signal_score": "SCANNING (No trigger)",
        },
        risk_data=rm.get_risk_status(),
        execution_data={},
    )

    layout = dashboard.render()
    assert layout is not None
