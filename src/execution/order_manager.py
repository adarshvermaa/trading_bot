import asyncio
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from src.utils.logger import get_logger, new_correlation_id
from src.risk.risk_manager import RiskManager, round_to_tick, format_price
from src.execution.delta import DeltaExchangeClient

logger = get_logger(__name__)

class OrderState(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class ActiveOrder:
    order_id: str
    client_order_id: str
    state: OrderState
    symbol: str
    side: str
    size: float
    entry_price: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    product_id: Optional[int] = None
    last_event: Optional[str] = "CREATED"
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

class OrderManager:
    def __init__(
        self,
        delta_client: DeltaExchangeClient,
        risk_manager: RiskManager,
        config: Any,
        account_manager: Optional[Any] = None,
    ):
        self.delta_client = delta_client
        self.risk_manager = risk_manager
        self.config = config
        self.account_manager = account_manager
        self.active_orders: Dict[str, ActiveOrder] = {}
        self.active_client_ids: set[str] = set()

    @property
    def active_order(self) -> Optional[ActiveOrder]:
        if not self.active_orders:
            return None
        for order in reversed(list(self.active_orders.values())):
            if order.state in (OrderState.PENDING, OrderState.OPEN, OrderState.FILLED):
                return order
        return None

    def update_active_sl(self, new_sl: float) -> bool:
        """Update the stop loss price for the currently active order and portfolio position."""
        ao = self.active_order
        if not ao:
            return False
        ao.sl_price = new_sl
        ao.last_event = "SL_TRAILED"
        if self.account_manager and ao.symbol in self.account_manager.positions:
            self.account_manager.positions[ao.symbol].sl = new_sl
        return True

    async def has_active_position(self, product_id: Optional[int] = None) -> bool:
        try:
            if product_id is not None:
                positions = await self.delta_client.get_position(product_id)
            else:
                positions = await self.delta_client.get_positions()
            for pos in positions:
                if abs(float(pos.get("size", 0))) > 0:
                    return True
        except Exception as e:
            logger.warning(f"Error checking exchange positions: {e}")

        for order in self.active_orders.values():
            if order.state in (OrderState.OPEN, OrderState.FILLED):
                if product_id is None or getattr(order, "product_id", None) == product_id:
                    return True
        return False

    async def validate_and_place_order(
        self,
        symbol: str,
        side: str,
        size: Optional[float] = None,
        price: Optional[float] = None,
        leverage: Optional[int] = None,
        client_order_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        atr: Optional[float] = None,
        spread_bps: float = 0.0,
        equity: Optional[float] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        if not client_order_id:
            client_order_id = new_correlation_id()

        if client_order_id in self.active_client_ids:
            logger.warning(f"Duplicate order protection: {client_order_id} already active.")
            return None

        # Resolve price
        price = price if price is not None else entry_price
        if price is None:
            logger.error(f"Missing price for order {symbol}")
            return None

        # Step 1: Fetch/validate contract
        product = await self.delta_client.get_product(symbol)
        if not product or product.get("state") != "live" or product.get("trading_status") != "operational":
            logger.error(f"Invalid contract state for {symbol}")
            return None

        # Futures-only verification
        if product.get("contract_type") not in ("perpetual_futures", "futures"):
            logger.error(f"Rejecting {symbol}: contract_type '{product.get('contract_type')}' is not a futures contract")
            return None

        product_id = product["id"]

        # One-position rule
        if await self.has_active_position(product_id):
            logger.warning(f"Already have active position for {symbol}. Rejecting.")
            return None

        # Step 2: Validate leverage
        max_leverage = float(product.get("max_leverage", 100.0))
        if leverage is None:
            lev_res = self.risk_manager.validate_leverage(symbol, max_leverage)
            if isinstance(lev_res, tuple):
                lev_ok, leverage, lev_reason = lev_res
            else:
                lev_ok, leverage, lev_reason = bool(lev_res), 10, ""
            if not lev_ok:
                logger.error(f"Leverage validation failed for {symbol}: {lev_reason}")
                return None
        else:
            lev_res = self.risk_manager.validate_leverage(leverage, max_leverage)
            if isinstance(lev_res, tuple):
                lev_ok = lev_res[0]
            else:
                lev_ok = bool(lev_res)
            if not lev_ok:
                logger.error(f"Leverage {leverage} exceeds max {max_leverage}")
                return None

        await self.delta_client.set_leverage(product_id, int(leverage))

        # Step 3 & 4: Notional and Margin
        tick_size = float(product.get("tick_size", 0.1))
        contract_val = float(product.get("contract_value", 1.0))
        price = round_to_tick(price, tick_size)

        if equity is None:
            balances = await self.delta_client.get_wallet_balances()
            equity = float(balances.get("equity", 10000.0 if not self.delta_client.live_trading else 0.0))

        if size is None:
            computed_size, computed_margin, computed_notional = self.risk_manager.calculate_position_size(
                equity=equity,
                price=price,
                leverage=int(leverage),
                contract_value=contract_val,
            )
            size = float(computed_size if computed_size > 0 else 1)
            margin = computed_margin
            notional = computed_notional
        else:
            notional = size * contract_val * price
            margin = notional / leverage

        # Step 5: Liquidation distance
        liquidation_distance = price / leverage

        # Step 6: Calculate SL/TP with tick_size rounding
        calc_sl, calc_tp = self.risk_manager.calculate_sl_tp(
            side=side,
            entry_price=price,
            margin=margin,
            leverage=int(leverage),
            contract_value=contract_val,
            size=int(size) if size >= 1 else 1,
            atr=atr if atr is not None else 100.0,
            tick_size=tick_size,
        )
        final_sl = round_to_tick(sl_price if sl_price is not None else calc_sl, tick_size)
        final_tp = round_to_tick(tp_price if tp_price is not None else calc_tp, tick_size)

        # Step 7: Fees + Slippage
        fees = notional * max(float(product.get("taker_commission_rate", 0.0005)), float(product.get("maker_commission_rate", 0.0002)))
        slippage = notional * 0.001

        # Step 8: Validate Trade via RiskManager
        val_res = self.risk_manager.validate_trade(
            equity=equity,
            margin=margin,
            notional=notional,
            leverage=int(leverage),
            sl_price=final_sl,
            tp_price=final_tp,
            entry_price=price,
            side=side,
            atr=atr if atr is not None else 100.0,
            spread_bps=spread_bps,
            has_position=False,
            fees=fees,
            slippage=slippage,
            liquidation_distance=liquidation_distance,
        )
        is_valid = val_res[0] if isinstance(val_res, tuple) else bool(val_res)
        if not is_valid:
            reason = val_res[1] if isinstance(val_res, tuple) else "Risk check failed"
            logger.error(f"Trade rejected by RiskManager: {reason}")
            return None

        # Log details
        account_pnl = 0.0
        leveraged_pnl = 0.0
        logger.info(f"account_pnl: {account_pnl}, position_notional: {notional}, position_margin: {margin}, leveraged_pnl: {leveraged_pnl}")

        # Step 9: Place Order with rounded prices
        self.active_client_ids.add(client_order_id)

        # Normalize side to Delta API format ("buy" / "sell")
        delta_side = "buy" if side.lower() in ("buy", "long") else "sell"

        try:
            order_res = await self.delta_client.place_order(
                product_id=product_id,
                side=delta_side,
                size=size,
                order_type="limit",
                limit_price=price,
                bracket_stop_loss_price=final_sl,
                bracket_take_profit_price=final_tp,
                client_order_id=client_order_id,
                tick_size=tick_size,
            )

            order_id = str(order_res.get("id")) if order_res else None
            if order_id:
                active_order = ActiveOrder(
                    order_id=order_id,
                    client_order_id=client_order_id,
                    state=OrderState.OPEN if self.delta_client.live_trading else OrderState.FILLED,
                    symbol=symbol,
                    side=side.upper(),
                    size=size,
                    entry_price=price,
                    sl_price=final_sl,
                    tp_price=final_tp,
                    product_id=product_id,
                    last_event="ORDER_PLACED",
                )
                self.active_orders[order_id] = active_order

                if not self.delta_client.live_trading and self.account_manager:
                    self.account_manager.update_paper_position(
                        symbol=symbol,
                        side=side.upper(),
                        size=int(size) if size >= 1 else 1,
                        price=price,
                        leverage=int(leverage),
                        margin=margin,
                        notional=notional,
                        sl=final_sl,
                        tp=final_tp,
                    )
                return order_res
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            self.active_client_ids.discard(client_order_id)
            return None

        return None

    async def on_structure_invalidated(
        self,
        is_filled: bool = False,
        order_id: Optional[str] = None,
        product_id: Optional[int] = None,
        side: Optional[str] = None,
        size: Optional[float] = None,
    ):
        target_order = None
        if order_id and order_id in self.active_orders:
            target_order = self.active_orders[order_id]
        else:
            target_order = self.active_order

        if target_order:
            order_id = order_id or target_order.order_id
            product_id = product_id or getattr(target_order, "product_id", None)
            size = size or target_order.size

        if not is_filled:
            if order_id:
                try:
                    await self.delta_client.cancel_order(order_id, product_id or 0)
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order_id}: {e}")
                if order_id in self.active_orders:
                    self.active_orders[order_id].state = OrderState.CANCELLED
                    self.active_orders[order_id].last_event = "STRUCTURE_INVALIDATED"
            logger.info("Signal marked invalid, cancelled pending order")
        else:
            if side is not None:
                current_side = side
            elif target_order:
                current_side = target_order.side
            else:
                current_side = "buy"

            if product_id:
                try:
                    await self.delta_client.close_position(product_id, current_side, size or 1.0)
                except Exception as e:
                    logger.warning(f"Failed to close position: {e}")
            if target_order:
                target_order.state = OrderState.CANCELLED
                target_order.last_event = "STRUCTURE_INVALIDATED"
                if not self.delta_client.live_trading and self.account_manager:
                    self.account_manager.close_paper_position(
                        symbol=target_order.symbol,
                        close_price=target_order.entry_price,
                    )
            logger.info("Closed position due to structure invalidation, reason=STRUCTURE_INVALIDATION")

    async def close_and_record(
        self,
        reason: str,
        close_price: float,
        contract_value: float = 1.0,
    ) -> float:
        """Close current position, record P&L, clean up state. Returns realized P&L."""
        target = self.active_order
        if not target:
            logger.warning("close_and_record called with no active order")
            return 0.0

        product_id = target.product_id
        symbol = target.symbol
        pnl = 0.0

        if not self.delta_client.live_trading:
            # Paper mode close
            if product_id:
                pnl = self.delta_client.close_paper_position(product_id, close_price)
            if self.account_manager:
                self.account_manager.close_paper_position(symbol, close_price, contract_value)
        else:
            # Live mode close
            if product_id and target.size > 0:
                try:
                    await self.delta_client.close_position(product_id, target.side, target.size)
                except Exception as e:
                    logger.error(f"Failed to close live position: {e}")

        # Record in risk manager
        self.risk_manager.record_trade_result(pnl)

        # Update order state
        target.state = OrderState.CANCELLED
        target.last_event = reason

        logger.info(
            f"Position closed: symbol={symbol}, reason={reason}, pnl={pnl:.2f}, "
            f"close_price={close_price:.2f}"
        )
        return pnl

    def clear_closed_orders(self) -> None:
        """Remove all cancelled/rejected orders from active tracking."""
        to_remove = [
            oid for oid, order in self.active_orders.items()
            if order.state in (OrderState.CANCELLED, OrderState.REJECTED)
        ]
        for oid in to_remove:
            order = self.active_orders.pop(oid)
            self.active_client_ids.discard(order.client_order_id)
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} closed orders from active tracking")
