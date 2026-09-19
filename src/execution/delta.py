import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import aiohttp
import websockets
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from src.utils.logger import get_logger, new_correlation_id
from src.risk.risk_manager import round_to_tick, format_price

logger = get_logger(__name__)

class DeltaAPIError(Exception):
    pass

@dataclass
class PaperPosition:
    product_id: int
    symbol: str
    size: float = 0.0
    entry_price: float = 0.0
    bracket_sl: float = 0.0
    bracket_tp: float = 0.0
    side: str = "buy"

@dataclass
class PaperAccount:
    equity: float = 10000.0
    available_balance: float = 10000.0
    positions: Dict[int, PaperPosition] = field(default_factory=dict)


class RateLimiter:
    def __init__(self, capacity: int = 20000, window_sec: int = 300):
        self.capacity = capacity
        self.window_sec = window_sec
        self.budget = capacity
        self.last_reset = time.time()

    async def consume(self, weight: int):
        now = time.time()
        if now - self.last_reset >= self.window_sec:
            self.budget = self.capacity
            self.last_reset = now

        if self.budget < weight or self.budget < 100:
            sleep_time = self.window_sec - (now - self.last_reset)
            if sleep_time > 0:
                logger.warning(f"Rate limit budget low. Sleeping for {sleep_time} seconds.")
                await asyncio.sleep(sleep_time)
            self.budget = self.capacity
            self.last_reset = time.time()

        self.budget -= weight


class DeltaExchangeClient:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "https://api.india.delta.exchange",
        ws_url: str = "wss://socket.india.delta.exchange",
        live_trading: bool = False,
        paper_balance: float = 10000.0,
        **kwargs: Any,
    ):
        cfg = config or {}
        self.api_key = cfg.get("delta_api_key", api_key)
        self.api_secret = cfg.get("delta_api_secret", api_secret)
        self.base_url = cfg.get("delta_rest_url", cfg.get("base_url", base_url))
        self.ws_url = cfg.get("delta_ws_url", cfg.get("ws_url", ws_url))
        self.live_trading = cfg.get("live_trading", live_trading)
        bal = cfg.get("paper_balance", paper_balance)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        
        self.rate_limiter = RateLimiter()
        self.products_cache: Dict[str, Dict[str, Any]] = {}
        self._latest_prices: Dict[str, float] = {}
        self._latest_mark_prices: Dict[str, float] = {}
        
        self.paper_account = PaperAccount(equity=bal, available_balance=bal) if not self.live_trading else None

    async def _init_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        if self.ws:
            await self.ws.close()

    def _generate_signature(self, method: str, timestamp: str, path: str, query_string: str = "", body: str = "") -> str:
        payload = method + timestamp + path + query_string + body
        return hmac.new(
            self.api_secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None, weight: int = 1) -> Any:
        await self.rate_limiter.consume(weight)
        await self._init_session()
        
        timestamp = str(int(time.time()))
        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items())
            
        body_str = json.dumps(body) if body else ""
        
        signature = self._generate_signature(method, timestamp, path, query_string, body_str)
        
        headers = {
            "api-key": self.api_key,
            "signature": signature,
            "timestamp": timestamp,
            "User-Agent": "python-scalping-bot",
            "Content-Type": "application/json"
        }
        
        url = self.base_url + path + query_string
        
        @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), retry=retry_if_exception_type(aiohttp.ClientError))
        async def make_request():
            async with self.session.request(method, url, headers=headers, data=body_str if body else None) as response:
                if response.status == 429 or response.status >= 500:
                    response.raise_for_status()
                data = await response.json()
                if not data.get("success", False):
                    err_code = data.get("error", {}).get("code", "UNKNOWN")
                    raise DeltaAPIError(f"API Error {err_code}")
                return data.get("result", {})
                
        return await make_request()

    async def get_products(self, contract_types: str = "perpetual_futures,futures") -> List[Dict[str, Any]]:
        """Fetch products from Delta Exchange filtered strictly to futures contracts."""
        params = {"contract_types": contract_types, "states": "live"}
        result = await self._request("GET", "/v2/products", params=params, weight=3)
        futures_products = []
        if isinstance(result, list):
            for product in result:
                if product.get("contract_type") in ("perpetual_futures", "futures"):
                    self.products_cache[product["symbol"]] = product
                    futures_products.append(product)
        return futures_products

    async def get_product(self, symbol: str) -> Dict[str, Any]:
        """Fetch and validate a single product, ensuring it is a futures contract."""
        if symbol not in self.products_cache:
            await self.get_products()
            
        if symbol in self.products_cache:
            product = self.products_cache[symbol]
        else:
            product = await self._request("GET", f"/v2/products/{symbol}", weight=3)
            if product.get("contract_type") not in ("perpetual_futures", "futures"):
                logger.error(f"Product {symbol} rejected: type is {product.get('contract_type')} (futures only supported)")
                raise DeltaAPIError(f"Product {symbol} is {product.get('contract_type')}. Only futures products are supported.")
            self.products_cache[symbol] = product
            
        initial_margin = float(product.get("initial_margin", "0.01"))
        def_lev = float(product.get("default_leverage", 0.0))
        prod_max_lev = float(product.get("max_leverage", 0.0))
        if prod_max_lev > 0:
            max_leverage = prod_max_lev
        elif def_lev > 0:
            max_leverage = def_lev
        elif initial_margin > 0:
            max_leverage = (100.0 / initial_margin) if initial_margin > 0.05 else (1.0 / initial_margin)
        else:
            max_leverage = 100.0
        return {
            "id": product["id"],
            "symbol": product.get("symbol", symbol),
            "contract_type": product.get("contract_type", "perpetual_futures"),
            "contract_value": float(product.get("contract_value", 1)),
            "tick_size": float(product.get("tick_size", 0.1)),
            "initial_margin": initial_margin,
            "max_leverage": max_leverage,
            "taker_commission_rate": float(product.get("taker_commission_rate", 0)),
            "maker_commission_rate": float(product.get("maker_commission_rate", 0)),
            "state": product.get("state"),
            "trading_status": product.get("trading_status")
        }

    async def get_wallet_balances(self) -> Dict[str, Any]:
        if not self.live_trading:
            return {
                "equity": self.paper_account.equity,
                "available_balance": self.paper_account.available_balance,
                "available_margin": self.paper_account.available_balance,
                "position_margin": 0.0,
                "used_margin": 0.0,
                "order_margin": 0.0,
                "margin": 0.0,
                "asset_symbol": "USD"
            }
        result = await self._request("GET", "/v2/wallet/balances", weight=3)
        total_equity = 0.0
        total_available = 0.0
        total_pos_margin = 0.0
        total_order_margin = 0.0
        primary_asset = "USD"
        
        if isinstance(result, list) and len(result) > 0:
            for b in result:
                asset = b.get("asset_symbol", "")
                bal = float(b.get("balance", 0.0))
                avail = float(b.get("available_balance", 0.0))
                pos_m = float(b.get("position_margin", 0.0))
                ord_m = float(b.get("order_margin", 0.0))
                eq = float(b.get("equity", bal))
                if asset in ("USD", "USDT") or bal > 0:
                    total_equity += eq
                    total_available += avail
                    total_pos_margin += pos_m
                    total_order_margin += ord_m
                    if bal > 0:
                        primary_asset = asset
                        
            # Fallback if no matching asset had positive balance
            if total_equity == 0.0 and total_available == 0.0 and len(result) > 0:
                first = result[0]
                total_equity = float(first.get("equity", first.get("balance", 0.0)))
                total_available = float(first.get("available_balance", 0.0))
                total_pos_margin = float(first.get("position_margin", 0.0))
                total_order_margin = float(first.get("order_margin", 0.0))
                primary_asset = first.get("asset_symbol", "USD")

        return {
            "equity": total_equity,
            "available_balance": total_available,
            "available_margin": total_available,
            "position_margin": total_pos_margin,
            "used_margin": total_pos_margin,
            "order_margin": total_order_margin,
            "margin": total_pos_margin,
            "asset_symbol": primary_asset
        }

    async def get_positions(self) -> List[Dict[str, Any]]:
        if not self.live_trading:
            return [{"product_id": p.product_id, "size": p.size, "entry_price": p.entry_price} for p in self.paper_account.positions.values()]
        return await self._request("GET", "/v2/positions/margined", params={"contract_types": "perpetual_futures,futures"}, weight=3)

    async def get_position(self, product_id: int) -> List[Dict[str, Any]]:
        if not self.live_trading:
            pos = self.paper_account.positions.get(product_id)
            if pos:
                return [{"product_id": pos.product_id, "size": pos.size, "entry_price": pos.entry_price}]
            return []
        return await self._request("GET", "/v2/positions", params={"product_id": product_id}, weight=3)

    async def set_leverage(self, product_id: int, leverage: int) -> Dict[str, Any]:
        if not self.live_trading:
            return {"success": True, "leverage": leverage}
        return await self._request("POST", f"/v2/products/{product_id}/orders/leverage", body={"leverage": leverage}, weight=5)

    async def place_order(
        self, 
        product_id: int, 
        side: str, 
        size: float, 
        order_type: str, 
        limit_price: Optional[float] = None, 
        bracket_stop_loss_price: Optional[float] = None, 
        bracket_take_profit_price: Optional[float] = None, 
        client_order_id: Optional[str] = None, 
        reduce_only: bool = False,
        tick_size: Optional[float] = None
    ) -> Dict[str, Any]:
        normalized_order_type = "limit_order" if order_type.lower() in ("limit", "limit_order") else "market_order"
        normalized_side = "buy" if side.lower() in ("buy", "long") else "sell"
        
        # Format prices to tick_size if available
        if tick_size and tick_size > 0:
            if limit_price is not None:
                limit_price = round_to_tick(limit_price, tick_size)
            if bracket_stop_loss_price is not None:
                bracket_stop_loss_price = round_to_tick(bracket_stop_loss_price, tick_size)
            if bracket_take_profit_price is not None:
                bracket_take_profit_price = round_to_tick(bracket_take_profit_price, tick_size)

        if not self.live_trading:
            fill_price = limit_price if limit_price else 1000.0
            pos = self.paper_account.positions.get(product_id, PaperPosition(product_id=product_id, symbol=str(product_id)))
            size_change = size if normalized_side == "buy" else -size
            new_size = pos.size + size_change
            if new_size != 0:
                pos.entry_price = fill_price
                pos.bracket_sl = bracket_stop_loss_price or 0.0
                pos.bracket_tp = bracket_take_profit_price or 0.0
                pos.side = normalized_side
            else:
                pos.entry_price = 0.0
                pos.bracket_sl = 0.0
                pos.bracket_tp = 0.0
            pos.size = new_size
            self.paper_account.positions[product_id] = pos
            return {
                "id": "paper_" + str(int(time.time())), 
                "status": "filled", 
                "product_id": product_id, 
                "size": size, 
                "side": normalized_side,
                "limit_price": limit_price,
                "bracket_stop_loss_price": bracket_stop_loss_price,
                "bracket_take_profit_price": bracket_take_profit_price
            }
            
        body: Dict[str, Any] = {
            "product_id": int(product_id),
            "side": normalized_side,
            "size": int(size) if size >= 1 else size,
            "order_type": normalized_order_type,
            "reduce_only": reduce_only
        }
        if limit_price is not None:
            body["limit_price"] = format_price(limit_price, tick_size) if (tick_size and tick_size > 0) else str(limit_price)
        if bracket_stop_loss_price is not None:
            body["bracket_stop_loss_price"] = format_price(bracket_stop_loss_price, tick_size) if (tick_size and tick_size > 0) else str(bracket_stop_loss_price)
        if bracket_take_profit_price is not None:
            body["bracket_take_profit_price"] = format_price(bracket_take_profit_price, tick_size) if (tick_size and tick_size > 0) else str(bracket_take_profit_price)
        if client_order_id is not None:
            body["client_order_id"] = client_order_id
            
        return await self._request("POST", "/v2/orders", body=body, weight=5)

    async def cancel_order(self, order_id: str, product_id: int) -> Dict[str, Any]:
        if not self.live_trading:
            return {"success": True, "id": order_id}
        return await self._request("DELETE", "/v2/orders", body={"id": order_id, "product_id": product_id}, weight=5)

    async def cancel_all_orders(self, product_id: Optional[int] = None) -> Dict[str, Any]:
        if not self.live_trading:
            return {"success": True}
        body: Dict[str, Any] = {"contract_types": "perpetual_futures,futures"}
        if product_id is not None:
            body["product_id"] = product_id
        return await self._request("DELETE", "/v2/orders/all", body=body, weight=5)

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        if not self.live_trading:
            return {"id": order_id, "state": "filled"}
        return await self._request("GET", f"/v2/orders/{order_id}", weight=3)

    async def get_open_orders(self, product_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.live_trading:
            return []
        params: Dict[str, Any] = {"contract_types": "perpetual_futures,futures"}
        if product_id is not None:
            params["product_id"] = product_id
        return await self._request("GET", "/v2/orders", params=params, weight=3)

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        return await self._request("GET", f"/v2/tickers/{symbol}", weight=1)

    async def get_orderbook(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        return await self._request("GET", f"/v2/l2orderbook/{symbol}", params={"depth": depth}, weight=1)

    async def close_position(self, product_id: int, side_to_close: str, size: float) -> Dict[str, Any]:
        side = "sell" if side_to_close.lower() in ("buy", "long") else "buy"
        return await self.place_order(
            product_id=product_id,
            side=side,
            size=size,
            order_type="market",
            reduce_only=True
        )

    def check_paper_sl_tp(self, product_id: int, current_price: float) -> Optional[str]:
        """Check if paper SL or TP has been hit. Returns 'SL_HIT', 'TP_HIT', or None."""
        if self.live_trading or not self.paper_account:
            return None
        pos = self.paper_account.positions.get(product_id)
        if not pos or pos.size == 0:
            return None
        
        is_long = pos.side == "buy"
        
        # Check SL
        if pos.bracket_sl > 0:
            if is_long and current_price <= pos.bracket_sl:
                logger.info(f"Paper SL hit: price={current_price:.2f} <= SL={pos.bracket_sl:.2f}")
                return "SL_HIT"
            if not is_long and current_price >= pos.bracket_sl:
                logger.info(f"Paper SL hit: price={current_price:.2f} >= SL={pos.bracket_sl:.2f}")
                return "SL_HIT"
        
        # Check TP
        if pos.bracket_tp > 0:
            if is_long and current_price >= pos.bracket_tp:
                logger.info(f"Paper TP hit: price={current_price:.2f} >= TP={pos.bracket_tp:.2f}")
                return "TP_HIT"
            if not is_long and current_price <= pos.bracket_tp:
                logger.info(f"Paper TP hit: price={current_price:.2f} <= TP={pos.bracket_tp:.2f}")
                return "TP_HIT"
        
        return None

    def close_paper_position(self, product_id: int, close_price: float) -> float:
        """Close a paper position at given price. Returns realized P&L."""
        if not self.paper_account:
            return 0.0
        pos = self.paper_account.positions.get(product_id)
        if not pos or pos.size == 0:
            return 0.0
        
        is_long = pos.side == "buy"
        if is_long:
            pnl = (close_price - pos.entry_price) * abs(pos.size)
        else:
            pnl = (pos.entry_price - close_price) * abs(pos.size)
        
        # Update paper account
        self.paper_account.equity += pnl
        self.paper_account.available_balance = self.paper_account.equity
        
        # Clear position
        pos.size = 0
        pos.entry_price = 0.0
        pos.bracket_sl = 0.0
        pos.bracket_tp = 0.0
        
        logger.info(f"Paper position closed: pnl={pnl:.2f}, new_equity={self.paper_account.equity:.2f}")
        return pnl

    def update_paper_sl(self, product_id: int, new_sl: float) -> None:
        """Update the stop loss price on an active paper position."""
        if self.paper_account and product_id in self.paper_account.positions:
            self.paper_account.positions[product_id].bracket_sl = new_sl

    async def update_bracket_stop_loss(
        self, product_id: int, new_sl: float, tick_size: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update the bracket stop loss price on Delta Exchange (or paper position)."""
        if tick_size and tick_size > 0:
            new_sl = round_to_tick(new_sl, tick_size)

        if not self.live_trading:
            self.update_paper_sl(product_id, new_sl)
            return {"success": True, "bracket_stop_loss_price": new_sl}

        body: Dict[str, Any] = {
            "product_id": int(product_id),
            "bracket_stop_loss_price": format_price(new_sl, tick_size) if (tick_size and tick_size > 0) else str(new_sl)
        }
        try:
            return await self._request("PUT", "/v2/orders/bracket", body=body, weight=5)
        except Exception as e:
            logger.warning(f"Failed to update Delta bracket stop loss via PUT /v2/orders/bracket: {e}")
            return {"success": False, "error": str(e)}

    async def ws_connect(self):
        if not self.live_trading or not self.api_key or not self.api_secret:
            logger.info("Delta private WS disabled (not in live mode or credentials missing).")
            return

        while True:
            try:
                self.ws = await websockets.connect(self.ws_url)
                timestamp = str(int(time.time()))
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    ('GET' + timestamp + '/live').encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                auth_payload = {
                    "type": "key-auth",
                    "payload": {
                        "api-key": self.api_key,
                        "signature": signature,
                        "timestamp": timestamp
                    }
                }
                await self.ws.send(json.dumps(auth_payload))
                
                sub_payload = {
                    "type": "subscribe",
                    "payload": {
                        "channels": [
                            {"name": "orders", "symbols": ["all"]},
                            {"name": "positions", "symbols": ["all"]},
                            {"name": "user_trades", "symbols": ["all"]}
                        ]
                    }
                }
                await self.ws.send(json.dumps(sub_payload))
                
                async for message in self.ws:
                    data = json.loads(message)
                    logger.debug(f"WS Msg: {data}")
                    
            except Exception as e:
                logger.error(f"WS error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return the freshest LTP for a symbol from Delta Exchange."""
        return self._latest_prices.get(symbol.upper())

    def get_latest_mark_price(self, symbol: str) -> Optional[float]:
        """Return the freshest mark price for a symbol from Delta Exchange."""
        return self._latest_mark_prices.get(symbol.upper())

    async def fetch_ticker_price(self, symbol: str) -> Optional[float]:
        """Fetch live ticker from Delta REST and update cache."""
        try:
            res = await self.get_ticker(symbol)
            if res:
                close = float(res.get("close") or 0.0)
                mark = float(res.get("mark_price") or 0.0)
                if close > 0:
                    self._latest_prices[symbol.upper()] = close
                if mark > 0:
                    self._latest_mark_prices[symbol.upper()] = mark
                return close or mark
        except Exception as e:
            logger.debug(f"Failed to fetch Delta ticker for {symbol}: {e}")
        return None

    async def run_public_ticker(self, symbols: List[str] = ["BTCUSD", "ETHUSD"]):
        """Stream real-time public ticker from Delta Exchange without requiring authentication."""
        sub_payload = {
            "type": "subscribe",
            "payload": {
                "channels": [
                    {"name": "v2/ticker", "symbols": symbols}
                ]
            }
        }
        # Bootstrap once via REST first so prices are instantly available
        for s in symbols:
            try:
                await self.fetch_ticker_price(s)
            except Exception:
                pass

        while True:
            try:
                logger.info(f"Connecting to Delta Public WS ticker: {self.ws_url}")
                async with websockets.connect(self.ws_url, open_timeout=8, ping_interval=20) as ws:
                    await ws.send(json.dumps(sub_payload))
                    logger.info(f"Subscribed to Delta public ticker streams for {symbols}")
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if data.get("type") == "v2/ticker":
                                sym = data.get("symbol")
                                if sym:
                                    cl = float(data.get("close", 0.0))
                                    mk = float(data.get("mark_price", 0.0))
                                    if cl > 0:
                                        self._latest_prices[sym.upper()] = cl
                                    if mk > 0:
                                        self._latest_mark_prices[sym.upper()] = mk
                        except Exception:
                            pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Delta Public WS disconnected: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)
