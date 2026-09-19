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
    contract_value: float = 1.0

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


DELTA_DEFAULT_PRODUCTS: Dict[str, Dict[str, Any]] = {
    "BTCUSD": {
        "id": 27,
        "symbol": "BTCUSD",
        "contract_type": "perpetual_futures",
        "contract_value": 0.001,
        "tick_size": 0.5,
        "initial_margin": 0.5,
        "default_leverage": 200.0,
        "max_leverage": 200.0,
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
        "state": "live",
        "trading_status": "operational",
    },
    "ETHUSD": {
        "id": 3136,
        "symbol": "ETHUSD",
        "contract_type": "perpetual_futures",
        "contract_value": 0.01,
        "tick_size": 0.05,
        "initial_margin": 0.5,
        "default_leverage": 200.0,
        "max_leverage": 200.0,
        "taker_commission_rate": 0.0005,
        "maker_commission_rate": 0.0002,
        "state": "live",
        "trading_status": "operational",
    },
}

def normalize_delta_symbol(symbol: str) -> str:
    """Normalize asset or symbol string into Delta's official perpetual symbol (BTCUSD or ETHUSD)."""
    s = (symbol or "").upper().strip()
    if s in ("BTC", "BTCUSD", "BTCUSDT"):
        return "BTCUSD"
    if s in ("ETH", "ETHUSD", "ETHUSDT"):
        return "ETHUSD"
    return s

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
        self.products_by_id_cache: Dict[int, Dict[str, Any]] = {}
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
        
        headers = {
            "User-Agent": "rest-client",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Only attach signing headers if API credentials are present
        if self.api_key and self.api_secret:
            signature = self._generate_signature(method, timestamp, path, query_string, body_str)
            headers["api-key"] = self.api_key
            headers["signature"] = signature
            headers["timestamp"] = timestamp
        
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

    async def fetch_target_products(self, symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Fetch and cache official Delta product specifications for BTC and ETH only."""
        target_symbols = symbols if symbols is not None else ["BTCUSD", "ETHUSD"]
        fetched = {}
        for sym in target_symbols:
            norm_sym = normalize_delta_symbol(sym)
            try:
                prod = await self.get_product(norm_sym)
                fetched[norm_sym] = prod
            except Exception as e:
                logger.debug(f"Could not fetch product {norm_sym} from Delta REST: {e}")
                if norm_sym in DELTA_DEFAULT_PRODUCTS:
                    fetched[norm_sym] = DELTA_DEFAULT_PRODUCTS[norm_sym]
                    self.products_cache[norm_sym] = DELTA_DEFAULT_PRODUCTS[norm_sym]
        return fetched

    async def get_products(self, contract_types: str = "perpetual_futures,futures") -> List[Dict[str, Any]]:
        """Fetch products from Delta Exchange filtered strictly to futures contracts."""
        params = {"contract_types": contract_types, "states": "live"}
        result = await self._request("GET", "/v2/products", params=params, weight=3)
        futures_products = []
        if isinstance(result, list):
            for product in result:
                if product.get("contract_type") in ("perpetual_futures", "futures"):
                    self.products_cache[product["symbol"]] = product
                    if "id" in product:
                        self.products_by_id_cache[int(product["id"])] = product
                    futures_products.append(product)
        return futures_products

    async def get_product(self, symbol: str) -> Dict[str, Any]:
        """Fetch and validate a single product, ensuring it is a futures contract."""
        norm_sym = normalize_delta_symbol(symbol)
        if norm_sym not in self.products_cache:
            try:
                await self.get_products()
            except Exception:
                pass
            
        if norm_sym in self.products_cache:
            product = self.products_cache[norm_sym]
        else:
            try:
                product = await self._request("GET", f"/v2/products/{norm_sym}", weight=3)
                self.products_cache[norm_sym] = product
            except Exception as e:
                if norm_sym in DELTA_DEFAULT_PRODUCTS:
                    product = DELTA_DEFAULT_PRODUCTS[norm_sym]
                    self.products_cache[norm_sym] = product
                else:
                    raise DeltaAPIError(f"Product {symbol} fetch failed: {e}")

        if product.get("contract_type") not in ("perpetual_futures", "futures"):
            logger.error(f"Product {symbol} rejected: type is {product.get('contract_type')} (futures only supported)")
            raise DeltaAPIError(f"Product {symbol} is {product.get('contract_type')}. Only futures products are supported.")
            
        if "id" in product:
            self.products_by_id_cache[int(product["id"])] = product
            
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
            max_leverage = 200.0

        contract_val = float(product.get("contract_value", 0.001 if "BTC" in norm_sym else 0.01))
        tick_sz = float(product.get("tick_size", 0.5 if "BTC" in norm_sym else 0.05))

        return {
            "id": int(product["id"]),
            "symbol": product.get("symbol", norm_sym),
            "contract_type": product.get("contract_type", "perpetual_futures"),
            "contract_value": contract_val,
            "tick_size": tick_sz,
            "initial_margin": initial_margin,
            "max_leverage": max_leverage,
            "taker_commission_rate": float(product.get("taker_commission_rate", 0.0005)),
            "maker_commission_rate": float(product.get("maker_commission_rate", 0.0002)),
            "state": product.get("state", "live"),
            "trading_status": product.get("trading_status", "operational")
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
        tick_size: Optional[float] = None,
        symbol: Optional[str] = None,
        market_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        # Force market order only on Delta Exchange - limit orders are removed
        normalized_order_type = "market_order"
        normalized_side = "buy" if side.lower() in ("buy", "long") else "sell"
        passed_price = market_price if (market_price is not None and market_price > 0) else limit_price
        limit_price = None  # Market orders strictly do not use limit price

        # Format prices to tick_size if available
        if tick_size and tick_size > 0:
            if bracket_stop_loss_price is not None:
                bracket_stop_loss_price = round_to_tick(bracket_stop_loss_price, tick_size)
            if bracket_take_profit_price is not None:
                bracket_take_profit_price = round_to_tick(bracket_take_profit_price, tick_size)

        # Resolve symbol if not provided
        resolved_symbol = symbol
        if not resolved_symbol:
            prod_info = self.products_by_id_cache.get(int(product_id))
            if prod_info:
                resolved_symbol = prod_info.get("symbol")
            else:
                for s, p in self.products_cache.items():
                    if p.get("id") == product_id or str(p.get("id")) == str(product_id):
                        resolved_symbol = s
                        break

        # Resolve contract value (0.001 for BTCUSD, 0.01 for ETHUSD)
        cv = 1.0
        if resolved_symbol:
            if "BTC" in resolved_symbol.upper():
                cv = 0.001
            elif "ETH" in resolved_symbol.upper():
                cv = 0.01
            elif resolved_symbol in self.products_cache:
                cv = float(self.products_cache[resolved_symbol].get("contract_value", 1.0))
        elif int(product_id) in self.products_by_id_cache:
            cv = float(self.products_by_id_cache[int(product_id)].get("contract_value", 1.0))

        if not self.live_trading:
            fill_price: Optional[float] = None

            # 1) Try Delta L2 Orderbook for true market execution (Best Ask for Buy, Best Bid for Sell)
            if resolved_symbol:
                try:
                    best_bid, best_ask = await self.get_best_bid_ask(resolved_symbol)
                    if normalized_side == "buy" and best_ask and best_ask > 0:
                        fill_price = best_ask
                    elif normalized_side == "sell" and best_bid and best_bid > 0:
                        fill_price = best_bid
                except Exception as e:
                    logger.debug(f"Could not fetch orderbook for {resolved_symbol}: {e}")

            # 2) If orderbook gave no price, try Delta's latest price cache
            if fill_price is None and resolved_symbol:
                fill_price = self.get_latest_price(resolved_symbol) or self.get_latest_mark_price(resolved_symbol)

            # 3) If not in cache, fetch ticker from Delta REST
            if fill_price is None and resolved_symbol:
                try:
                    fill_price = await self.fetch_ticker_price(resolved_symbol)
                except Exception as e:
                    logger.debug(f"Could not fetch ticker for {resolved_symbol}: {e}")

            # 4) If still None, use caller's live market_price or passed price
            if fill_price is None and passed_price is not None and passed_price > 0:
                fill_price = passed_price

            # 5) If still None, check numeric string in _latest_prices
            if fill_price is None and str(product_id) in self._latest_prices:
                fill_price = self._latest_prices[str(product_id)]

            if fill_price is None or fill_price <= 0:
                logger.error(f"Cannot execute market order: No live market price available for {resolved_symbol or product_id}")
                raise DeltaAPIError(f"No market price available from Delta Exchange for {resolved_symbol or product_id}")

            if tick_size and tick_size > 0:
                fill_price = round_to_tick(fill_price, tick_size)

            pos = self.paper_account.positions.get(
                product_id, 
                PaperPosition(
                    product_id=product_id, 
                    symbol=resolved_symbol or str(product_id),
                    contract_value=cv,
                )
            )
            pos.symbol = resolved_symbol or pos.symbol
            pos.contract_value = cv
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
                "order_type": "market_order",
                "limit_price": None,
                "fill_price": fill_price,
                "average_fill_price": fill_price,
                "bracket_stop_loss_price": bracket_stop_loss_price,
                "bracket_take_profit_price": bracket_take_profit_price
            }
            
        prod_sym = resolved_symbol
        if not prod_sym:
            prod_sym = "BTCUSD" if int(product_id) == 27 else "ETHUSD"

        body: Dict[str, Any] = {
            "product_id": int(product_id),
            "product_symbol": prod_sym,
            "side": normalized_side,
            "size": max(1, int(round(size))),
            "order_type": "market_order",
            "time_in_force": "gtc",
            "reduce_only": reduce_only
        }
        if bracket_stop_loss_price is not None:
            body["bracket_stop_loss_price"] = format_price(bracket_stop_loss_price, tick_size) if (tick_size and tick_size > 0) else str(bracket_stop_loss_price)
            body["stop_trigger_method"] = "last_traded_price"
            body["bracket_stop_trigger_method"] = "last_traded_price"
        if bracket_take_profit_price is not None:
            body["bracket_take_profit_price"] = format_price(bracket_take_profit_price, tick_size) if (tick_size and tick_size > 0) else str(bracket_take_profit_price)
            body["stop_trigger_method"] = "last_traded_price"
            body["bracket_stop_trigger_method"] = "last_traded_price"
        if client_order_id is not None:
            body["client_order_id"] = client_order_id
            
        return await self._request("POST", "/v2/orders", body=body, weight=5)

    async def get_best_bid_ask(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """Return (best_bid, best_ask) from Delta's live L2 orderbook."""
        try:
            ob = await self.get_orderbook(symbol, depth=5)
            if not ob:
                return None, None
            bids = ob.get("buy", [])
            asks = ob.get("sell", [])
            best_bid = float(bids[0]["price"]) if bids and "price" in bids[0] else None
            best_ask = float(asks[0]["price"]) if asks and "price" in asks[0] else None
            return best_bid, best_ask
        except Exception as e:
            logger.debug(f"Failed to fetch best bid/ask for {symbol}: {e}")
            return None, None


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
        
        cv = getattr(pos, "contract_value", 1.0)
        if cv == 1.0:
            if "BTC" in pos.symbol.upper():
                cv = 0.001
            elif "ETH" in pos.symbol.upper():
                cv = 0.01

        is_long = pos.side == "buy"
        if is_long:
            pnl = (close_price - pos.entry_price) * abs(pos.size) * cv
        else:
            pnl = (pos.entry_price - close_price) * abs(pos.size) * cv
        
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
        self, product_id: int, new_sl: float, tick_size: Optional[float] = None, order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update the bracket stop loss price on Delta Exchange (or paper position)."""
        if tick_size and tick_size > 0:
            new_sl = round_to_tick(new_sl, tick_size)

        if not self.live_trading:
            self.update_paper_sl(product_id, new_sl)
            return {"success": True, "bracket_stop_loss_price": new_sl}

        prod_symbol = "BTCUSD" if int(product_id) == 27 else "ETHUSD"
        if int(product_id) in self.products_by_id_cache:
            prod_symbol = self.products_by_id_cache[int(product_id)].get("symbol", prod_symbol)

        body: Dict[str, Any] = {
            "product_id": int(product_id),
            "product_symbol": prod_symbol,
            "bracket_stop_loss_price": format_price(new_sl, tick_size) if (tick_size and tick_size > 0) else str(new_sl),
            "bracket_stop_trigger_method": "last_traded_price",
        }
        if order_id:
            try:
                body["id"] = int(order_id)
            except ValueError:
                pass

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
        if not symbol:
            return None
        if symbol.isdigit() and int(symbol) in self.products_by_id_cache:
            real_sym = self.products_by_id_cache[int(symbol)].get("symbol", "")
            if real_sym and real_sym.upper() in self._latest_prices:
                return self._latest_prices[real_sym.upper()]
        return self._latest_prices.get(symbol.upper())

    def get_latest_mark_price(self, symbol: str) -> Optional[float]:
        """Return the freshest mark price for a symbol from Delta Exchange."""
        if not symbol:
            return None
        if symbol.isdigit() and int(symbol) in self.products_by_id_cache:
            real_sym = self.products_by_id_cache[int(symbol)].get("symbol", "")
            if real_sym and real_sym.upper() in self._latest_mark_prices:
                return self._latest_mark_prices[real_sym.upper()]
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
