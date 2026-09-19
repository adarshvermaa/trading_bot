import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    is_closed: bool

class CandleStore:
    def __init__(self, max_len: int = 200):
        self.max_len = max_len
        # (symbol, timeframe) -> deque of Candles
        self._store: Dict[Tuple[str, str], deque[Candle]] = {}

    def add_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        key = (symbol, timeframe)
        if key not in self._store:
            self._store[key] = deque(maxlen=self.max_len)
        self._store[key].append(candle)

    def get_candles(self, symbol: str, timeframe: str) -> List[Candle]:
        key = (symbol, timeframe)
        if key not in self._store:
            return []
        return list(self._store[key])


class BinanceWSClient:
    def __init__(
        self,
        symbols: List[str],
        binance_to_internal_symbol_map: Dict[str, str],
        stale_data_seconds: float = 30.0,
        ws_url: str = "wss://fstream.binance.com/stream",
        max_retries: int = 10,
        backoff_base: int = 2,
    ):
        """
        symbols: list of binance lowercase symbols like 'btcusdt'
        binance_to_internal_symbol_map: map from 'BTCUSDT' -> 'BTC-USD' etc.
        """
        self.symbols = [s.lower() for s in symbols]
        self.symbol_map = binance_to_internal_symbol_map
        self.timeframes = ["1m", "5m", "15m"]
        self.ws_url = ws_url
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        
        self.store = CandleStore(max_len=200)
        self.new_candle_event = asyncio.Event()
        
        self.stale_data_seconds = stale_data_seconds
        self._last_message_time = 0.0
        self._running = False
        self._ws = None
        
        # Real-time tick prices dictionary updated on every WS message
        self._latest_prices: Dict[str, float] = {}

        # To avoid duplicate closed candles: (symbol, interval, open_time)
        self._seen_candles: Set[Tuple[str, str, int]] = set()

    @property
    def candle_store(self) -> CandleStore:
        """Alias for self.store."""
        return self.store

    @property
    def is_connected(self) -> bool:
        if self._ws is None:
            return False
        state = getattr(self._ws, "state", None)
        if state is not None:
            return getattr(state, "name", "") == "OPEN" or state == 1
        return not getattr(self._ws, "closed", True)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the freshest real-time price for a symbol (internal, Binance, or fallback to last 1m candle)."""
        if symbol in self._latest_prices:
            return self._latest_prices[symbol]
        sym_lower = symbol.lower()
        if sym_lower in self._latest_prices:
            return self._latest_prices[sym_lower]
        sym_upper = symbol.upper()
        if sym_upper in self._latest_prices:
            return self._latest_prices[sym_upper]
        # Fallback to last closed 1m candle
        c_list = self.store.get_candles(symbol, "1m")
        if not c_list:
            c_list = self.store.get_candles(sym_upper, "1m")
        if c_list:
            return c_list[-1].close
        return None

    @property
    def is_stale(self) -> bool:
        if self._last_message_time == 0.0:
            return True
        return (time.time() - self._last_message_time) > self.stale_data_seconds

    def _get_stream_url(self) -> str:
        streams = []
        for sym in self.symbols:
            for tf in self.timeframes:
                streams.append(f"{sym.lower()}@kline_{tf}")
        streams_str = "/".join(streams)
        base = self.ws_url.rstrip("/")
        if "/stream" in base:
            return f"{base}?streams={streams_str}"
        return f"{base}/stream?streams={streams_str}"

    def _get_candidate_ws_endpoints(self) -> List[str]:
        streams = []
        for sym in self.symbols:
            for tf in self.timeframes:
                streams.append(f"{sym.lower()}@kline_{tf}")
        streams_str = "/".join(streams)

        # Primary verified endpoints: combined stream with zero subscription handshake latency
        endpoints: List[str] = [
            f"wss://stream.binance.com:9443/stream?streams={streams_str}",
            f"wss://stream.binance.com/stream?streams={streams_str}",
            "wss://stream.binance.com:9443/ws",
            "wss://stream.binance.com/ws",
        ]
        if self.ws_url:
            custom = self.ws_url.strip()
            if "/stream" in custom and "streams=" not in custom:
                custom = f"{custom.rstrip('/')}?streams={streams_str}"
            if custom not in endpoints and "fstream" not in custom:
                endpoints.insert(0, custom)
        return endpoints

    async def _handle_message(self, message: str) -> None:
        self._last_message_time = time.time()
        try:
            payload = json.loads(message)
            if not isinstance(payload, dict):
                return

            # Skip subscription acknowledgements: {"result": null, "id": 1}
            if "result" in payload and "id" in payload:
                return

            # Support both wrapped {"stream": "...", "data": {"e": "kline", ...}} and direct {"e": "kline", ...}
            data = payload.get("data", payload)
            if data.get("e") != "kline":
                return

            k = data.get("k", {})
            binance_symbol = k.get("s") or data.get("s", "")
            internal_symbol = self.symbol_map.get(
                binance_symbol, 
                self.symbol_map.get(binance_symbol.lower(), binance_symbol)
            )

            # Record real-time tick price on every message (sub-second live updates)
            cur_price = float(k.get("c", 0.0))
            if cur_price > 0:
                self._latest_prices[internal_symbol] = cur_price
                if binance_symbol:
                    self._latest_prices[binance_symbol] = cur_price
                    self._latest_prices[binance_symbol.lower()] = cur_price

            is_closed = k.get("x", False)
            if not is_closed:
                return

            interval = k.get("i", "")
            open_time = k.get("t", 0)

            seen_key = (internal_symbol, interval, open_time)
            if seen_key in self._seen_candles:
                return
            self._seen_candles.add(seen_key)

            if len(self._seen_candles) > 10000:
                self._seen_candles.clear()

            candle = Candle(
                open_time=open_time,
                open=float(k.get("o", 0.0)),
                high=float(k.get("h", 0.0)),
                low=float(k.get("l", 0.0)),
                close=float(k.get("c", 0.0)),
                volume=float(k.get("v", 0.0)),
                close_time=k.get("T", 0),
                is_closed=is_closed
            )

            self.store.add_candle(internal_symbol, interval, candle)
            if binance_symbol and binance_symbol != internal_symbol:
                self.store.add_candle(binance_symbol, interval, candle)
                self.store.add_candle(binance_symbol.lower(), interval, candle)
            self.new_candle_event.set()

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _monitor_stale(self) -> None:
        start_time = time.time()
        while self._running:
            await asyncio.sleep(5)
            is_initial_stall = (self._last_message_time == 0.0 and (time.time() - start_time) > 15.0)
            is_runtime_stall = (self._last_message_time > 0 and (time.time() - self._last_message_time) > self.stale_data_seconds)
            if is_initial_stall or is_runtime_stall:
                logger.warning(
                    f"WebSocket data is stale! No messages received in last {self.stale_data_seconds}s. Forcing reconnect..."
                )
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                try:
                    await self.bootstrap_historical_candles(limit=2)
                except Exception as e:
                    logger.debug(f"REST candle fallback failed: {e}")

    async def run(self) -> None:
        self._running = True
        stale_monitor_task = asyncio.create_task(self._monitor_stale())

        endpoints = self._get_candidate_ws_endpoints()
        streams: List[str] = []
        for sym in self.symbols:
            for tf in self.timeframes:
                streams.append(f"{sym.lower()}@kline_{tf}")

        retry_count = 0
        max_retries = 10
        base_delay = 2.0
        endpoint_idx = 0

        while self._running:
            target_url = endpoints[endpoint_idx % len(endpoints)]
            try:
                logger.info(f"Connecting to Binance WS: {target_url}")
                async with websockets.connect(
                    target_url,
                    ping_interval=15,
                    ping_timeout=10,
                    open_timeout=8,
                    close_timeout=3,
                ) as ws:
                    self._ws = ws
                    logger.info(f"Connected to Binance WS via {target_url}")
                    retry_count = 0  # Reset on successful connection
                    self._last_message_time = time.time()

                    # Subscribe only if endpoint is a bare /ws endpoint (not combined stream?streams=)
                    if "/ws" in target_url and "/stream" not in target_url:
                        for i in range(0, len(streams), 10):
                            batch = streams[i:i+10]
                            await ws.send(json.dumps({
                                "method": "SUBSCRIBE",
                                "params": batch,
                                "id": i + 1
                            }))
                        logger.info(f"Subscribed to all {len(streams)} market streams")

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except (ConnectionClosed, WebSocketException, Exception) as e:
                logger.warning(f"WebSocket disconnected from {target_url}: {e}")
                endpoint_idx += 1  # Cycle to next candidate endpoint

            if not self._running:
                break

            retry_count += 1
            if retry_count > max_retries:
                logger.error("Max retries reached. Stopping WS client.")
                break

            delay = min(10.0, base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1))
            logger.info(f"Reconnecting in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

        self._running = False
        stale_monitor_task.cancel()

    async def bootstrap_historical_candles(self, limit: int = 100) -> int:
        """
        Pre-seed CandleStore with historical closed candles from Binance REST API.
        Fetches limit (default 100) candles for all symbols and timeframes (1m, 5m, 15m).
        Ensures market structure and indicator requirements (>= 50 candles) are immediately met.
        """
        import aiohttp
        total_bootstrapped = 0
        logger.info(f"Bootstrapping {limit} historical candles for {len(self.symbols)} assets across {self.timeframes}...")
        
        async with aiohttp.ClientSession(headers={"User-Agent": "python-scalping-bot"}) as session:
            tasks = []
            for sym in self.symbols:
                for tf in self.timeframes:
                    tasks.append(self._fetch_symbol_klines(session, sym, tf, limit))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, int):
                    total_bootstrapped += res
                    
        if total_bootstrapped > 0:
            self._last_message_time = time.time()
            self.new_candle_event.set()
            logger.info(f"Successfully bootstrapped {total_bootstrapped} historical candles.")
        else:
            logger.warning("No historical candles were bootstrapped. Strategy will wait for live WebSocket candles.")
            
        return total_bootstrapped

    async def _fetch_symbol_klines(self, session: Any, binance_symbol: str, interval: str, limit: int) -> int:
        sym_upper = binance_symbol.upper()
        # 1. Try Binance Futures REST API
        fut_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym_upper}&interval={interval}&limit={limit + 1}"
        klines = None
        try:
            import aiohttp
            async with session.get(fut_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    klines = await resp.json()
        except Exception as e:
            logger.debug(f"Futures klines fetch failed for {sym_upper} {interval}: {e}")

        # 2. Fallback to Binance Spot REST API if needed
        if not klines or not isinstance(klines, list):
            spot_url = f"https://api.binance.com/api/v3/klines?symbol={sym_upper}&interval={interval}&limit={limit + 1}"
            try:
                import aiohttp
                async with session.get(spot_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        klines = await resp.json()
            except Exception as e:
                logger.debug(f"Spot klines fallback failed for {sym_upper} {interval}: {e}")

        if not klines or not isinstance(klines, list) or len(klines) == 0:
            return 0

        # The last candle is typically the currently open/incomplete candle. Exclude it so store has only closed candles.
        closed_klines = klines[:-1] if len(klines) > limit else klines

        internal_symbol = self.symbol_map.get(
            sym_upper,
            self.symbol_map.get(binance_symbol.lower(), sym_upper)
        )

        count = 0
        for k in closed_klines:
            try:
                open_time = int(k[0])
                candle = Candle(
                    open_time=open_time,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    close_time=int(k[6]),
                    is_closed=True
                )
                self.store.add_candle(internal_symbol, interval, candle)
                if sym_upper != internal_symbol:
                    self.store.add_candle(sym_upper, interval, candle)
                    self.store.add_candle(binance_symbol.lower(), interval, candle)
                seen_key = (internal_symbol, interval, open_time)
                self._seen_candles.add(seen_key)
                count += 1
            except Exception:
                continue

        if closed_klines:
            try:
                last_close = float(closed_klines[-1][4])
                if last_close > 0:
                    self._latest_prices[internal_symbol] = last_close
                    self._latest_prices[sym_upper] = last_close
                    self._latest_prices[binance_symbol.lower()] = last_close
            except Exception:
                pass

        return count

    async def stop(self) -> None:
        logger.info("Stopping Binance WS Client...")
        self._running = False
        if self._ws:
            await self._ws.close()
