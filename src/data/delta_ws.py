import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import aiohttp
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


class DeltaWSClient:
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        ws_url: str = "wss://socket.india.delta.exchange",
        rest_url: str = "https://api.india.delta.exchange",
        stale_data_seconds: float = 30.0,
        max_retries: int = 10,
        backoff_base: int = 2,
    ):
        """
        DeltaWSClient handles real-time market data directly from Delta Exchange:
        - Subscribes to candlestick_1m, candlestick_5m, candlestick_15m for target assets (BTCUSD & ETHUSD).
        - Subscribes to v2/ticker for sub-second live prices and quote updates.
        - Provides CandleStore and historical candle bootstrapping via Delta REST API.
        """
        raw_symbols = symbols if symbols is not None else ["BTCUSD", "ETHUSD"]
        self.symbols: List[str] = [s.upper() for s in raw_symbols]
        self.timeframes: List[str] = ["1m", "5m", "15m"]
        self.ws_url = ws_url.strip()
        self.rest_url = rest_url.rstrip("/")
        self.stale_data_seconds = stale_data_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        self.store = CandleStore(max_len=200)
        self.new_candle_event = asyncio.Event()

        self._last_message_time = 0.0
        self._running = False
        self._ws = None

        # Real-time tick prices dictionary updated on every WS message
        self._latest_prices: Dict[str, float] = {}
        self._latest_quotes: Dict[str, Dict[str, Any]] = {}

        # Tracking forming candle per (symbol, timeframe) to detect closure
        self._forming_candles: Dict[Tuple[str, str], Candle] = {}

        # Set of seen closed candles: (symbol, timeframe, open_time)
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

    @property
    def is_stale(self) -> bool:
        if self._last_message_time == 0.0:
            return True
        return (time.time() - self._last_message_time) > self.stale_data_seconds

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the freshest real-time price for a symbol or fallback to the last 1m candle."""
        sym_upper = symbol.upper()
        if sym_upper in self._latest_prices:
            return self._latest_prices[sym_upper]
        if symbol in self._latest_prices:
            return self._latest_prices[symbol]
        sym_lower = symbol.lower()
        if sym_lower in self._latest_prices:
            return self._latest_prices[sym_lower]

        # Fallback to last closed 1m candle
        c_list = self.store.get_candles(sym_upper, "1m")
        if not c_list and sym_upper != symbol:
            c_list = self.store.get_candles(symbol, "1m")
        if c_list:
            return c_list[-1].close
        return None

    def get_latest_quotes(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get best bid / best ask quotes from v2/ticker."""
        return self._latest_quotes.get(symbol.upper(), self._latest_quotes.get(symbol))

    @staticmethod
    def _resolution_seconds(resolution: str) -> int:
        if resolution == "1m":
            return 60
        elif resolution == "5m":
            return 300
        elif resolution == "15m":
            return 900
        elif resolution == "30m":
            return 1800
        elif resolution == "1h":
            return 3600
        return 60

    @staticmethod
    def _normalize_time_to_ms(t_val: Any) -> int:
        """Convert timestamps (microseconds, milliseconds, or seconds) into milliseconds."""
        try:
            t = int(t_val)
            if t > 10_000_000_000_000:  # Microseconds
                return t // 1_000
            elif t > 10_000_000_000:    # Milliseconds
                return t
            elif t > 0:                 # Seconds
                return t * 1_000
        except (ValueError, TypeError):
            pass
        return int(time.time() * 1000)

    def _build_subscription_payload(self) -> Dict[str, Any]:
        channels = [
            {"name": "candlestick_1m", "symbols": self.symbols},
            {"name": "candlestick_5m", "symbols": self.symbols},
            {"name": "candlestick_15m", "symbols": self.symbols},
            {"name": "v2/ticker", "symbols": self.symbols},
        ]
        return {
            "type": "subscribe",
            "payload": {
                "channels": channels
            }
        }

    async def _handle_message(self, message: str) -> None:
        self._last_message_time = time.time()
        try:
            payload = json.loads(message)
            if not isinstance(payload, dict):
                return

            msg_type = payload.get("type", "")

            # Heartbeat pong or subscription acknowledgement
            if msg_type in ("pong", "subscriptions"):
                return

            # 1. Process v2/ticker messages for high-frequency pricing
            if msg_type == "v2/ticker":
                symbol = payload.get("symbol", "").upper()
                if not symbol:
                    return

                try:
                    close_price = float(payload.get("close", 0.0))
                    if close_price > 0:
                        self._latest_prices[symbol] = close_price
                except (ValueError, TypeError):
                    pass

                quotes = payload.get("quotes")
                if isinstance(quotes, dict):
                    self._latest_quotes[symbol] = quotes
                    best_bid = quotes.get("best_bid")
                    best_ask = quotes.get("best_ask")
                    if best_bid and best_ask:
                        try:
                            # Update tick price with mid-market if valid
                            mid = (float(best_bid) + float(best_ask)) / 2.0
                            if symbol not in self._latest_prices:
                                self._latest_prices[symbol] = mid
                        except (ValueError, TypeError):
                            pass
                return

            # 2. Process candlestick messages (candlestick_1m, candlestick_5m, candlestick_15m)
            if msg_type.startswith("candlestick_"):
                symbol = payload.get("symbol", "").upper()
                resolution = payload.get("resolution", "")
                if not resolution and "_" in msg_type:
                    resolution = msg_type.split("_", 1)[1]

                if not symbol or not resolution:
                    return

                # Record real-time tick price from candlestick
                try:
                    cur_close = float(payload.get("close", 0.0))
                    if cur_close > 0:
                        self._latest_prices[symbol] = cur_close
                except (ValueError, TypeError):
                    cur_close = 0.0

                raw_start_time = payload.get("candle_start_time", payload.get("time", 0))
                open_time_ms = self._normalize_time_to_ms(raw_start_time)
                sec = self._resolution_seconds(resolution)
                close_time_ms = open_time_ms + (sec * 1000)

                key = (symbol, resolution)
                existing = self._forming_candles.get(key)

                op = float(payload.get("open", cur_close))
                hi = float(payload.get("high", cur_close))
                lo = float(payload.get("low", cur_close))
                cl = cur_close
                vol = float(payload.get("volume", 0.0))

                if existing is not None:
                    if open_time_ms > existing.open_time:
                        # Preceding candle has completed and rolled over!
                        existing.is_closed = True
                        seen_key = (symbol, resolution, existing.open_time)
                        if seen_key not in self._seen_candles:
                            self._seen_candles.add(seen_key)
                            self.store.add_candle(symbol, resolution, existing)
                            self.new_candle_event.set()

                        # Start new forming candle
                        self._forming_candles[key] = Candle(
                            open_time=open_time_ms,
                            open=op,
                            high=hi,
                            low=lo,
                            close=cl,
                            volume=vol,
                            close_time=close_time_ms,
                            is_closed=False
                        )
                    else:
                        # Update ongoing forming candle
                        existing.high = max(existing.high, hi)
                        existing.low = min(existing.low, lo) if existing.low > 0 else lo
                        existing.close = cl
                        existing.volume = vol
                else:
                    # Initial candle observed for this (symbol, resolution)
                    self._forming_candles[key] = Candle(
                        open_time=open_time_ms,
                        open=op,
                        high=hi,
                        low=lo,
                        close=cl,
                        volume=vol,
                        close_time=close_time_ms,
                        is_closed=False
                    )

                if len(self._seen_candles) > 10000:
                    self._seen_candles.clear()

        except Exception as e:
            logger.error(f"Error handling Delta WS message: {e}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic ping to Delta Exchange WebSocket to keep connection healthy."""
        while self._running:
            try:
                await asyncio.sleep(15)
                if self.is_connected and self._ws is not None:
                    await self._ws.send(json.dumps({"type": "ping"}))
            except Exception as e:
                logger.debug(f"Delta WS ping exception: {e}")

    async def _monitor_stale(self) -> None:
        start_time = time.time()
        while self._running:
            await asyncio.sleep(5)
            is_initial_stall = (self._last_message_time == 0.0 and (time.time() - start_time) > 15.0)
            is_runtime_stall = (self._last_message_time > 0 and (time.time() - self._last_message_time) > self.stale_data_seconds)
            if is_initial_stall or is_runtime_stall:
                logger.warning(
                    f"Delta WebSocket data is stale! No messages received in last {self.stale_data_seconds}s. Forcing reconnect..."
                )
                if self._ws is not None:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                try:
                    await self.bootstrap_historical_candles(limit=5)
                except Exception as e:
                    logger.debug(f"Delta REST candle fallback failed: {e}")

    async def run(self) -> None:
        self._running = True
        stale_monitor_task = asyncio.create_task(self._monitor_stale())
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        sub_payload = self._build_subscription_payload()
        sub_str = json.dumps(sub_payload)

        retry_count = 0
        base_delay = float(self.backoff_base)

        while self._running:
            try:
                logger.info(f"Connecting to Delta Exchange WS: {self.ws_url}")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=8,
                    close_timeout=3,
                ) as ws:
                    self._ws = ws
                    logger.info(f"Connected to Delta WS ({self.ws_url}). Subscribing to {self.symbols} candlestick and ticker channels...")
                    await ws.send(sub_str)
                    retry_count = 0  # Reset retry count on successful handshake
                    self._last_message_time = time.time()

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(message)

            except (ConnectionClosed, WebSocketException, Exception) as e:
                logger.warning(f"Delta WebSocket disconnected: {e}")

            if not self._running:
                break

            retry_count += 1
            if retry_count > self.max_retries:
                logger.error("Max retries reached for Delta WS client. Stopping.")
                break

            delay = min(10.0, base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1))
            logger.info(f"Reconnecting to Delta WS in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

        self._running = False
        stale_monitor_task.cancel()
        heartbeat_task.cancel()

    async def bootstrap_historical_candles(self, limit: int = 100) -> int:
        """
        Pre-seed CandleStore with historical closed candles directly from Delta Exchange REST API.
        GET /v2/history/candles?symbol={symbol}&resolution={tf}&start={start}&end={end}
        Ensures market structure and indicator requirements (>= 50 candles) are immediately met.
        """
        total_bootstrapped = 0
        logger.info(f"Bootstrapping {limit} historical candles from Delta Exchange for {self.symbols} across {self.timeframes}...")

        headers = {
            "User-Agent": "python-scalping-bot",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = []
            for sym in self.symbols:
                for tf in self.timeframes:
                    tasks.append(self._fetch_delta_klines(session, sym, tf, limit))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, int):
                    total_bootstrapped += res
                elif isinstance(res, Exception):
                    logger.debug(f"Historical candle task failed: {res}")

        if total_bootstrapped > 0:
            self._last_message_time = time.time()
            self.new_candle_event.set()
            logger.info(f"Successfully bootstrapped {total_bootstrapped} historical candles from Delta Exchange.")
        else:
            logger.warning("No historical candles bootstrapped from Delta Exchange. Strategy will wait for live WebSocket candles.")

        return total_bootstrapped

    async def _fetch_delta_klines(self, session: aiohttp.ClientSession, symbol: str, resolution: str, limit: int) -> int:
        sym_upper = symbol.upper()
        sec = self._resolution_seconds(resolution)
        now = int(time.time())
        # Fetch limit + 2 candles worth of time window to account for current unclosed candle
        start = now - ((limit + 2) * sec)

        url = f"{self.rest_url}/v2/history/candles?symbol={sym_upper}&resolution={resolution}&start={start}&end={now}"

        raw_candles = None
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict) and "result" in data:
                        raw_candles = data["result"]
                    elif isinstance(data, list):
                        raw_candles = data
        except Exception as e:
            logger.debug(f"Delta candles fetch failed for {sym_upper} {resolution}: {e}")

        if not raw_candles or not isinstance(raw_candles, list):
            return 0

        # Sort chronological (oldest to newest) since Delta returns descending by time
        sorted_candles = sorted(raw_candles, key=lambda c: int(c.get("time", 0)))

        # Exclude the latest candle if it is currently in progress (within duration of now)
        if len(sorted_candles) > 1:
            latest_time = int(sorted_candles[-1].get("time", 0))
            if (now - latest_time) < sec:
                sorted_candles = sorted_candles[:-1]

        # Truncate to limit
        if len(sorted_candles) > limit:
            sorted_candles = sorted_candles[-limit:]

        count = 0
        for c in sorted_candles:
            try:
                t_sec = int(c.get("time", 0))
                open_time_ms = t_sec * 1000
                close_time_ms = (t_sec + sec) * 1000

                candle = Candle(
                    open_time=open_time_ms,
                    open=float(c.get("open", 0.0)),
                    high=float(c.get("high", 0.0)),
                    low=float(c.get("low", 0.0)),
                    close=float(c.get("close", 0.0)),
                    volume=float(c.get("volume", 0.0)),
                    close_time=close_time_ms,
                    is_closed=True
                )
                self.store.add_candle(sym_upper, resolution, candle)
                seen_key = (sym_upper, resolution, open_time_ms)
                self._seen_candles.add(seen_key)
                count += 1
            except Exception:
                continue

        if sorted_candles:
            try:
                last_close = float(sorted_candles[-1].get("close", 0.0))
                if last_close > 0:
                    self._latest_prices[sym_upper] = last_close
            except Exception:
                pass

        return count

    async def stop(self) -> None:
        logger.info("Stopping Delta WS Client...")
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
