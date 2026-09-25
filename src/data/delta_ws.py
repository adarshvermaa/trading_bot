import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Any
import aiohttp
import numpy as np
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
    def __init__(self, max_len: int = 5000):
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

        self.store = CandleStore(max_len=5000)
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

        # HTF levels cache: symbol -> {"PDH": ..., "PDL": ..., "PDC": ..., "POC": ...}
        self._htf_levels: Dict[str, Dict[str, float]] = {}

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
    def calculate_htf_levels(candles: List[Candle]) -> Dict[str, float]:
        """Calculate Previous Day High (PDH), Low (PDL), Close (PDC), Volume POC, VAH, VAL, and Asian H/L.
        
        Uses closed candles across the recent 24h-48h window and up to 168h (1 week).
        """
        if not candles:
            return {
                "PWH": 0.0, "PWL": 0.0,
                "PDH": 0.0, "PDL": 0.0, "PDC": 0.0, "POC": 0.0,
                "VAH": 0.0, "VAL": 0.0, "ASIAN_HIGH": 0.0, "ASIAN_LOW": 0.0,
            }

        # Daily window: last 24 candles if 1h, or last 96 candles if 15m
        day_window = candles[-25:-1] if len(candles) >= 25 else candles
        if not day_window:
            day_window = candles

        pdh = float(max(c.high for c in day_window))
        pdl = float(min(c.low for c in day_window))
        pdc = float(day_window[-1].close)

        # Weekly window: last 168 candles if 1h (7 days), or last 7 candles if 1d
        if len(candles) >= 168:
            week_window = candles[-169:-1]
        elif len(candles) >= 48:
            week_window = candles[:-1]
        else:
            week_window = candles

        pwh = float(max(c.high for c in week_window))
        pwl = float(min(c.low for c in week_window))

        # Deep Volume Profile across all available candles (up to 3000 candles)
        vp_sample = candles[-3000:] if len(candles) > 3000 else candles
        hi_arr = np.array([c.high for c in vp_sample])
        lo_arr = np.array([c.low for c in vp_sample])
        cl_arr = np.array([c.close for c in vp_sample])
        vol_arr = np.array([c.volume for c in vp_sample])

        from src.strategy.structure import compute_volume_profile, calculate_asian_range
        poc, vah, val = compute_volume_profile(hi_arr, lo_arr, cl_arr, vol_arr)
        asian_high, asian_low = calculate_asian_range(candles)

        return {
            "PWH": round(pwh, 2),
            "PWL": round(pwl, 2),
            "PDH": round(pdh, 2),
            "PDL": round(pdl, 2),
            "PDC": round(pdc, 2),
            "POC": round(float(poc), 2),
            "VAH": round(float(vah), 2),
            "VAL": round(float(val), 2),
            "ASIAN_HIGH": round(float(asian_high), 2),
            "ASIAN_LOW": round(float(asian_low), 2),
        }

    def get_htf_levels(self, symbol: str) -> Dict[str, float]:
        """Get calculated HTF levels (PWH, PWL, PDH, PDL, PDC, POC, VAH, VAL, Asian H/L) for a symbol."""
        sym_upper = symbol.upper()
        if sym_upper in self._htf_levels and self._htf_levels[sym_upper].get("PDH", 0.0) > 0:
            return self._htf_levels[sym_upper]

        # Calculate from 1h candles if present, or fallback to 15m candles
        c_1h = self.store.get_candles(sym_upper, "1h")
        if c_1h and len(c_1h) >= 10:
            levels = self.calculate_htf_levels(c_1h)
            self._htf_levels[sym_upper] = levels
            return levels

        c_15m = self.store.get_candles(sym_upper, "15m")
        if c_15m:
            levels = self.calculate_htf_levels(c_15m)
            self._htf_levels[sym_upper] = levels
            return levels

        return {
            "PWH": 0.0, "PWL": 0.0,
            "PDH": 0.0, "PDL": 0.0, "PDC": 0.0, "POC": 0.0,
            "VAH": 0.0, "VAL": 0.0, "ASIAN_HIGH": 0.0, "ASIAN_LOW": 0.0,
        }

    @staticmethod
    def _resolution_seconds(resolution: str) -> int:
        res_map = {
            "5s": 5, "15s": 15, "30s": 30,
            "1m": 60, "3m": 180, "5m": 300,
            "15m": 900, "30m": 1800, "45m": 2700,
            "1h": 3600, "2h": 7200, "4h": 14400,
            "1d": 86400, "1w": 604800,
        }
        return res_map.get(resolution, 60)

    @staticmethod
    def synthesize_candles_from_base(
        base_candles: List[Candle], target_seconds: int
    ) -> List[Candle]:
        """Synthesize higher-timeframe candles from base candles (e.g. 3m from 1m, 45m from 15m)."""
        if not base_candles or target_seconds <= 0:
            return base_candles

        target_ms = target_seconds * 1000
        buckets: Dict[int, List[Candle]] = {}

        for c in base_candles:
            bucket_key = (c.open_time // target_ms) * target_ms
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(c)

        synthesized = []
        for bucket_key in sorted(buckets.keys()):
            group = buckets[bucket_key]
            if not group:
                continue
            synth = Candle(
                open_time=bucket_key,
                open=group[0].open,
                high=max(item.high for item in group),
                low=min(item.low for item in group),
                close=group[-1].close,
                volume=sum(item.volume for item in group),
                close_time=bucket_key + target_ms,
                is_closed=group[-1].is_closed,
            )
            synthesized.append(synth)

        return synthesized

    @staticmethod
    def synthesize_micro_candles_from_1m(candles_1m: List[Candle]) -> List[Candle]:
        """Interpolate 1m candles into 5s synthetic candles to bootstrap micro timeframes."""
        micro_5s = []
        for c in candles_1m:
            base_time = c.open_time
            n_sub = 12
            prices = np.linspace(c.open, c.close, n_sub)
            vol_per_bar = c.volume / n_sub if n_sub > 0 else 0.0
            for i in range(n_sub):
                bar_time = base_time + (i * 5000)
                sub_open = prices[i - 1] if i > 0 else c.open
                sub_close = prices[i]
                sub_high = max(sub_open, sub_close, c.high if i == n_sub // 2 else max(sub_open, sub_close))
                sub_low = min(sub_open, sub_close, c.low if i == n_sub // 4 else min(sub_open, sub_close))
                micro_5s.append(Candle(
                    open_time=bar_time,
                    open=float(sub_open),
                    high=float(sub_high),
                    low=float(sub_low),
                    close=float(sub_close),
                    volume=float(vol_per_bar),
                    close_time=bar_time + 5000,
                    is_closed=True,
                ))
        return micro_5s

    @staticmethod
    def synthesize_sub_minute_candles(
        candles_5s: List[Candle], target_seconds: int = 15
    ) -> List[Candle]:
        """Synthesize sub-minute candles (e.g. 15s, 30s) from raw 5s candles."""
        return DeltaWSClient.synthesize_candles_from_base(candles_5s, target_seconds)

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
        subscribed_tfs = list(dict.fromkeys(self.timeframes + ["1m"]))
        channels = [{"name": f"candlestick_{tf}", "symbols": self.symbols} for tf in subscribed_tfs]
        channels.append({"name": "v2/ticker", "symbols": self.symbols})
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
                        # Real-time sub-minute 5s candle synthesis from live ticks
                        now_ms = int(time.time() * 1000)
                        bucket_5s = (now_ms // 5000) * 5000
                        key_5s = (symbol, "5s")
                        existing_5s = self._forming_candles.get(key_5s)
                        if existing_5s is not None:
                            if bucket_5s > existing_5s.open_time:
                                existing_5s.is_closed = True
                                seen_key = (symbol, "5s", existing_5s.open_time)
                                if seen_key not in self._seen_candles:
                                    self._seen_candles.add(seen_key)
                                    self.store.add_candle(symbol, "5s", existing_5s)
                                    self.new_candle_event.set()
                                for synth_sec in (15, 30):
                                    synth_tf = f"{synth_sec}s"
                                    synth_bucket = (existing_5s.open_time // (synth_sec * 1000)) * (synth_sec * 1000)
                                    s_key = (symbol, synth_tf)
                                    s_existing = self._forming_candles.get(s_key)
                                    if s_existing is None or s_existing.open_time < synth_bucket:
                                        if s_existing is not None:
                                            s_existing.is_closed = True
                                            self.store.add_candle(symbol, synth_tf, s_existing)
                                        self._forming_candles[s_key] = Candle(
                                            open_time=synth_bucket,
                                            open=existing_5s.open,
                                            high=existing_5s.high,
                                            low=existing_5s.low,
                                            close=existing_5s.close,
                                            volume=existing_5s.volume,
                                            close_time=synth_bucket + (synth_sec * 1000),
                                            is_closed=False,
                                        )
                                    else:
                                        s_existing.high = max(s_existing.high, existing_5s.high)
                                        s_existing.low = min(s_existing.low, existing_5s.low)
                                        s_existing.close = existing_5s.close
                                        s_existing.volume += existing_5s.volume

                                self._forming_candles[key_5s] = Candle(
                                    open_time=bucket_5s,
                                    open=close_price,
                                    high=close_price,
                                    low=close_price,
                                    close=close_price,
                                    volume=0.0,
                                    close_time=bucket_5s + 5000,
                                    is_closed=False
                                )
                            else:
                                existing_5s.high = max(existing_5s.high, close_price)
                                existing_5s.low = min(existing_5s.low, close_price)
                                existing_5s.close = close_price
                        else:
                            self._forming_candles[key_5s] = Candle(
                                open_time=bucket_5s,
                                open=close_price,
                                high=close_price,
                                low=close_price,
                                close=close_price,
                                volume=0.0,
                                close_time=bucket_5s + 5000,
                                is_closed=False
                            )
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

                        if resolution == "5s":
                            for synth_sec in (15, 30):
                                synth_tf = f"{synth_sec}s"
                                synth_bucket = (existing.open_time // (synth_sec * 1000)) * (synth_sec * 1000)
                                s_key = (symbol, synth_tf)
                                s_existing = self._forming_candles.get(s_key)
                                if s_existing is None or s_existing.open_time < synth_bucket:
                                    if s_existing is not None:
                                        s_existing.is_closed = True
                                        self.store.add_candle(symbol, synth_tf, s_existing)
                                    self._forming_candles[s_key] = Candle(
                                        open_time=synth_bucket,
                                        open=existing.open,
                                        high=existing.high,
                                        low=existing.low,
                                        close=existing.close,
                                        volume=existing.volume,
                                        close_time=synth_bucket + (synth_sec * 1000),
                                        is_closed=False,
                                    )
                                else:
                                    s_existing.high = max(s_existing.high, existing.high)
                                    s_existing.low = min(s_existing.low, existing.low)
                                    s_existing.close = existing.close
                                    s_existing.volume += existing.volume

                        if resolution == "1m":
                            synth_sec = 180  # 3m
                            synth_tf = "3m"
                            synth_bucket = (existing.open_time // (synth_sec * 1000)) * (synth_sec * 1000)
                            s_key = (symbol, synth_tf)
                            s_existing = self._forming_candles.get(s_key)
                            if s_existing is None or s_existing.open_time < synth_bucket:
                                if s_existing is not None:
                                    s_existing.is_closed = True
                                    self.store.add_candle(symbol, synth_tf, s_existing)
                                self._forming_candles[s_key] = Candle(
                                    open_time=synth_bucket,
                                    open=existing.open,
                                    high=existing.high,
                                    low=existing.low,
                                    close=existing.close,
                                    volume=existing.volume,
                                    close_time=synth_bucket + (synth_sec * 1000),
                                    is_closed=False,
                                )
                            else:
                                s_existing.high = max(s_existing.high, existing.high)
                                s_existing.low = min(s_existing.low, existing.low)
                                s_existing.close = existing.close
                                s_existing.volume += existing.volume

                        if resolution == "15m":
                            synth_sec = 2700  # 45m
                            synth_tf = "45m"
                            synth_bucket = (existing.open_time // (synth_sec * 1000)) * (synth_sec * 1000)
                            s_key = (symbol, synth_tf)
                            s_existing = self._forming_candles.get(s_key)
                            if s_existing is None or s_existing.open_time < synth_bucket:
                                if s_existing is not None:
                                    s_existing.is_closed = True
                                    self.store.add_candle(symbol, synth_tf, s_existing)
                                self._forming_candles[s_key] = Candle(
                                    open_time=synth_bucket,
                                    open=existing.open,
                                    high=existing.high,
                                    low=existing.low,
                                    close=existing.close,
                                    volume=existing.volume,
                                    close_time=synth_bucket + (synth_sec * 1000),
                                    is_closed=False,
                                )
                            else:
                                s_existing.high = max(s_existing.high, existing.high)
                                s_existing.low = min(s_existing.low, existing.low)
                                s_existing.close = existing.close
                                s_existing.volume += existing.volume

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

    async def bootstrap_historical_candles(
        self, limit: int = 3000, timeframes: Optional[List[str]] = None
    ) -> int:
        """
        Pre-seed CandleStore with up to 3000+ historical closed candles directly from Delta Exchange REST API.
        GET /v2/history/candles?symbol={symbol}&resolution={tf}&start={start}&end={end}
        Ensures market structure, Volume Profile, and deep indicator requirements (EMA 200/50, HTF) are fully met.
        """
        total_bootstrapped = 0
        target_tfs = timeframes if timeframes is not None else self.timeframes
        logger.info(f"Bootstrapping historical candles (up to {limit}) from Delta Exchange for {self.symbols} across {target_tfs}...")

        headers = {
            "User-Agent": "python-scalping-bot",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = []
            for sym in self.symbols:
                for tf in target_tfs:
                    tasks.append(self._fetch_delta_klines(session, sym, tf, limit))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, int):
                    total_bootstrapped += res
                elif isinstance(res, Exception):
                    logger.debug(f"Historical candle task failed: {res}")

        # Synthesize missing timeframes (3m, 45m, and sub-minute candles)
        for sym in self.symbols:
            c_1m = self.store.get_candles(sym, "1m")
            if c_1m:
                # 1. Synthesize 3m candles from 1m
                synth_3m = self.synthesize_candles_from_base(c_1m, target_seconds=180)
                for sc in synth_3m:
                    self.store.add_candle(sym, "3m", sc)
                    self._seen_candles.add((sym, "3m", sc.open_time))
                total_bootstrapped += len(synth_3m)

                # 2. If 5s candles are not populated via REST, bootstrap micro candles from 1m
                c_5s = self.store.get_candles(sym, "5s")
                if not c_5s:
                    synth_5s = self.synthesize_micro_candles_from_1m(c_1m[-100:])
                    for sc in synth_5s:
                        self.store.add_candle(sym, "5s", sc)
                        self._seen_candles.add((sym, "5s", sc.open_time))
                    total_bootstrapped += len(synth_5s)

            c_15m = self.store.get_candles(sym, "15m")
            if c_15m:
                # 3. Synthesize 45m candles from 15m
                synth_45m = self.synthesize_candles_from_base(c_15m, target_seconds=2700)
                for sc in synth_45m:
                    self.store.add_candle(sym, "45m", sc)
                    self._seen_candles.add((sym, "45m", sc.open_time))
                total_bootstrapped += len(synth_45m)

            # 4. Synthesize 15s and 30s from 5s
            c_5s_now = self.store.get_candles(sym, "5s")
            if c_5s_now:
                for target_sec in (15, 30):
                    synth_tf = f"{target_sec}s"
                    synth_candles = self.synthesize_sub_minute_candles(c_5s_now, target_seconds=target_sec)
                    for sc in synth_candles:
                        self.store.add_candle(sym, synth_tf, sc)
                        self._seen_candles.add((sym, synth_tf, sc.open_time))
                    total_bootstrapped += len(synth_candles)

        # Compute HTF levels for all symbols
        for sym in self.symbols:
            self.get_htf_levels(sym)

        if total_bootstrapped > 0:
            self._last_message_time = time.time()
            self.new_candle_event.set()
            logger.info(f"Successfully bootstrapped {total_bootstrapped} historical candles from Delta Exchange.")
        else:
            logger.warning("No historical candles bootstrapped from Delta Exchange. Strategy will wait for live WebSocket candles.")

        return total_bootstrapped

    async def bootstrap_htf_candles(self, limit: int = 168) -> int:
        """Fetch 1h candles (up to 168 = 7 days) specifically for Weekly & Daily S/R levels."""
        total = 0
        headers = {"User-Agent": "python-scalping-bot", "Content-Type": "application/json"}
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [self._fetch_delta_klines(session, sym, "1h", limit) for sym in self.symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, int):
                    total += res
                elif isinstance(res, Exception):
                    logger.debug(f"HTF historical candle task failed: {res}")

        for sym in self.symbols:
            self.get_htf_levels(sym)
        return total

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
