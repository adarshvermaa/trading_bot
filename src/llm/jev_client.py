import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

from src.config import JevConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class JevPreTradeAudit:
    is_trap: float
    direction_bias: str
    direction_confidence: float
    direction_probabilities: Dict[str, float]
    setup_grade: float
    grade_confidence: float
    cost_headwind: float
    is_vetoed: bool
    veto_reason: str
    is_boosted: bool
    boost_amount: float
    latency_ms: float


@dataclass
class JevActiveTradeAudit:
    momentum_state: str
    momentum_confidence: float
    momentum_probabilities: Dict[str, float]
    should_exit_early: float
    latency_ms: float


@dataclass
class JevMistakeForensics:
    root_cause: str
    confidence: float
    probabilities: Dict[str, float]
    was_preventable: float
    latency_ms: float


class JevClient:
    """
    High-performance async client for TypeSafe AI's Jev (System One) model.

    Evaluates typed questions (Noul, Choice, Score) directly against structured
    market state in parallel with zero-lag fallback to local ONNX + ICT rules.
    """

    def __init__(
        self,
        api_key: str,
        config: JevConfig,
        base_url: str = "https://api.typesafe.ai/v1/systemone",
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.config = config
        self.base_url = base_url
        self.enabled = bool(self.api_key and config.enabled)

        self._session: Optional[aiohttp.ClientSession] = None
        self._active_model_name: str = config.model

        # Real-time Telemetry
        self.total_requests: int = 0
        self.total_vetoes: int = 0
        self.total_boosts: int = 0
        self.total_fallbacks: int = 0
        self.last_latency_ms: float = 0.0
        self.last_audit: Optional[JevPreTradeAudit] = None

        if not self.enabled:
            logger.info("Jev AI Client is disabled or API key is not configured.")
        else:
            logger.info(f"Jev AI Client initialized (model={config.model}, timeout={config.timeout_ms}ms)")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=max(1.0, self.config.timeout_ms / 1000.0 * 2))
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=60, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close underlying connection pool cleanly."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def get_status(self) -> str:
        if not self.enabled:
            return "DISABLED"
        v_str = f"V:{self.total_vetoes}" if self.total_vetoes > 0 else ""
        b_str = f"B:{self.total_boosts}" if self.total_boosts > 0 else ""
        extra = f" ({v_str} {b_str})".strip() if (v_str or b_str) else ""
        lat = f"{self.last_latency_ms:.0f}ms" if self.last_latency_ms > 0 else "READY"
        return f"ACTIVE [{self._active_model_name} | {lat}]{extra}"

    async def _query_system_one(
        self, state: str, questions: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Low-level query to Jev System One endpoint with strict timeout guardrail.
        Returns parsed 'answers' dictionary or None on timeout/error.
        """
        if not self.enabled:
            return None

        self.total_requests += 1
        t0 = time.perf_counter()
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "state": state,
            "questions": questions,
        }

        timeout_sec = self.config.timeout_ms / 1000.0
        try:
            async with asyncio.timeout(timeout_sec):
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    latency = (time.perf_counter() - t0) * 1000.0
                    self.last_latency_ms = latency

                    if resp.status == 200:
                        data = await resp.json()
                        self._active_model_name = data.get("model", self.config.model)
                        return data.get("answers")
                    else:
                        err_text = await resp.text()
                        logger.warning(
                            f"[JEV API HTTP {resp.status}] in {latency:.1f}ms: {err_text}. "
                            "Falling back to local rules."
                        )
                        self.total_fallbacks += 1
                        return None

        except asyncio.TimeoutError:
            latency = (time.perf_counter() - t0) * 1000.0
            self.last_latency_ms = latency
            self.total_fallbacks += 1
            logger.warning(
                f"[JEV TIMEOUT] Call exceeded {self.config.timeout_ms}ms ({latency:.1f}ms). "
                "Immediate zero-lag fallback to local ONNX + ICT rules."
            )
            return None
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            self.last_latency_ms = latency
            self.total_fallbacks += 1
            logger.warning(
                f"[JEV EXCEPTION] {type(e).__name__}: {e}. "
                "Immediate zero-lag fallback to local ONNX + ICT rules."
            )
            return None

    # -----------------------------------------------------------------------
    # 1. Pre-Trade Confluence Gatekeeper (Trap & Direction & Quality Audit)
    # -----------------------------------------------------------------------

    async def evaluate_pre_trade_setup(
        self,
        symbol: str,
        direction: str,
        indicators: Dict[str, Any],
        structure: Any,
        fingerprint: Any,
        session_info: Any = None,
    ) -> Optional[JevPreTradeAudit]:
        """
        Evaluate candidate setup against Jev System One model before order placement.

        Returns JevPreTradeAudit with trap probability, direction bias, setup grade,
        and verdict (vetoed vs boosted), or None if Jev is disabled/timed out.
        """
        if not self.enabled:
            return None

        def _safe_float(val: Any, default: float) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        rsi = _safe_float(indicators.get("rsi"), 50.0)
        adx = _safe_float(indicators.get("adx"), 20.0)
        rel_vol = _safe_float(indicators.get("relative_volume"), 1.0)
        vwap_pos = str(indicators.get("vwap_position", "NEUTRAL"))
        atr = _safe_float(indicators.get("atr"), 0.0)

        bias_15m = getattr(structure, "bias_15m", "NEUTRAL")
        bos_5m = getattr(structure, "bos_5m", False)
        choch_5m = getattr(structure, "choch_5m", False)
        sweep_5m = getattr(structure, "liquidity_sweep_5m", False)
        disp_1m = getattr(structure, "displacement_1m", False)
        retest_1m = getattr(structure, "retest_1m", False)
        is_bull_trap = bool(getattr(structure, "is_bull_trap", False) or _safe_float(getattr(fingerprint, "is_bull_trap", 0), 0.0) > 0.5)
        is_bear_trap = bool(getattr(structure, "is_bear_trap", False) or _safe_float(getattr(fingerprint, "is_bear_trap", 0), 0.0) > 0.5)
        is_judas = bool(getattr(structure, "is_judas_swing", False) or _safe_float(getattr(fingerprint, "is_judas_swing", 0), 0.0) > 0.5)
        spread_bps = _safe_float(getattr(fingerprint, "spread_bps", 0.8), 0.8)
        session_zone = getattr(getattr(session_info, "zone", None), "value", "NORMAL") if session_info else "NORMAL"

        state = (
            f"Symbol: {symbol}, Scalping Timeframes: 15m/5m/1m. "
            f"Candidate Entry Direction: {direction}. "
            f"Momentum & Trend: RSI={rsi:.1f}, ADX={adx:.1f}, Relative Volume={rel_vol:.2f}x, VWAP={vwap_pos}, ATR={atr:.2f}. "
            f"ICT Structure: 15m Bias={bias_15m}, 5m BOS={bos_5m}, 5m CHoCH={choch_5m}, 1m Displacement={disp_1m}, 1m Retest={retest_1m}. "
            f"Liquidity & Traps: 5m Sweep={sweep_5m}, Bull Trap={is_bull_trap}, Bear Trap={is_bear_trap}, Judas Swing={is_judas}. "
            f"Execution Environment: Bid-Ask Spread={spread_bps:.1f} bps, Market Session={session_zone}."
        )

        questions = {
            "is_trap": {
                "type": "noul",
                "instructions": (
                    "Is this setup a bull trap, bear trap, liquidity sweep fakeout, or retail trap "
                    "designed to reverse sharply on entering traders?"
                ),
            },
            "direction_bias": {
                "type": "choice",
                "instructions": "What is the dominant high-probability institutional direction?",
                "criteria": {
                    "LONG": "High probability bullish continuation or breakout",
                    "SHORT": "High probability bearish reversal or breakdown",
                    "NEUTRAL": "Choppy sideways range, low edge, or conflicting signals",
                },
            },
            "setup_grade": {
                "type": "score",
                "instructions": "Rate the institutional quality and edge of this scalping setup from 0 to 4",
                "criteria": [
                    "Grade 0: F - Terrible, high trap risk, no edge",
                    "Grade 1: D - Weak, unconfirmed structure, poor volume",
                    "Grade 2: C - Average setup, moderate risk, borderline hurdle",
                    "Grade 3: B - High quality setup, strong structure and volume confluence",
                    "Grade 4: A+ - Premium institutional setup, clean displacement, low risk",
                ],
            },
            "cost_headwind": {
                "type": "noul",
                "instructions": (
                    "Are spread, fees, or slippage friction too high relative to the expected scalp profit target?"
                ),
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        # 1. Parse is_trap (Noul)
        trap_ans = answers.get("is_trap", {})
        is_trap_val = float(trap_ans.get("noul", 0.0))

        # 2. Parse direction_bias (Choice)
        dir_ans = answers.get("direction_bias", {})
        direction_bias = str(dir_ans.get("choice", "NEUTRAL")).upper()
        dir_conf = float(dir_ans.get("confidence", 0.0))
        dir_probs = {k: float(v) for k, v in dir_ans.get("probabilities", {}).items()}

        # 3. Parse setup_grade (Score 0-4)
        grade_ans = answers.get("setup_grade", {})
        setup_grade = float(grade_ans.get("score", 2.0))
        grade_conf = float(grade_ans.get("confidence", 0.0))

        # 4. Parse cost_headwind (Noul)
        cost_ans = answers.get("cost_headwind", {})
        cost_headwind = float(cost_ans.get("noul", 0.0))

        # --- Gating & Decision Logic ---
        is_vetoed = False
        veto_reason = ""
        is_boosted = False
        boost_amount = 0.0

        # Rule A: Trap Probability VETO
        if is_trap_val >= self.config.max_trap_probability:
            is_vetoed = True
            veto_reason = (
                f"Trap risk too high: {is_trap_val:.2f} >= max {self.config.max_trap_probability:.2f}"
            )

        # Rule B: Direction Conflict VETO
        elif direction_bias != direction and direction_bias != "NEUTRAL" and dir_conf >= self.config.min_confidence:
            is_vetoed = True
            veto_reason = (
                f"Direction conflict: Candidate is {direction}, but Jev bias is {direction_bias} "
                f"({dir_conf*100:.0f}% confidence)"
            )

        # Rule C: Cost Headwind VETO
        elif cost_headwind >= 0.70:
            is_vetoed = True
            veto_reason = (
                f"Cost friction headwind too high: spread/fee friction prob={cost_headwind:.2f}"
            )

        # Rule D: Setup Grade VETO (if setup is graded F or low D < 1.5)
        elif setup_grade < 1.5:
            is_vetoed = True
            veto_reason = f"Poor setup quality: grade {setup_grade:.2f}/4.0 < minimum 1.5"

        # Rule E: Confluence Quality BOOST
        elif (
            direction_bias == direction
            and dir_conf >= self.config.min_confidence
            and setup_grade >= self.config.min_setup_grade
            and is_trap_val <= 0.25
        ):
            is_boosted = True
            # Grade 2.5 gives +0.10, Grade 4.0 gives +0.175
            boost_amount = min(0.20, 0.10 + (setup_grade - self.config.min_setup_grade) * 0.05)

        if is_vetoed:
            self.total_vetoes += 1
            logger.warning(
                f"[JEV VETO] {symbol} {direction} blocked: {veto_reason} "
                f"(Grade={setup_grade:.2f}, Trap={is_trap_val:.2f}, Latency={self.last_latency_ms:.0f}ms)"
            )
        elif is_boosted:
            self.total_boosts += 1
            logger.info(
                f"[JEV BOOST] {symbol} {direction} confirmed! Score Boost=+{boost_amount:.2f} "
                f"(Grade={setup_grade:.2f}, Conf={dir_conf*100:.0f}%, Latency={self.last_latency_ms:.0f}ms)"
            )
        else:
            logger.info(
                f"[JEV AUDIT] {symbol} {direction} neutral pass: Grade={setup_grade:.2f}, "
                f"Trap={is_trap_val:.2f}, Latency={self.last_latency_ms:.0f}ms"
            )

        audit = JevPreTradeAudit(
            is_trap=is_trap_val,
            direction_bias=direction_bias,
            direction_confidence=dir_conf,
            direction_probabilities=dir_probs,
            setup_grade=setup_grade,
            grade_confidence=grade_conf,
            cost_headwind=cost_headwind,
            is_vetoed=is_vetoed,
            veto_reason=veto_reason,
            is_boosted=is_boosted,
            boost_amount=boost_amount,
            latency_ms=self.last_latency_ms,
        )
        self.last_audit = audit
        return audit

    # -----------------------------------------------------------------------
    # 2. Active Trade Health & Momentum Monitor
    # -----------------------------------------------------------------------

    async def evaluate_active_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        pnl: float,
        duration_seconds: float,
        rsi: float = 50.0,
        vwap_dist_pct: float = 0.0,
        structure_health_notes: str = "",
    ) -> Optional[JevActiveTradeAudit]:
        """
        Evaluate active open position momentum and determine if an early exit
        is required before stop loss is struck.
        """
        if not self.enabled or not self.config.enable_active_monitoring:
            return None

        pnl_pct = (
            ((current_price - entry_price) / entry_price * 100.0)
            if side.upper() in ("BUY", "LONG")
            else ((entry_price - current_price) / entry_price * 100.0)
        )

        state = (
            f"Active Position: {side.upper()} {symbol}. "
            f"Entry Price: ${entry_price:.2f}, Current Price: ${current_price:.2f}. "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%). Holding Duration: {duration_seconds:.0f} seconds. "
            f"Current RSI: {rsi:.1f}, VWAP distance: {vwap_dist_pct:+.2%}. "
            f"Structure Notes: {structure_health_notes or 'Normal pullback/advance'}."
        )

        questions = {
            "momentum_state": {
                "type": "choice",
                "instructions": "What is the current health and momentum state of this active position?",
                "criteria": {
                    "STRONG": "Bullish continuation remains intact, minor healthy pullback",
                    "FADING": "Momentum is stalling or slowing down",
                    "REVERSAL": "Sharp counter-trend breakdown or institutional reversal threatening stop loss",
                },
            },
            "should_exit_early": {
                "type": "noul",
                "instructions": (
                    "Should the bot exit immediately to avoid taking a full stop loss or severe drawdown?"
                ),
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        mom_ans = answers.get("momentum_state", {})
        mom_state = str(mom_ans.get("choice", "STRONG")).upper()
        mom_conf = float(mom_ans.get("confidence", 0.0))
        mom_probs = {k: float(v) for k, v in mom_ans.get("probabilities", {}).items()}

        exit_ans = answers.get("should_exit_early", {})
        should_exit = float(exit_ans.get("noul", 0.0))

        audit = JevActiveTradeAudit(
            momentum_state=mom_state,
            momentum_confidence=mom_conf,
            momentum_probabilities=mom_probs,
            should_exit_early=should_exit,
            latency_ms=self.last_latency_ms,
        )

        if should_exit >= 0.75 or (mom_state == "REVERSAL" and mom_conf >= 0.85):
            logger.warning(
                f"[JEV ACTIVE MONITOR] Warning on {symbol} {side}: state={mom_state} "
                f"(exit_prob={should_exit:.2f}, conf={mom_conf:.2f})"
            )

        return audit

    # -----------------------------------------------------------------------
    # 3. Post-Trade Mistake Forensics (Root Cause Classification)
    # -----------------------------------------------------------------------

    async def diagnose_stopped_out_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        close_reason: str,
        market_context: str = "",
    ) -> Optional[JevMistakeForensics]:
        """
        Perform root cause analysis on a losing trade using Jev Choice + Noul primitives.
        Returns classified root cause to update pattern memory.
        """
        if not self.enabled or not self.config.enable_mistake_forensics:
            return None

        state = (
            f"Stopped Out Trade: {side.upper()} {symbol}. "
            f"Entry Price: ${entry_price:.2f}, Exit Price: ${exit_price:.2f}. "
            f"Realized PnL: ${pnl:+.2f}. Triggered Close Reason: {close_reason}. "
            f"Context: {market_context or 'Trade struck stop loss during session'}"
        )

        questions = {
            "root_cause": {
                "type": "choice",
                "instructions": "What was the primary root cause of this loss?",
                "criteria": {
                    "CHOP_TRAP": "Entered in sideways low-liquidity chop without genuine breakout follow-through",
                    "SPREAD_FEE_SLIPPAGE": "High slippage or wide spread ate into scalp margin",
                    "LIQUIDITY_SWEEP": "Stop hunt or liquidity sweep took out tight stop before reversing",
                    "COUNTER_TREND": "Trading against dominant higher-timeframe trend direction",
                    "VALID_SETUP_BAD_LUCK": "Valid structural setup that failed due to normal statistical variance",
                },
            },
            "was_preventable": {
                "type": "noul",
                "instructions": (
                    "Was this loss preventable with better timing, wider sweep buffer, or avoiding chop?"
                ),
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        rc_ans = answers.get("root_cause", {})
        root_cause = str(rc_ans.get("choice", "VALID_SETUP_BAD_LUCK")).upper()
        confidence = float(rc_ans.get("confidence", 0.0))
        probs = {k: float(v) for k, v in rc_ans.get("probabilities", {}).items()}

        prev_ans = answers.get("was_preventable", {})
        was_preventable = float(prev_ans.get("noul", 0.0))

        logger.info(
            f"[JEV MISTAKE FORENSICS] {symbol} {side} Loss Classified: {root_cause} "
            f"(conf={confidence*100:.0f}%, preventable={was_preventable:.2f}, pnl=${pnl:.2f})"
        )

        return JevMistakeForensics(
            root_cause=root_cause,
            confidence=confidence,
            probabilities=probs,
            was_preventable=was_preventable,
            latency_ms=self.last_latency_ms,
        )
