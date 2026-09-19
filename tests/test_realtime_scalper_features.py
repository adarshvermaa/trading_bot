import pytest
import numpy as np
from src.portfolio.account import AccountManager, Position
from src.execution.delta import DeltaExchangeClient, PaperPosition
from src.strategy.structure import MarketStructure
from src.strategy.signals import SignalGenerator

def test_position_realtime_pnl_update():
    mgr = AccountManager(is_paper=True, initial_paper_balance=10000.0)
    mgr.update_paper_position(
        symbol="BTCUSD",
        side="LONG",
        price=80000.0,
        size=100,
        leverage=150,
        margin=53.33,
        notional=8000.0,
        sl=79600.0,
        tp=80800.0,
    )
    pos = mgr.positions["BTCUSD"]
    assert pos.current_price == 80000.0
    assert pos.unrealized_pnl == 0.0
    
    mgr.update_current_price("BTCUSD", 80500.0)
    pos_dict = mgr.get_position_dict()
    assert pos_dict["current_price"] == 80500.0
    assert pos.unrealized_pnl > 0
    assert pos_dict["unrealized_pnl"] == pos.unrealized_pnl

    mgr.update_current_price("BTCUSD", 79800.0)
    pos_dict = mgr.get_position_dict()
    assert pos_dict["current_price"] == 79800.0
    assert pos.unrealized_pnl < 0
    assert pos_dict["unrealized_pnl"] == pos.unrealized_pnl

def test_paper_sl_tp_detection():
    client = DeltaExchangeClient(live_trading=False)
    client.paper_account.positions[1] = PaperPosition(
        product_id=1,
        symbol="BTCUSD",
        size=10.0,
        entry_price=80000.0,
        bracket_sl=79600.0,
        bracket_tp=80800.0,
        side="buy"
    )
    assert client.check_paper_sl_tp(1, 80200.0) is None
    assert client.check_paper_sl_tp(1, 79550.0) == "SL_HIT"
    assert client.check_paper_sl_tp(1, 80850.0) == "TP_HIT"
    
    client.paper_account.positions[2] = PaperPosition(
        product_id=2,
        symbol="ETHUSD",
        size=-50.0,
        entry_price=3000.0,
        bracket_sl=3030.0,
        bracket_tp=2940.0,
        side="sell"
    )
    assert client.check_paper_sl_tp(2, 2990.0) is None
    assert client.check_paper_sl_tp(2, 3035.0) == "SL_HIT"
    assert client.check_paper_sl_tp(2, 2930.0) == "TP_HIT"

def test_support_resistance_extraction():
    ms = MarketStructure(lookback=2)
    highs = np.array([100, 102, 110, 105, 103, 98, 92, 95, 99, 101], dtype=float)
    lows  = np.array([98,  100, 106, 101,  97, 94, 88, 91, 95,  98], dtype=float)
    closes = np.array([99, 101, 108, 103, 100, 96, 90, 94, 98, 100], dtype=float)
    
    sup_list, res_list, nearest_sup, nearest_res = ms.get_support_resistance_levels(highs, lows, closes)
    assert 110.0 in res_list
    assert 88.0 in sup_list
    assert nearest_res == 110.0
    assert nearest_sup == 88.0

def test_position_health_evaluation():
    sg = SignalGenerator()
    n = 30
    candles_1m = [
        {"open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i, "close": 100.5 + i, "volume": 10.0}
        for i in range(n)
    ]
    candles_5m = [
        {"open": 100.0 + i*5, "high": 105.0 + i*5, "low": 98.0 + i*5, "close": 104.0 + i*5, "volume": 50.0}
        for i in range(10)
    ]
    
    status, reason = sg.evaluate_position_health(
        position_side="LONG",
        entry_price=120.0,
        current_price=130.0,
        candles_1m=candles_1m,
        candles_5m=candles_5m,
        nearest_support=115.0,
        nearest_resistance=140.0
    )
    assert status in ("STRONG", "WEAK")

@pytest.mark.asyncio
async def test_btc_eth_asset_ranking():
    from src.main import ScalpingBot
    from src.config import load_config
    bot = ScalpingBot(config=load_config(), mode="paper")
    
    # Pre-seed candles for both BTCUSD and ETHUSD
    for sym in ["BTCUSD", "ETHUSD"]:
        bin_sym = bot.config.strategy.assets.binance_symbol_map[sym]
        for tf in ["1m", "5m", "15m"]:
            for i in range(60):
                from src.data.binance_ws import Candle
                c = Candle(
                    open_time=1000 + i*60,
                    open=100.0 + i,
                    high=102.0 + i,
                    low=99.0 + i,
                    close=101.0 + i,
                    volume=50.0 + i,
                    close_time=1000 + (i+1)*60,
                    is_closed=True
                )
                bot.binance_ws.candle_store.add_candle(bin_sym, tf, c)
                
    best = await bot._scan_assets()
    assert len(bot._asset_rankings) == 2
    symbols_ranked = [e["symbol"] for e in bot._asset_rankings]
    assert "BTCUSD" in symbols_ranked
    assert "ETHUSD" in symbols_ranked
    # Verify rankings string exists
    assert "#1" in bot._monitored_market["rankings"]
    assert "#2" in bot._monitored_market["rankings"]
