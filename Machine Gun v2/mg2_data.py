# region imports
from AlgorithmImports import *
from execution import *
import numpy as np
from collections import deque
from datetime import datetime, timedelta
# endregion

"""
mg2_data.py — Feature Engineering for Machine Gun v2
=====================================================
Enhanced data pipeline to support the setup-driven entry architecture:
- Compression detection (BB-width squeeze tracking)
- Breakout freshness (bars since last N-bar high was set)
- Order-flow persistence (consecutive above-average volume bars)
- Kalman slope / Kalman history for trend confirmation
- Volume persistence counter
- Bars since volume ignition (new: staleness filter)
- Bars since VWAP reclaim (new: staleness filter)
- Cross-sectional RS rank across active universe
"""


def is_ready(c):
    return len(c['prices']) >= 20 and c['rsi'].IsReady and c['adx'].IsReady


def initialize_symbol(algo, symbol):
    algo.crypto_data[symbol] = {
        # ── Core price / volume ────────────────────────────────────────────
        'prices':    deque(maxlen=algo.lookback),
        'returns':   deque(maxlen=algo.lookback),
        'volume':    deque(maxlen=algo.lookback),
        'volume_ma': deque(maxlen=algo.medium_period),
        'dollar_volume': deque(maxlen=algo.lookback),
        'highs':     deque(maxlen=algo.lookback),
        'lows':      deque(maxlen=algo.lookback),

        # ── Built-in indicators ────────────────────────────────────────────
        'ema_ultra_short': ExponentialMovingAverage(algo.ultra_short_period),
        'ema_short':       ExponentialMovingAverage(algo.short_period),
        'ema_medium':      ExponentialMovingAverage(algo.medium_period),
        'ema_5':           ExponentialMovingAverage(5),
        'atr':             AverageTrueRange(14),
        'adx':             AverageDirectionalIndex(algo.adx_min_period),
        'rsi':             RelativeStrengthIndex(7),

        # ── Volatility ─────────────────────────────────────────────────────
        'volatility': deque(maxlen=algo.medium_period),

        # ── Spread tracking ────────────────────────────────────────────────
        'spreads':    deque(maxlen=algo.spread_median_window),

        # ── Scoring state ──────────────────────────────────────────────────
        'recent_net_scores': deque(maxlen=3),
        'last_price':   0,
        'trail_stop':   None,
        'trade_count_today': 0,
        'last_loss_time': None,

        # ── Relative strength vs BTC ───────────────────────────────────────
        'rs_vs_btc':        deque(maxlen=algo.medium_period),
        # Medium-term RS: 20-bar cumulative relative return vs BTC (scalar)
        'rs_vs_btc_medium': 0.0,

        # ── Z-score ────────────────────────────────────────────────────────
        'zscore':     deque(maxlen=algo.short_period),

        # ── Bollinger Bands (manual) ───────────────────────────────────────
        'bb_upper':   deque(maxlen=algo.short_period),
        'bb_lower':   deque(maxlen=algo.short_period),
        'bb_width':   deque(maxlen=60),   # extended for compression detection

        # ── Compression / expansion tracking (NEW) ─────────────────────────
        # Count of consecutive bars where BB width < compression threshold
        'compression_bars':     0,
        # Minimum BB width seen during current or most recent compression phase
        'compression_min_width': 999.0,
        # Median BB width over last 30 bars (updated every bar)
        'bb_width_median_30':    0.0,

        # ── Breakout freshness ─────────────────────────────────────────────
        # We track the "age" (bars) since price last broke above the prior 20-bar high
        'breakout_freshness':    999,   # 999 = no recent breakout
        'prev_range_high_20':    0.0,   # previous 20-bar high (before current bar)
        # The price level of the 20-bar high that was broken (used for failed-breakout exit)
        'breakout_level':        0.0,

        # ── Freshness staleness counters (NEW) ────────────────────────────
        # Bars since volume reached ignition level (VOL_IGNITION_MULT × baseline)
        'bars_since_vol_ignition':  999,
        # Bars since price last crossed VWAP from below (reclaim event)
        'bars_since_vwap_reclaim':  999,
        # Previous bar's above-VWAP status (for edge detection)
        '_prev_above_vwap':         False,

        # ── Cross-sectional rank (updated by update_market_context) ───────
        # Percentile rank of this symbol's short RS vs BTC (0.0–1.0; higher = stronger)
        'rs_rank':              0.5,

        # ── Order-flow persistence (NEW) ──────────────────────────────────
        # How many consecutive bars have had above-baseline volume
        'vol_persistence':  0,
        # Long-term volume baseline for persistence calculation
        'volume_long':      deque(maxlen=1440),

        # ── VWAP (rolling 20-bar) ─────────────────────────────────────────
        'vwap_pv':          deque(maxlen=20),
        'vwap_v':           deque(maxlen=20),
        'vwap':             0.0,
        'vwap_sd':          0.0,
        'vwap_sd2_lower':   0.0,
        'vwap_sd3_lower':   0.0,

        # ── OBI tracking ──────────────────────────────────────────────────
        'bid_size':         0.0,
        'ask_size':         0.0,
        'obi_history':      deque(maxlen=5),

        # ── CVD (cumulative volume delta) ─────────────────────────────────
        'cvd':              deque(maxlen=algo.lookback),

        # ── Kalman filter (repurposed as trend slope / anti-chase) ─────────
        'kalman_estimate':  0.0,
        'kalman_error_cov': 1.0,
        'kalman_history':   deque(maxlen=10),   # stores last 10 estimates for slope
        'ker':              deque(maxlen=algo.short_period),   # Kalman efficiency ratio

        # ── Arming state machine ───────────────────────────────────────────
        # States: DORMANT / ARMING / READY / TRIGGERED / COOLDOWN / INVALIDATED
        'arm_state':        'DORMANT',
        'arm_state_bars':   0,   # bars spent in current state
    }


# Volume threshold multiplier to classify a bar as an "ignition" event
_VOL_IGNITION_MULT = 3.0

# Minimum consecutive bars showing leadership signals before a symbol reaches READY
_ARM_ARMING_BARS   = 2


def update_symbol_data(algo, symbol, bar, quote_bar=None):
    crypto = algo.crypto_data[symbol]

    price  = float(bar.Close)
    high   = float(bar.High)
    low    = float(bar.Low)
    volume = float(bar.Volume)

    # ── Core data ─────────────────────────────────────────────────────────
    crypto['prices'].append(price)
    crypto['highs'].append(high)
    crypto['lows'].append(low)

    if crypto['last_price'] > 0:
        ret = (price - crypto['last_price']) / crypto['last_price']
        crypto['returns'].append(ret)
    crypto['last_price'] = price

    crypto['volume'].append(volume)
    crypto['dollar_volume'].append(price * volume)
    if len(crypto['volume']) >= algo.short_period:
        crypto['volume_ma'].append(np.mean(list(crypto['volume'])[-algo.short_period:]))

    # ── Indicators ────────────────────────────────────────────────────────
    crypto['ema_ultra_short'].Update(bar.EndTime, price)
    crypto['ema_short'].Update(bar.EndTime, price)
    crypto['ema_medium'].Update(bar.EndTime, price)
    crypto['ema_5'].Update(bar.EndTime, price)
    crypto['atr'].Update(bar)
    crypto['adx'].Update(bar)
    crypto['rsi'].Update(bar.EndTime, price)

    # ── VWAP ──────────────────────────────────────────────────────────────
    crypto['vwap_pv'].append(price * volume)
    crypto['vwap_v'].append(volume)
    total_v = sum(crypto['vwap_v'])
    if total_v > 0:
        crypto['vwap'] = sum(crypto['vwap_pv']) / total_v
    crypto['volume_long'].append(volume)

    if len(crypto['vwap_v']) >= 5 and crypto['vwap'] > 0:
        vwap_val = crypto['vwap']
        pv_list  = list(crypto['vwap_pv'])
        v_list   = list(crypto['vwap_v'])
        bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
        if len(bar_prices) >= 5:
            sd = float(np.std(bar_prices))
            crypto['vwap_sd']       = sd
            crypto['vwap_sd2_lower'] = vwap_val - 2.0 * sd
            crypto['vwap_sd3_lower'] = vwap_val - 3.0 * sd

    # ── Volatility ────────────────────────────────────────────────────────
    if len(crypto['returns']) >= 10:
        crypto['volatility'].append(np.std(list(crypto['returns'])[-10:]))

    # ── Relative strength vs BTC ──────────────────────────────────────────
    if len(crypto['returns']) >= algo.short_period and len(algo.btc_returns) >= algo.short_period:
        coin_ret = np.sum(list(crypto['returns'])[-algo.short_period:])
        btc_ret  = np.sum(list(algo.btc_returns)[-algo.short_period:])
        crypto['rs_vs_btc'].append(coin_ret - btc_ret)

    # Medium-term RS (20-bar cumulative): updated every bar
    if len(crypto['returns']) >= 20 and len(algo.btc_returns) >= 20:
        coin_ret_med = np.sum(list(crypto['returns'])[-20:])
        btc_ret_med  = np.sum(list(algo.btc_returns)[-20:])
        crypto['rs_vs_btc_medium'] = float(coin_ret_med - btc_ret_med)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    if len(crypto['prices']) >= algo.medium_period:
        prices_arr = np.array(list(crypto['prices'])[-algo.medium_period:])
        std  = np.std(prices_arr)
        mean = np.mean(prices_arr)
        if std > 0:
            crypto['zscore'].append((price - mean) / std)
            crypto['bb_upper'].append(mean + 2 * std)
            crypto['bb_lower'].append(mean - 2 * std)
            w = 4 * std / mean if mean > 0 else 0.0
            crypto['bb_width'].append(w)

            # ── Compression / expansion tracking (NEW) ─────────────────────
            bb_widths = list(crypto['bb_width'])
            if len(bb_widths) >= 15:
                w_arr = np.array(bb_widths[-30:] if len(bb_widths) >= 30 else bb_widths)
                w_med = float(np.median(w_arr))
                crypto['bb_width_median_30'] = w_med

                # A bar is "compressed" if its width < 85% of the rolling median
                threshold = w_med * 0.85
                if w < threshold:
                    crypto['compression_bars'] += 1
                    if w < crypto['compression_min_width']:
                        crypto['compression_min_width'] = w
                else:
                    # Reset compression tracking when width is no longer compressed
                    # but preserve min_width for the expansion detection in the same bar
                    # (so the evaluator can still see it)
                    if crypto['compression_bars'] < 2:
                        # Very short compression — treat as no meaningful squeeze
                        crypto['compression_min_width'] = 999.0
                    crypto['compression_bars'] = 0

    # ── Breakout freshness (NEW) ──────────────────────────────────────────
    # Update freshness counter: how many bars ago did price break the 20-bar high?
    highs_list = list(crypto['highs'])
    if len(highs_list) >= 21:
        # 20-bar high BEFORE this bar (exclude current)
        range_high_20 = float(np.max(highs_list[-21:-1]))

        if high > crypto['prev_range_high_20'] and crypto['prev_range_high_20'] > 0:
            # Price just broke a new 20-bar high — reset freshness and record the level
            crypto['breakout_freshness'] = 0
            crypto['breakout_level']     = crypto['prev_range_high_20']
        elif crypto['breakout_freshness'] < 999:
            # Increment age of last breakout
            crypto['breakout_freshness'] = min(crypto['breakout_freshness'] + 1, 999)

        crypto['prev_range_high_20'] = range_high_20

    # ── Volume ignition freshness ─────────────────────────────────────────
    vols = list(crypto['volume'])
    long_vols = list(crypto['volume_long'])
    if len(vols) >= 5:
        baseline = (float(np.mean(long_vols[-120:])) if len(long_vols) >= 120
                    else float(np.mean(vols[-20:])))
        if baseline > 0 and volume >= baseline * _VOL_IGNITION_MULT:
            crypto['bars_since_vol_ignition'] = 0
        elif crypto['bars_since_vol_ignition'] < 999:
            crypto['bars_since_vol_ignition'] = min(crypto['bars_since_vol_ignition'] + 1, 999)

    # ── VWAP reclaim freshness ────────────────────────────────────────────
    vwap_now = float(crypto.get('vwap', 0.0))
    if vwap_now > 0:
        above_now = price > vwap_now
        if above_now and not crypto.get('_prev_above_vwap', False):
            # Price just crossed VWAP from below → reclaim event
            crypto['bars_since_vwap_reclaim'] = 0
        elif crypto['bars_since_vwap_reclaim'] < 999:
            crypto['bars_since_vwap_reclaim'] = min(crypto['bars_since_vwap_reclaim'] + 1, 999)
        crypto['_prev_above_vwap'] = above_now

    # ── Order-flow persistence ────────────────────────────────────────────
    if len(vols) >= 5:
        baseline = (float(np.mean(long_vols[-120:])) if len(long_vols) >= 120
                    else float(np.mean(vols[-20:])))
        if baseline > 0 and volume > baseline * 1.5:
            crypto['vol_persistence'] = min(crypto['vol_persistence'] + 1, 20)
        else:
            crypto['vol_persistence'] = max(crypto['vol_persistence'] - 1, 0)

    # ── CVD (cumulative volume delta) ─────────────────────────────────────
    high_low = high - low
    if high_low > 0:
        bar_delta = volume * ((price - low) - (high - price)) / high_low
    else:
        bar_delta = 0.0
    prev_cvd = crypto['cvd'][-1] if len(crypto['cvd']) > 0 else 0.0
    crypto['cvd'].append(prev_cvd + bar_delta)

    # ── Kalman efficiency ratio ────────────────────────────────────────────
    if len(crypto['prices']) >= 15:
        price_change     = abs(crypto['prices'][-1] - crypto['prices'][-15])
        volatility_sum   = sum(abs(crypto['prices'][i] - crypto['prices'][i - 1])
                               for i in range(-14, 0))
        if volatility_sum > 0:
            crypto['ker'].append(price_change / volatility_sum)
        else:
            crypto['ker'].append(0.0)

    # ── Kalman filter (repurposed: slope + anti-chase) ────────────────────
    Q = 1e-5
    R = 0.01
    if crypto['kalman_estimate'] == 0.0:
        crypto['kalman_estimate'] = price
    estimate_pred    = crypto['kalman_estimate']
    error_cov_pred   = crypto['kalman_error_cov'] + Q
    kalman_gain      = error_cov_pred / (error_cov_pred + R)
    crypto['kalman_estimate']  = estimate_pred + kalman_gain * (price - estimate_pred)
    crypto['kalman_error_cov'] = (1 - kalman_gain) * error_cov_pred
    # Store history for slope calculation
    crypto['kalman_history'].append(crypto['kalman_estimate'])

    # ── Spread tracking ───────────────────────────────────────────────────
    sp = get_spread_pct(algo, symbol)
    if sp is not None:
        crypto['spreads'].append(sp)

    # ── Quote bar OBI ─────────────────────────────────────────────────────
    if quote_bar is not None:
        try:
            bid_sz = float(quote_bar.LastBidSize) if quote_bar.LastBidSize else 0.0
            ask_sz = float(quote_bar.LastAskSize) if quote_bar.LastAskSize else 0.0
            if bid_sz > 0 or ask_sz > 0:
                crypto['bid_size'] = bid_sz
                crypto['ask_size'] = ask_sz
                total = bid_sz + ask_sz
                if total > 0:
                    obi_instant = (bid_sz - ask_sz) / total
                    crypto['obi_history'].append(obi_instant)
        except Exception:
            pass


def update_market_context(algo):
    if len(algo.btc_prices) >= 48:
        btc_arr     = np.array(list(algo.btc_prices))
        current_btc = btc_arr[-1]
        btc_mom_12  = (np.mean(list(algo.btc_returns)[-12:])
                       if len(algo.btc_returns) >= 12 else 0.0)
        btc_sma     = np.mean(btc_arr[-48:])

        if current_btc > btc_sma * 1.02:
            new_regime = "bull"
        elif current_btc < btc_sma * 0.98:
            new_regime = "bear"
        else:
            new_regime = "sideways"

        # Momentum nudge within sideways
        if new_regime == "sideways" and len(algo.btc_returns) >= 12:
            if btc_mom_12 > 0.0001:
                new_regime = "bull"
            elif btc_mom_12 < -0.0001:
                new_regime = "bear"

        # Require 3 consecutive bars to confirm regime change (noise filter)
        if new_regime != algo.market_regime:
            algo._regime_hold_count += 1
            if algo._regime_hold_count >= 3:
                algo.market_regime   = new_regime
                algo._regime_hold_count = 0
        else:
            algo._regime_hold_count = 0

    # Volatility regime
    if len(algo.btc_volatility) >= 5:
        current_vol = algo.btc_volatility[-1]
        avg_vol     = np.mean(list(algo.btc_volatility))
        if current_vol > avg_vol * 1.5:
            algo.volatility_regime = "high"
        elif current_vol < avg_vol * 0.5:
            algo.volatility_regime = "low"
        else:
            algo.volatility_regime = "normal"

    # Market breadth
    uptrend_count = 0
    total_ready   = 0
    for crypto in algo.crypto_data.values():
        if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
            total_ready += 1
            if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                uptrend_count += 1
    if total_ready > 5:
        algo.market_breadth = uptrend_count / total_ready

    # Cross-sectional RS rank: percentile rank each symbol's short RS vs BTC
    # so scoring.py can reward the strongest relative leaders
    _update_rs_ranks(algo)


def _update_rs_ranks(algo):
    """
    Compute per-symbol cross-sectional RS rank (0.0–1.0) across the active
    universe.  Higher rank = stronger short-term relative strength vs BTC.
    Updated every bar via update_market_context.
    """
    symbols_with_rs = []
    for sym, crypto in algo.crypto_data.items():
        rs_hist = list(crypto.get('rs_vs_btc', []))
        if rs_hist:
            symbols_with_rs.append((sym, float(rs_hist[-1])))

    if len(symbols_with_rs) < 3:
        return

    # Sort ascending; rank is 0-based index / (n-1)
    symbols_with_rs.sort(key=lambda x: x[1])
    n = len(symbols_with_rs)
    for rank_idx, (sym, _rs) in enumerate(symbols_with_rs):
        algo.crypto_data[sym]['rs_rank'] = rank_idx / max(n - 1, 1)


def annualized_vol(algo, crypto):
    if crypto is None:
        return None
    if len(crypto.get('volatility', [])) == 0:
        return None
    return float(crypto['volatility'][-1]) * algo.sqrt_annualization


def compute_portfolio_risk_estimate(algo):
    total_value = algo.Portfolio.TotalPortfolioValue
    if total_value <= 0:
        return 0.0
    risk = 0.0
    for kvp in algo.Portfolio:
        symbol, holding = kvp.Key, kvp.Value
        if not is_invested_not_dust(algo, symbol):
            continue
        crypto = algo.crypto_data.get(symbol)
        asset_vol_ann = annualized_vol(algo, crypto)
        if asset_vol_ann is None:
            asset_vol_ann = algo.min_asset_vol_floor
        weight = abs(holding.HoldingsValue) / total_value
        risk  += weight * asset_vol_ann
    return risk


def universe_filter(algo, universe):
    selected = []
    for crypto in universe:
        ticker = crypto.Symbol.Value
        if ticker in SYMBOL_BLACKLIST or ticker in algo._session_blacklist:
            continue
        if not ticker.endswith("USD"):
            continue
        base = ticker[:-3]
        if base in KNOWN_FIAT_CURRENCIES:
            continue
        if crypto.VolumeInUsd is None or crypto.VolumeInUsd == 0:
            continue
        if crypto.VolumeInUsd >= algo.min_volume_usd:
            selected.append(crypto)
    selected.sort(key=lambda x: x.VolumeInUsd, reverse=True)
    return [c.Symbol for c in selected[:algo.max_universe_size]]


# ─────────────────────────────────────────────────────────────────────────────
# Alternative.me Fear & Greed Index custom data feed
# (merged from alt_data.py to keep file layout at 6 files)
# ─────────────────────────────────────────────────────────────────────────────

class FearGreedData(PythonData):
    """
    Custom data feed for the Alternative.me Fear & Greed Index.
    Daily updates, free API, no key required.
    Value: 0-100 integer (0=Extreme Fear, 100=Extreme Greed)
    """

    def GetSource(self, config, date, isLiveMode):
        # limit=0 gets all history (necessary for historical backtesting).
        # format=csv forces line-by-line data instead of multiline JSON.
        limit = "2" if isLiveMode else "0"
        url = f"https://api.alternative.me/fng/?limit={limit}&format=csv"

        return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        # Skip empty lines and the CSV header
        if not line or line.strip() == "" or "value" in line.lower() or "timestamp" in line.lower():
            return None

        try:
            parts = line.split(',')

            timestamp = None
            value = None

            # Simple heuristic: The FNG value is 0-100. The Unix Timestamp is > 1 billion.
            for p in parts:
                p = p.strip()
                if not p: continue
                try:
                    num = float(p)
                    if num > 1000000000:
                        timestamp = int(num)
                    elif 0 <= num <= 100 and value is None:
                        value = num
                except ValueError:
                    pass

            # If we couldn't parse the row successfully, skip it
            if timestamp is None or value is None:
                return None

            result = FearGreedData()
            result.Symbol = config.Symbol
            result.Time = datetime.utcfromtimestamp(timestamp)
            result.Value = value
            result.EndTime = result.Time + timedelta(days=1)

            return result

        except Exception:
            return None
