"""
Machine Gun 3 — Data-layer helper functions
============================================
Plain functions that process per-bar market data and update the market
context (regime, breadth, volatility) on behalf of the algorithm.

All functions receive the algorithm instance as the first argument (``algo``)
so they can read and write algorithm state without inheriting from QCAlgorithm.
"""
import numpy as np
from execution import get_spread_pct


def update_symbol_data(algo, symbol, bar, quote_bar=None):
    """Update per-symbol indicator state from one OHLCV bar."""
    crypto = algo.crypto_data[symbol]
    price  = float(bar.Close)
    high   = float(bar.High)
    low    = float(bar.Low)
    volume = float(bar.Volume)

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

    crypto['ema_ultra_short'].Update(bar.EndTime, price)
    crypto['ema_short'].Update(bar.EndTime, price)
    crypto['ema_medium'].Update(bar.EndTime, price)
    crypto['ema_5'].Update(bar.EndTime, price)
    crypto['atr'].Update(bar)
    crypto['adx'].Update(bar)

    # Rolling 20-bar VWAP
    crypto['vwap_pv'].append(price * volume)
    crypto['vwap_v'].append(volume)
    total_v = sum(crypto['vwap_v'])
    if total_v > 0:
        crypto['vwap'] = sum(crypto['vwap_pv']) / total_v

    # Long-term volume baseline for adaptive scoring thresholds (~24h window)
    crypto['volume_long'].append(volume)

    # VWAP SD bands: compute std of bar prices within the rolling VWAP window
    if len(crypto['vwap_v']) >= 5 and crypto['vwap'] > 0:
        vwap_val   = crypto['vwap']
        pv_list    = list(crypto['vwap_pv'])
        v_list     = list(crypto['vwap_v'])
        bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
        if len(bar_prices) >= 5:
            sd = float(np.std(bar_prices))
            crypto['vwap_sd']        = sd
            crypto['vwap_sd2_lower'] = vwap_val - 2.0 * sd
            crypto['vwap_sd3_lower'] = vwap_val - 3.0 * sd

    if len(crypto['returns']) >= 10:
        crypto['volatility'].append(np.std(list(crypto['returns'])[-10:]))

    crypto['rsi'].Update(bar.EndTime, price)

    if (len(crypto['returns']) >= algo.short_period
            and len(algo.btc_returns) >= algo.short_period):
        coin_ret = np.sum(list(crypto['returns'])[-algo.short_period:])
        btc_ret  = np.sum(list(algo.btc_returns)[-algo.short_period:])
        crypto['rs_vs_btc'].append(coin_ret - btc_ret)

    if len(crypto['prices']) >= algo.medium_period:
        prices_arr = np.array(list(crypto['prices'])[-algo.medium_period:])
        std  = np.std(prices_arr)
        mean = np.mean(prices_arr)
        if std > 0:
            crypto['zscore'].append((price - mean) / std)
            crypto['bb_upper'].append(mean + 2 * std)
            crypto['bb_lower'].append(mean - 2 * std)
            crypto['bb_width'].append(4 * std / mean if mean > 0 else 0)

    # CVD: Tick Delta approximation
    high_low = high - low
    if high_low > 0:
        bar_delta = volume * ((price - low) - (high - price)) / high_low
    else:
        bar_delta = 0.0
    prev_cvd = crypto['cvd'][-1] if len(crypto['cvd']) > 0 else 0.0
    crypto['cvd'].append(prev_cvd + bar_delta)

    # KER: Kaufman Efficiency Ratio (15-period)
    if len(crypto['prices']) >= 15:
        price_change   = abs(crypto['prices'][-1] - crypto['prices'][-15])
        volatility_sum = sum(
            abs(crypto['prices'][i] - crypto['prices'][i - 1])
            for i in range(-14, 0)
        )
        crypto['ker'].append(price_change / volatility_sum if volatility_sum > 0 else 0.0)

    # 1-D Kalman Filter for price
    Q = 1e-5   # process noise variance
    R = 0.01   # measurement noise variance
    if crypto['kalman_estimate'] == 0.0:
        crypto['kalman_estimate'] = price
    estimate_pred  = crypto['kalman_estimate']
    error_cov_pred = crypto['kalman_error_cov'] + Q
    kalman_gain    = error_cov_pred / (error_cov_pred + R)
    crypto['kalman_estimate']  = estimate_pred + kalman_gain * (price - estimate_pred)
    crypto['kalman_error_cov'] = (1 - kalman_gain) * error_cov_pred

    sp = get_spread_pct(algo, symbol)
    if sp is not None:
        crypto['spreads'].append(sp)

    # Update bid/ask sizes from QuoteBar for Order Book Imbalance calculation
    if quote_bar is not None:
        try:
            bid_sz = float(quote_bar.LastBidSize) if quote_bar.LastBidSize else 0.0
            ask_sz = float(quote_bar.LastAskSize) if quote_bar.LastAskSize else 0.0
            if bid_sz > 0 or ask_sz > 0:
                crypto['bid_size'] = bid_sz
                crypto['ask_size'] = ask_sz
        except Exception:
            pass


def update_market_context(algo):
    """Update market regime, volatility regime, and market breadth from BTC data."""
    if len(algo.btc_prices) >= 48:
        btc_arr      = np.array(list(algo.btc_prices))
        current_btc  = btc_arr[-1]
        btc_mom_12   = (
            np.mean(list(algo.btc_returns)[-12:])
            if len(algo.btc_returns) >= 12 else 0.0
        )
        btc_sma = np.mean(btc_arr[-48:])

        if current_btc > btc_sma * 1.02:
            new_regime = "bull"
        elif current_btc < btc_sma * 0.98:
            new_regime = "bear"
        else:
            new_regime = "sideways"

        # Momentum confirmation — more sensitive than SMA alone
        if new_regime == "sideways" and len(algo.btc_returns) >= 12:
            if btc_mom_12 > 0.0001:
                new_regime = "bull"
            elif btc_mom_12 < -0.0001:
                new_regime = "bear"

        # Hysteresis: only commit to a regime change after 3 consecutive bars
        if new_regime != algo.market_regime:
            algo._regime_hold_count += 1
            if algo._regime_hold_count >= 3:
                algo.market_regime     = new_regime
                algo._regime_hold_count = 0
        else:
            algo._regime_hold_count = 0

    if len(algo.btc_volatility) >= 5:
        current_vol = algo.btc_volatility[-1]
        avg_vol     = np.mean(list(algo.btc_volatility))
        if current_vol > avg_vol * 1.5:
            algo.volatility_regime = "high"
        elif current_vol < avg_vol * 0.5:
            algo.volatility_regime = "low"
        else:
            algo.volatility_regime = "normal"

    uptrend_count = 0
    total_ready   = 0
    for crypto in algo.crypto_data.values():
        if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
            total_ready += 1
            if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                uptrend_count += 1
    if total_ready > 5:
        algo.market_breadth = uptrend_count / total_ready
