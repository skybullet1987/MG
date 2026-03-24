# region imports
from AlgorithmImports import *
from execution import *
import numpy as np
from collections import deque
# endregion


def is_ready(c):
    return len(c['prices']) >= 10 and c['rsi'].IsReady


def initialize_symbol(algo, symbol):
    algo.crypto_data[symbol] = {
        'prices': deque(maxlen=algo.lookback),
        'returns': deque(maxlen=algo.lookback),
        'volume': deque(maxlen=algo.lookback),
        'volume_ma': deque(maxlen=algo.medium_period),
        'dollar_volume': deque(maxlen=algo.lookback),
        'ema_ultra_short': ExponentialMovingAverage(algo.ultra_short_period),
        'ema_short': ExponentialMovingAverage(algo.short_period),
        'ema_medium': ExponentialMovingAverage(algo.medium_period),
        'ema_5': ExponentialMovingAverage(5),
        'atr': AverageTrueRange(14),
        'adx': AverageDirectionalIndex(algo.adx_min_period),
        'volatility': deque(maxlen=algo.medium_period),
        'rsi': RelativeStrengthIndex(7),
        'rs_vs_btc': deque(maxlen=algo.medium_period),
        'zscore': deque(maxlen=algo.short_period),
        'last_price': 0,
        'recent_net_scores': deque(maxlen=3),
        'spreads': deque(maxlen=algo.spread_median_window),
        'trail_stop': None,
        'highs': deque(maxlen=algo.lookback),
        'lows': deque(maxlen=algo.lookback),
        'bb_upper': deque(maxlen=algo.short_period),
        'bb_lower': deque(maxlen=algo.short_period),
        'bb_width': deque(maxlen=algo.medium_period),
        'trade_count_today': 0,
        'last_loss_time': None,
        'bid_size': 0.0,
        'ask_size': 0.0,
        'obi_history': deque(maxlen=5),
        'vwap_pv': deque(maxlen=20),
        'vwap_v': deque(maxlen=20),
        'vwap': 0.0,
        'volume_long': deque(maxlen=1440),
        'vwap_sd': 0.0,
        'vwap_sd2_lower': 0.0,
        'vwap_sd3_lower': 0.0,
        'cvd': deque(maxlen=algo.lookback),
        'ker': deque(maxlen=algo.short_period),
        'kalman_estimate': 0.0,
        'kalman_error_cov': 1.0,
    }


def update_symbol_data(algo, symbol, bar, quote_bar=None):
    crypto = algo.crypto_data[symbol]
    price = float(bar.Close)
    high = float(bar.High)
    low = float(bar.Low)
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
    crypto['vwap_pv'].append(price * volume)
    crypto['vwap_v'].append(volume)
    total_v = sum(crypto['vwap_v'])
    if total_v > 0:
        crypto['vwap'] = sum(crypto['vwap_pv']) / total_v
    crypto['volume_long'].append(volume)
    if len(crypto['vwap_v']) >= 5 and crypto['vwap'] > 0:
        vwap_val = crypto['vwap']
        pv_list = list(crypto['vwap_pv'])
        v_list = list(crypto['vwap_v'])
        bar_prices = [pv / v for pv, v in zip(pv_list, v_list) if v > 0]
        if len(bar_prices) >= 5:
            sd = float(np.std(bar_prices))
            crypto['vwap_sd'] = sd
            crypto['vwap_sd2_lower'] = vwap_val - 2.0 * sd
            crypto['vwap_sd3_lower'] = vwap_val - 3.0 * sd
    if len(crypto['returns']) >= 10:
        crypto['volatility'].append(np.std(list(crypto['returns'])[-10:]))
    crypto['rsi'].Update(bar.EndTime, price)
    if len(crypto['returns']) >= algo.short_period and len(algo.btc_returns) >= algo.short_period:
        coin_ret = np.sum(list(crypto['returns'])[-algo.short_period:])
        btc_ret = np.sum(list(algo.btc_returns)[-algo.short_period:])
        crypto['rs_vs_btc'].append(coin_ret - btc_ret)
    if len(crypto['prices']) >= algo.medium_period:
        prices_arr = np.array(list(crypto['prices'])[-algo.medium_period:])
        std = np.std(prices_arr)
        mean = np.mean(prices_arr)
        if std > 0:
            crypto['zscore'].append((price - mean) / std)
            crypto['bb_upper'].append(mean + 2 * std)
            crypto['bb_lower'].append(mean - 2 * std)
            crypto['bb_width'].append(4 * std / mean if mean > 0 else 0)
    high_low = high - low
    if high_low > 0:
        bar_delta = volume * ((price - low) - (high - price)) / high_low
    else:
        bar_delta = 0.0
    prev_cvd = crypto['cvd'][-1] if len(crypto['cvd']) > 0 else 0.0
    crypto['cvd'].append(prev_cvd + bar_delta)
    if len(crypto['prices']) >= 15:
        price_change = abs(crypto['prices'][-1] - crypto['prices'][-15])
        volatility_sum = sum(abs(crypto['prices'][i] - crypto['prices'][i-1]) for i in range(-14, 0))
        if volatility_sum > 0:
            crypto['ker'].append(price_change / volatility_sum)
        else:
            crypto['ker'].append(0.0)
    Q = 1e-5
    R = 0.01
    if crypto['kalman_estimate'] == 0.0:
        crypto['kalman_estimate'] = price
    estimate_pred = crypto['kalman_estimate']
    error_cov_pred = crypto['kalman_error_cov'] + Q
    kalman_gain = error_cov_pred / (error_cov_pred + R)
    crypto['kalman_estimate'] = estimate_pred + kalman_gain * (price - estimate_pred)
    crypto['kalman_error_cov'] = (1 - kalman_gain) * error_cov_pred
    sp = get_spread_pct(algo, symbol)
    if sp is not None:
        crypto['spreads'].append(sp)
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
        btc_arr = np.array(list(algo.btc_prices))
        current_btc = btc_arr[-1]
        btc_mom_12 = np.mean(list(algo.btc_returns)[-12:]) if len(algo.btc_returns) >= 12 else 0.0
        btc_sma = np.mean(btc_arr[-48:])
        if current_btc > btc_sma * 1.02:
            new_regime = "bull"
        elif current_btc < btc_sma * 0.98:
            new_regime = "bear"
        else:
            new_regime = "sideways"
        if new_regime == "sideways" and len(algo.btc_returns) >= 12:
            if btc_mom_12 > 0.0001:
                new_regime = "bull"
            elif btc_mom_12 < -0.0001:
                new_regime = "bear"
        if new_regime != algo.market_regime:
            algo._regime_hold_count += 1
            if algo._regime_hold_count >= 3:
                algo.market_regime = new_regime
                algo._regime_hold_count = 0
        else:
            algo._regime_hold_count = 0
    if len(algo.btc_volatility) >= 5:
        current_vol = algo.btc_volatility[-1]
        avg_vol = np.mean(list(algo.btc_volatility))
        if current_vol > avg_vol * 1.5:
            algo.volatility_regime = "high"
        elif current_vol < avg_vol * 0.5:
            algo.volatility_regime = "low"
        else:
            algo.volatility_regime = "normal"
    uptrend_count = 0
    total_ready = 0
    for crypto in algo.crypto_data.values():
        if crypto['ema_short'].IsReady and crypto['ema_medium'].IsReady:
            total_ready += 1
            if crypto['ema_short'].Current.Value > crypto['ema_medium'].Current.Value:
                uptrend_count += 1
    if total_ready > 5:
        algo.market_breadth = uptrend_count / total_ready


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
        risk += weight * asset_vol_ann
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
