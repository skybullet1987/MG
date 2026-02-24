class MicroScalpEngine:
    """Scoring engine for the Machine Gun strategy.

    Evaluates multiple micro-scalp signals and returns a composite score
    in the range [0.0, 1.0].  A score at or above ``entry_threshold``
    (default 0.40) triggers a trade in the main algorithm.
    """

    # --- Thresholds (aggressive Machine Gun settings) ---
    OBI_STRONG_THRESHOLD = 0.50
    OBI_PARTIAL_THRESHOLD = 0.25
    VOL_SURGE_STRONG = 3.0
    VOL_SURGE_PARTIAL = 2.0
    ADX_STRONG_THRESHOLD = 20
    ADX_MODERATE_THRESHOLD = 15
    VWAP_BUFFER = 1.0005
    RSI_OVERSOLD_THRESHOLD = 45
    RSI_MILDLY_OVERSOLD_THRESHOLD = 50
    BB_NEAR_LOWER_PCT = 0.03

    def __init__(self, algo):
        self.algo = algo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, symbol):
        """Return a composite conviction score for *symbol* in [0.0, 1.0]."""
        try:
            security = self.algo.Securities[symbol]
        except KeyError:
            return 0.0

        price = security.Price
        if price <= 0:
            return 0.0

        total_score = 0.0
        max_score = 0.0

        # Order Book Imbalance (OBI) component
        obi_score, obi_max = self._score_obi(security)
        total_score += obi_score
        max_score += obi_max

        # Volume surge component
        vol_score, vol_max = self._score_volume(symbol)
        total_score += vol_score
        max_score += vol_max

        # ADX trend-strength component
        adx_score, adx_max = self._score_adx(symbol)
        total_score += adx_score
        max_score += adx_max

        # VWAP component
        vwap_score, vwap_max = self._score_vwap(symbol, price)
        total_score += vwap_score
        max_score += vwap_max

        # RSI component
        rsi_score, rsi_max = self._score_rsi(symbol)
        total_score += rsi_score
        max_score += rsi_max

        # Bollinger Band component
        bb_score, bb_max = self._score_bb(symbol, price)
        total_score += bb_score
        max_score += bb_max

        if max_score <= 0:
            return 0.0

        return min(total_score / max_score, 1.0)

    # ------------------------------------------------------------------
    # Individual signal scorers
    # ------------------------------------------------------------------

    def _score_obi(self, security):
        """Order Book Imbalance: uses bid/ask sizes when available."""
        max_score = 2.0
        try:
            bid_size = security.BidSize
            ask_size = security.AskSize
            total = bid_size + ask_size
            if total <= 0:
                return 0.0, max_score
            obi = (bid_size - ask_size) / total
        except Exception:
            return 0.0, max_score

        if obi >= self.OBI_STRONG_THRESHOLD:
            return max_score, max_score
        if obi >= self.OBI_PARTIAL_THRESHOLD:
            return 1.0, max_score
        return 0.0, max_score

    def _score_volume(self, symbol):
        """Volume surge relative to recent average."""
        max_score = 2.0
        try:
            history = self.algo.History(symbol, 20, Resolution.Minute)
            if history.empty:
                return 0.0, max_score
            volumes = history["volume"].values
            if len(volumes) < 2:
                return 0.0, max_score
            avg_vol = volumes[:-1].mean()
            current_vol = volumes[-1]
            if avg_vol <= 0:
                return 0.0, max_score
            surge = current_vol / avg_vol
        except Exception:
            return 0.0, max_score

        if surge >= self.VOL_SURGE_STRONG:
            return max_score, max_score
        if surge >= self.VOL_SURGE_PARTIAL:
            return 1.0, max_score
        return 0.0, max_score

    def _score_adx(self, symbol):
        """ADX trend strength component."""
        max_score = 1.5
        try:
            adx = self.algo.ADX(symbol, 14, Resolution.Minute)
            if not adx.IsReady:
                return 0.0, max_score
            adx_val = adx.Current.Value
        except Exception:
            return 0.0, max_score

        if adx_val >= self.ADX_STRONG_THRESHOLD:
            return max_score, max_score
        if adx_val >= self.ADX_MODERATE_THRESHOLD:
            return 0.75, max_score
        return 0.0, max_score

    def _score_vwap(self, symbol, price):
        """Price vs. VWAP component."""
        max_score = 1.5
        try:
            vwap = self.algo.VWAP(symbol, Resolution.Minute)
            if not vwap.IsReady:
                return 0.0, max_score
            vwap_val = vwap.Current.Value
            if vwap_val <= 0:
                return 0.0, max_score
        except Exception:
            return 0.0, max_score

        if price >= vwap_val * self.VWAP_BUFFER:
            return max_score, max_score
        if price >= vwap_val:
            return 0.75, max_score
        return 0.0, max_score

    def _score_rsi(self, symbol):
        """RSI oversold component."""
        max_score = 1.5
        try:
            rsi = self.algo.RSI(symbol, 14, MovingAverageType.Simple, Resolution.Minute)
            if not rsi.IsReady:
                return 0.0, max_score
            rsi_val = rsi.Current.Value
        except Exception:
            return 0.0, max_score

        if rsi_val <= self.RSI_OVERSOLD_THRESHOLD:
            return max_score, max_score
        if rsi_val <= self.RSI_MILDLY_OVERSOLD_THRESHOLD:
            return 0.75, max_score
        return 0.0, max_score

    def _score_bb(self, symbol, price):
        """Bollinger Band lower-band proximity component."""
        max_score = 1.5
        try:
            bb = self.algo.BB(symbol, 20, 2.0, MovingAverageType.Simple, Resolution.Minute)
            if not bb.IsReady:
                return 0.0, max_score
            lower_band = bb.LowerBand.Current.Value
            if lower_band <= 0:
                return 0.0, max_score
        except Exception:
            return 0.0, max_score

        if price <= lower_band * (1.0 + self.BB_NEAR_LOWER_PCT):
            return max_score, max_score
        return 0.0, max_score
