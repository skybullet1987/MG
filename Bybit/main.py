    def _calculate_factor_scores(self, symbol, crypto):
        """Evaluate both signals and FADE them (reverse directions)."""
        long_score, long_components = self._scoring_engine.calculate_scalp_score(crypto)
        short_score, short_components = self._scoring_engine.calculate_short_score(crypto)

        # FADE LOGIC: If system sees a bullish signal, we SHORT it.
        if long_score >= short_score and long_score > 0:
            components = long_components.copy()
            components['_scalp_score'] = long_score
            components['_direction'] = -1  # FADE: bullish signal -> enter SHORT
            components['_long_score'] = 0.0
            components['_short_score'] = long_score

        # FADE LOGIC: If system sees a bearish signal, we LONG it.
        elif short_score > long_score and short_score > 0:
            components = short_components.copy()
            components['_scalp_score'] = short_score
            components['_direction'] = 1   # FADE: bearish signal -> enter LONG
            components['_long_score'] = short_score
            components['_short_score'] = 0.0

        else:
            components = long_components.copy()
            components['_scalp_score'] = 0.0
            components['_direction'] = 1
            components['_long_score'] = 0.0
            components['_short_score'] = 0.0

        return components
