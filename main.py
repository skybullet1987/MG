from QuantConnect.Orders.Fees import FeeModel, OrderFee
from QuantConnect.Securities import CashAmount
from AlgorithmImports import *


class MakerTakerFeeModel(FeeModel):
    """Custom Fee Model: 0.25% Maker (Limit), 0.40% Taker (Market)"""
    def GetOrderFee(self, parameters):
        order = parameters.Order
        fee_pct = 0.0025 if order.Type == OrderType.Limit else 0.0040
        trade_value = order.AbsoluteQuantity * parameters.Security.Price
        return OrderFee(CashAmount(trade_value * fee_pct, "USD"))


class RealisticCryptoSlippage(SlippageModel):
    """Slippage model for crypto: 0.05% per trade."""
    def GetSlippageApproximation(self, asset, order):
        return asset.Price * 0.0005


class SimplifiedCryptoStrategy(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100)

        # Machine Gun strategy parameters
        self.entry_threshold = 0.40
        self.high_conviction_threshold = 0.60
        self.quick_take_profit = 0.080
        self.tight_stop_loss = 0.025
        self.atr_tp_mult = 4.0
        self.atr_sl_mult = 2.0
        self.trail_activation = 0.010
        self.trail_stop_pct = 0.005
        self.time_stop_hours = 2.0
        self.time_stop_pnl_min = 0.003
        self.extended_time_stop_hours = 4.0
        self.extended_time_stop_pnl_max = 0.015
        self.stale_position_hours = 6.0
        self.position_size_pct = 0.15
        self.base_max_positions = 6
        self.max_positions = 6
        self.min_notional = 5.5
        self.max_universe_size = 60

        self.SetSecurityInitializer(self.CustomSecurityInitializer)

        self.UniverseSettings.Resolution = Resolution.Minute
        self.AddUniverse(self.CoarseSelectionFunction)

        self.symbols = {}
        self.open_positions = {}

        from scoring import MicroScalpEngine
        from execution import ExecutionEngine
        self.scorer = MicroScalpEngine(self)
        self.executor = ExecutionEngine(self)

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(minutes=5)),
            self.ScanForTrades,
        )
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(timedelta(minutes=1)),
            self.ManagePositions,
        )

    def CustomSecurityInitializer(self, security):
        security.SetSlippageModel(RealisticCryptoSlippage())
        security.SetFeeModel(MakerTakerFeeModel())

    def CoarseSelectionFunction(self, coarse):
        selected = [
            x.Symbol for x in coarse
            if x.Price > 0
        ]
        return selected[: self.max_universe_size]

    def ScanForTrades(self):
        if len(self.open_positions) >= self.max_positions:
            return

        portfolio_value = self.Portfolio.TotalPortfolioValue
        target_value = portfolio_value * self.position_size_pct
        trade_size_usd = max(target_value, self.min_notional)

        for symbol in list(self.symbols.keys()):
            if symbol in self.open_positions:
                continue
            if len(self.open_positions) >= self.max_positions:
                break

            score = self.scorer.score(symbol)
            if score >= self.entry_threshold:
                security = self.Securities[symbol]
                price = security.Price
                if price <= 0:
                    continue
                quantity = trade_size_usd / price
                self.executor.execute_buy(symbol, quantity)
                self.open_positions[symbol] = {
                    "entry_price": price,
                    "entry_time": self.Time,
                    "quantity": quantity,
                    "high_water_mark": price,
                    "trailing_active": False,
                }

    def ManagePositions(self):
        to_close = []
        for symbol, pos in list(self.open_positions.items()):
            if symbol not in self.Securities:
                continue
            security = self.Securities[symbol]
            price = security.Price
            if price <= 0:
                continue

            entry_price = pos["entry_price"]
            entry_time = pos["entry_time"]
            pnl_pct = (price - entry_price) / entry_price
            elapsed_hours = (self.Time - entry_time).total_seconds() / 3600.0

            # Update high water mark and trailing stop
            if price > pos["high_water_mark"]:
                pos["high_water_mark"] = price

            if pnl_pct >= self.trail_activation:
                pos["trailing_active"] = True

            if pos["trailing_active"]:
                trail_stop = pos["high_water_mark"] * (1.0 - self.trail_stop_pct)
                if price <= trail_stop:
                    to_close.append((symbol, "trailing_stop"))
                    continue

            # Quick take-profit
            if pnl_pct >= self.quick_take_profit:
                to_close.append((symbol, "take_profit"))
                continue

            # Tight stop-loss
            if pnl_pct <= -self.tight_stop_loss:
                to_close.append((symbol, "stop_loss"))
                continue

            # Time stop
            if elapsed_hours >= self.time_stop_hours:
                if pnl_pct < self.time_stop_pnl_min:
                    to_close.append((symbol, "time_stop"))
                    continue

            # Extended time stop
            if elapsed_hours >= self.extended_time_stop_hours:
                if pnl_pct < self.extended_time_stop_pnl_max:
                    to_close.append((symbol, "extended_time_stop"))
                    continue

            # Stale position stop
            if elapsed_hours >= self.stale_position_hours:
                to_close.append((symbol, "stale_position"))
                continue

        for symbol, reason in to_close:
            if symbol in self.open_positions:
                qty = self.open_positions[symbol]["quantity"]
                self.executor.execute_sell(symbol, qty)
                del self.open_positions[symbol]
                self.Log(f"Closed {symbol} reason={reason}")

    def OnSecuritiesChanged(self, changes):
        for security in changes.AddedSecurities:
            self.symbols[security.Symbol] = security
        for security in changes.RemovedSecurities:
            sym = security.Symbol
            self.symbols.pop(sym, None)
            self.open_positions.pop(sym, None)
