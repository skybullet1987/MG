class ExecutionEngine:
    """Handles order execution for the Machine Gun strategy.

    Uses Limit Orders (Maker) for both buys and sells to take advantage
    of the lower 0.25% Maker fee vs. 0.40% Taker fee.
    """

    def __init__(self, algo):
        self.algo = algo

    def execute_buy(self, symbol, quantity):
        """Submit a Maker (Limit) buy order at the current best bid."""
        security = self.algo.Securities[symbol]
        limit_price = security.BidPrice if security.BidPrice > 0 else security.Price * 0.9995
        self.algo.LimitOrder(symbol, quantity, limit_price, "MG Maker Buy")

    def execute_sell(self, symbol, quantity):
        """Submit a Maker (Limit) sell order at the current best ask."""
        security = self.algo.Securities[symbol]
        limit_price = security.AskPrice if security.AskPrice > 0 else security.Price * 1.0005
        self.algo.LimitOrder(symbol, -quantity, limit_price, "MG Maker Sell")
