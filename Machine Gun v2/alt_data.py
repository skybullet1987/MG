# region imports
from AlgorithmImports import *
import json
# endregion


class FearGreedData(PythonData):
    """
    Custom data feed for the Alternative.me Fear & Greed Index.
    Daily updates, free API, no key required.
    Value: 0-100 integer (0=Extreme Fear, 100=Extreme Greed)
    """

    def GetSource(self, config, date, isLiveMode):
        url = "https://api.alternative.me/fng/?limit=1&format=json"
        return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.strip() == "":
            return None
        try:
            obj = json.loads(line)
            data_list = obj.get("data", [])
            if not data_list:
                return None
            entry = data_list[0]
            value = float(entry["value"])
            timestamp = int(entry["timestamp"])
            result = FearGreedData()
            result.Symbol = config.Symbol
            result.Time = datetime.utcfromtimestamp(timestamp)
            result.Value = value
            result.EndTime = result.Time + timedelta(days=1)
            return result
        except Exception:
            return None


class WhaleAlertData(PythonData):
    """
    Custom data feed for Whale Alert exchange flow data.
    Tracks large on-chain transactions to/from exchanges.
    Requires whale_alert_api_key parameter; degrades gracefully if absent.
    """

    def GetSource(self, config, date, isLiveMode):
        try:
            api_key = config.Symbol.Value.replace("WHALE", "").strip() or ""
        except Exception:
            api_key = ""
        start = int((date - timedelta(hours=1)).timestamp())
        url = (
            f"https://api.whale-alert.io/v1/transactions"
            f"?api_key={api_key}&min_value=500000&start={start}"
        )
        return SubscriptionDataSource(url, SubscriptionTransportMedium.RemoteFile)

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.strip() == "":
            return None
        try:
            obj = json.loads(line)
            transactions = obj.get("transactions", [])
            exchange_inflow = sum(
                1 for tx in transactions
                if tx.get("to", {}).get("owner_type") == "exchange"
            )
            exchange_outflow = sum(
                1 for tx in transactions
                if tx.get("from", {}).get("owner_type") == "exchange"
            )
            result = WhaleAlertData()
            result.Symbol = config.Symbol
            result.Time = date
            result.Value = float(len(transactions))
            result.net_exchange_flow = float(exchange_inflow - exchange_outflow)
            result.EndTime = result.Time + timedelta(hours=1)
            return result
        except Exception:
            return None
