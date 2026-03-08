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
        # Use Rest instead of RemoteFile so the entire JSON response is
        # delivered to Reader() as a single string, avoiding the line-by-line
        # split that breaks JSON parsing in live mode.
        return SubscriptionDataSource(url, SubscriptionTransportMedium.Rest)

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
    Custom data feed for Whale Alert large transaction monitoring.
    Tracks large BTC transfers as a sentiment/flow indicator.
    Value: total USD volume of large transactions in the period.
    """

    def GetSource(self, config, date, isLiveMode):
        # Whale Alert free tier: last 1 hour of transactions > $500k
        # Note: requires API key for production use
        url = "https://api.whale-alert.io/v1/transactions?min_value=500000&api_key=demo"
        # Use Rest to receive the full JSON body as a single string
        return SubscriptionDataSource(url, SubscriptionTransportMedium.Rest)

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.strip() == "":
            return None
        try:
            obj = json.loads(line)
            transactions = obj.get("transactions", [])
            if not transactions:
                return None
            # Aggregate total USD volume of whale transactions
            total_usd = sum(float(tx.get("amount_usd", 0)) for tx in transactions)
            result = WhaleAlertData()
            result.Symbol = config.Symbol
            result.Time = datetime.utcnow()
            result.Value = total_usd
            result.EndTime = result.Time + timedelta(hours=1)
            return result
        except Exception:
            return None
