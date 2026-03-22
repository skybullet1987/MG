# region imports
from AlgorithmImports import *
import json
from datetime import datetime, timedelta
# endregion

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
