import bitmex
import pandas as pd
import time
import json
import requests

bitmex_api_key = "YrCRra1dCrI0M383_JPc11qy" 
bitmex_api_secret = ""
client = bitmex.bitmex(test=True, api_key=bitmex_api_key, api_secret=bitmex_api_secret)
# client = bitmex.bitmex()

# open = client.Position.Position_get(filter=json.dumps({'symbol': 'XBTUSD'})).result()[0]

timeBucket = "1h"
data = client.Trade.Trade_getBucketed(binSize=timeBucket, count=1000, symbol='XBTUSD', reverse=True).result()[0]
result = pd.DataFrame({k: [v[k] for v in data] for k in data[0].keys()})
# print(result)
result.to_csv("data.csv", index=False)