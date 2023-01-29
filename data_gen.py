import bitmex
import pandas as pd
import matplotlib.pyplot as plt

bitmex_api_key = "" 
bitmex_api_secret = ""
client = bitmex.bitmex(test=True, api_key=bitmex_api_key, api_secret=bitmex_api_secret)

timeBucket = "1d"
start = 0
data = []
while client.Trade.Trade_getBucketed(binSize=timeBucket, count=1000, symbol='XBTUSD', start=start).result()[0] != []:
    data += client.Trade.Trade_getBucketed(binSize=timeBucket, count=1000, symbol='XBTUSD', start=start).result()[0]
    start += 1000
result = pd.DataFrame({k: [v[k] for v in data] for k in data[0].keys()})
result.to_csv("data.csv", index=False)