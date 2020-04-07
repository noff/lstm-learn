import csv
import requests
url = "https://api.hitbtc.com/api/2/public/ticker/btcusd"

f = open("data/btc.csv", "w")
with f:
    writer = csv.writer(f)
    writer.writerow(['ask', 'bid', 'last', 'open', 'low', 'high', 'volume', 'volumeQuote', 'timestamp'])
    for i in range(10000):
        r = requests.get(url)
        row = r.json()
        writer.writerow([row['ask'], row['bid'], row['last'], row['open'], row['low'], row['high'], row['volume'], row['volumeQuote'], row['timestamp']])
