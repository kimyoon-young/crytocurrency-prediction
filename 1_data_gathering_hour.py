
#polonix data
#https://docs.poloniex.com/#returntradehistory-public
# 300, 900, 1800, 7200


#참고사이트
#https://nicholastsmith.wordpress.com/2017/11/13/cryptocurrency-price-prediction-using-deep-learning-in-tensorflow/
#https://medium.com/@huangkh19951228/predicting-cryptocurrency-price-with-tensorflow-and-keras-e1674b0dc58a


import json
import numpy as np
import os
import pandas as pd
import urllib.request
import time

# connect to poloniex's API
# 2015-02-29일부터 시작
start = 1424368800
end = 9999999999
# 6만개씩 더함
#interval = 6000000
df = pd.DataFrame([])


url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start={}&end={}&period=7200'.format(start,end)
#url.format(start, end)
# parse json returned from the API to Pandas DF

openUrl = urllib.request.urlopen(url)
r = openUrl.read()
print(r)
openUrl.close()
sep_df = pd.read_json(r.decode())
df = df.append(sep_df, ignore_index=True )


print(df.head())
print(df.tail())
df['date'] = df['date'].astype(np.int64) // 1000000000
print(df.head())

original_columns=[u'close', u'date', u'high', u'low', u'open']
new_columns = ['Close','Timestamp','High','Low','Open']
df = df.loc[:,original_columns]
df.columns = new_columns
print(df.tail())
df.to_csv('data/bitcoin2015to2019_2hours.csv',index=None)

# 일단은 이상태로 진행