from fbprophet import Prophet
import pandas as pd


df = pd.read_csv("data/input/example_wp_log_R.csv")
print(df.head())

m = Prophet()
m.fit(df)

# 単純予測
future = m.make_future_dataframe(periods=365)
print(future.tail())

forecast = m.predict(future)
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

# 予測値プロット
fig1 = m.plot(forecast)
fig1.savefig("figure/12_prophet_forecast.png")

# 成分プロット
fig2 = m.plot_components(forecast)
fig2.savefig("figure/13_prophet_componets.png")



