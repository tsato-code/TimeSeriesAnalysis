from fbprophet import Prophet
import pandas as pd


df = pd.read_csv("data/input/example_wp_log_R.csv")
df["cap"] = 8.5
print(df.head())

# 飽和上限
m = Prophet(growth="logistic")
m.fit(df)

future = m.make_future_dataframe(periods=1826)
future["cap"] = 8.5

fcst = m.predict(future)
fig = m.plot(fcst)
fig.savefig("figure/14_prophet_saturating.png")

# 飽和下限
df["y"] = 10 - df["y"]
df["cap"] = 6
df["floor"] = 1.5

future["cap"] = 6
future["floor"] = 1.5
m = Prophet(growth="logistic")
m.fit(df)
fcst = m.predict(future)
fig = m.plot(fcst)
fig.savefig("figure/15_prophet_saturating_lower.png")

