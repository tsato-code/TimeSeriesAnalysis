import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import statsmodels.api as sm

df = pd.read_csv("data/input/AirPassengers.csv", parse_dates=["Month"], index_col="Month")
print("read csv ...")
print(f"{df.head()}")

plt.plot(df["#Passengers"])
plt.ylabel("#Passengers")
plt.savefig("figure/01_air_passenger.png")
plt.clf()

train = df[:"1958-12-31"]
train = train.reset_index().rename(columns={"Month": "ds", "#Passengers": "y"})

# モデル作成
m = Prophet(
    growth="linear",
    yearly_seasonality=10,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative").fit(train)

# 予測期間
future = m.make_future_dataframe(periods=24, freq="MS")
forecast = m.predict(df=future)
fig = m.plot(forecast)
fig.savefig("figure/02_prophet_forecast.png")
fig.clf()

# トレンドと年周期
fig = m.plot_components(forecast)
fig.savefig("figure/03_prophet_components.png")
fig.clf()

# 実測値と予測値、信頼区間
plt.scatter(df.index, df["#Passengers"], color="black", s=10, label="#Passengers")
plt.plot(forecast["ds"], forecast["yhat"], label="yhat")
plt.fill_between(
    df.index,
    forecast["yhat_upper"],
    forecast["yhat_lower"],
    color="blue",
    alpha=.1,
    label="confidence interval")
plt.xlabel("Date")
plt.ylabel("#Passengers")
plt.legend()
plt.savefig("figure/04_prophet_predict.png")
plt.clf()

# 予測値と実測値の誤差
observed = np.array(df.tail(24)["#Passengers"])
pred = np.array(forecast.tail(24)["yhat"])
resid = observed - pred
print(f"PREDICT: {pred}")

# year_seasonality=5 (周期を三角関数5個で)
m2 = Prophet(
    growth="linear",
    yearly_seasonality=5,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative").fit(train)
forecast2 = m2.predict(df=future)
fig = m2.plot(forecast2)
fig = m2.plot_components(forecast2)
fig.savefig("figure/05_prophet_year5.png")
fig.clf()

# year_seasonality=1 (周期を三角関数1個で)
m3 = Prophet(growth="linear",
yearly_seasonality=1,
weekly_seasonality=False,
daily_seasonality=False,
seasonality_mode="multiplicative").fit(train)
forecast3 = m3.predict(df=future)
fig = m3.plot(forecast2)
fig = m3.plot_components(forecast2)
fig.savefig("figure/06_prophet_year1.png")
fig.clf()

# トレンドの変化点
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
fig.savefig("figure/07_prophet_changepoints.png")
fig.clf()

# ラプラス分布
scale = 1
x = np.arange(-8, 8, .01)
pdf = np.exp(-abs(x) / scale) / (2*scale)
plt.plot(x, pdf)
plt.savefig("figure/08_prophet_laplace.png")
plt.clf()

# ラプラスの裾
x = np.arange(-0.5, 0.5, 0.01)
scale1 = 0.05
scale2 = 0.1

pdf_05 = np.exp(-abs(x)/scale1) / (2*scale1)
pdf_10 = np.exp(-abs(x)/scale2) / (2*scale2)
plt.plot(x, pdf_05, label="scale: 0.05")
plt.plot(x, pdf_10, label="scale: 0.10")
plt.legend()
plt.savefig("figure/09_prophet_laplace.png")

# ラプラス分布のスケールパラメータ
m4 = Prophet(
    growth="linear",
    yearly_seasonality=10,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.1
).fit(train)
forecast4 = m4.predict(df=future)
fig4 = m4.plot(forecast4)
a = add_changepoints_to_plot(fig4.gca(), m4, forecast)
fig4.savefig("figure/10_prophet_changepoints_scale.png")
fig4.clf()

# 自己相関
sm.graphics.tsa.plot_acf(df["#Passengers"], lags=30, alpha=None)
plt.xlabel("k")
plt.ylabel("ACF")
plt.savefig("figure/11_prophet_acf.png")
plt.clf()
