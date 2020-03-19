from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import pandas as pd


df = pd.read_csv("data/input/example_wp_log_R.csv")
print(df.head())

m = Prophet(changepoint_prior_scale=0.5)
m.fit(df)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# 変化点プロット
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
fig.savefig("figure/16_prophet_changepoints.png")