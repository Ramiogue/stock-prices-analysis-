import streamlit as st
import joblib
import pandas as pd
import numpy as np
from fbprophet import Prophet

# ✅ Load the best saved Prophet model
model = joblib.load("best_prophet_model.pkl")

# ✅ Streamlit UI
st.title("📈 Tesla Stock Price Prediction App")
st.write("🚀 This app predicts Tesla stock prices using a fine-tuned Prophet model.")

# ✅ User selects forecast period
days = st.slider("Select number of days to forecast:", min_value=1, max_value=200, value=30)

# ✅ Generate future dates
future_dates = model.make_future_dataframe(periods=days)

# ✅ Predict future stock prices
forecast = model.predict(future_dates)

# ✅ Plot results
st.subheader("📊 Stock Price Forecast")
st.line_chart(forecast.set_index("ds")["yhat"])

# ✅ Download CSV Button
csv = forecast[['ds', 'yhat']][-days:].to_csv(index=False)
st.download_button(label="📥 Download Forecast", data=csv, file_name="Tesla_Stock_Forecast.csv")

st.write("✅ Model trained using historical stock data & tuned for accuracy.")
