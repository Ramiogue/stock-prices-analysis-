import streamlit as st
import joblib
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
import io

# ✅ Load the best saved Prophet model from GitHub
MODEL_URL = "https://raw.githubusercontent.com/Ramiogue/stock-prices-analysis-/main/best_prophet_model.pkl"

@st.cache_data
def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        model_file = io.BytesIO(response.content)
        return joblib.load(model_file)
    else:
        st.error("❌ Failed to load model. Check the GitHub link or try again later.")
        return None

model = load_model()

# ✅ Streamlit UI
st.title("📈 Tesla Stock Price Prediction App")
st.write("🚀 This app predicts Tesla stock prices using a fine-tuned Prophet model.")

# ✅ User selects forecast period
days = st.slider("Select number of days to forecast:", min_value=1, max_value=200, value=30)

if model:
    # ✅ Generate future dates
    future_dates = model.make_future_dataframe(periods=days)

    # ✅ Predict future stock prices
    forecast = model.predict(future_dates)

    # ✅ Convert log-transformed prices back to normal scale
    forecast['yhat'] = np.exp(forecast['yhat'])
    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

    # ✅ Plot results
    st.subheader("📊 Stock Price Forecast")
    st.line_chart(forecast.set_index("ds")["yhat"])

    # ✅ Download CSV Button
    csv = forecast[['ds', 'yhat']][-days:].to_csv(index=False)
    st.download_button(label="📥 Download Forecast", data=csv, file_name="Tesla_Stock_Forecast.csv")

    st.write("✅ Model trained using historical stock data & tuned for accuracy.")

else:
    st.error("⚠️ Model not loaded. Please check the GitHub model link.")
