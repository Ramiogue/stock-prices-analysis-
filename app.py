import streamlit as st
import joblib
import pandas as pd
import numpy as np
from prophet import Prophet

# ✅ Load the model directly from the app folder
@st.cache_data
def load_model():
    try:
        return joblib.load("best_prophet_model.pkl")  # Load local model file
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# ✅ Streamlit UI
st.title("📈 Tesla Stock Price Prediction App")
st.write("🚀 This app predicts Tesla stock prices using a fine-tuned Prophet model.")

# ✅ User selects forecast period
days = st.slider("Select number of days to forecast:", min_value=1, max_value=200, value=30)

if model:
    try:
        # ✅ Generate future dates
        future_dates = model.make_future_dataframe(periods=days)

        # ✅ Identify missing regressors
        required_regressors = ['MA_7', 'MA_14', 'Volatility_7', 'RSI_14']  # List all regressors used in training

        # ✅ Fill missing regressors with last known values
        for reg in required_regressors:
            if reg in future_dates.columns:
                future_dates[reg] = future_dates[reg].fillna(method='ffill')  # Use last known value
            else:
                future_dates[reg] = 0  # Default value if no data

        # ✅ Check if all regressors exist
        missing = [r for r in required_regressors if r not in future_dates.columns]
        if missing:
            st.error(f"❌ Missing regressors: {missing}")
            st.stop()

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

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

else:
    st.error("⚠️ Model not loaded. Please check the file path.")
