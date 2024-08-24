import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
@st.cache_data
def load_data():
    df = pd.read_csv('/home/murali/Downloads/commodity.csv')
    return df
data = load_data()

st.write("## Agricultural Commodity Prices")
st.dataframe(data)

numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_columns) == 0:
    st.error("No numeric data available in the dataset.")
else:
    commodity = st.selectbox("Select a commodity to forecast", numeric_columns)

    data[commodity] = pd.to_numeric(data[commodity], errors='coerce')
    data = data.dropna(subset=[commodity])

    if data.empty:
        st.error(f"No valid data available for {commodity} after cleaning. Please select a different commodity.")
    else:
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data[commodity].values.reshape(-1, 1))

        order = st.selectbox("Select ARIMA order (p, d, q)", [(1, 1, 1), (2, 1, 2), (1, 0, 1)])
        model = ARIMA(data_scaled, order=order)
        model_fit = model.fit()

        steps = st.slider("Number of steps to forecast into the future", 1, 100, 10)
        forecast = model_fit.forecast(steps=steps)
        forecast = scaler.inverse_transform(forecast.reshape(-1, 1))  # Inverse transform to original scale

        original_prices = scaler.inverse_transform(data_scaled)
        forecast_index = range(len(data), len(data) + steps)
        forecast_series = pd.Series(forecast.flatten(), index=forecast_index)

        st.write(f"## {commodity} - Original Prices")
        fig, ax = plt.subplots()
        ax.plot(data.index, original_prices, label='Original Prices', color='blue')
        ax.set_xlabel("Index")
        ax.set_ylabel("Price (INR/kg)")
        ax.set_title(f'{commodity} - Original Prices')
        ax.legend()
        st.pyplot(fig)

        st.write(f"## {commodity} - Forecasted Prices")
        fig, ax = plt.subplots()
        ax.plot(forecast_series.index, forecast_series, label='Forecasted Prices', color='red', linestyle='--')
        ax.set_xlabel("Index")
        ax.set_ylabel("Price (INR/kg)")
        ax.set_title(f'{commodity} - Forecasted Prices')
        ax.legend()
        st.pyplot(fig)

        st.write(f"## Forecasted Prices for Next {steps} Steps")
        st.write(forecast)
