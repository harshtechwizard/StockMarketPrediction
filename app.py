import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import date, timedelta
import glob

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title('📈 Stock Market Price Predictor')

# ---------- Scan for available .keras models ----------
model_files = glob.glob("*.keras")
if not model_files:
    st.error("❌ No trained models found. Please train a model first.")
    st.stop()

# Helper to extract model type from filename
model_labels = []
model_types = []

for f in model_files:
    if f.strip().lower() == "stock predictions model.keras":
        model_labels.append(f"LSTM: {f}")
        model_types.append("LSTM")
    elif f.strip().lower() == "gru_model.keras":
        model_labels.append(f"GRU: {f}")
        model_types.append("GRU")
    else:
        model_labels.append(f"Unknown: {f}")
        model_types.append("Unknown")

selected_label = st.sidebar.selectbox("🧠 Select Trained Model", model_labels)
selected_index = model_labels.index(selected_label)
selected_model_file = model_files[selected_index]
selected_model_type = model_types[selected_index]
model = load_model(selected_model_file)
st.sidebar.success(f"Model Loaded: {selected_model_file}")

# Info box for model type
st.sidebar.info(f"**Model Type:** {selected_model_type}")

# Stock input
stock = st.sidebar.selectbox(
    '📊 Select Stock Symbol',
    ['GOOG', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'META'],
    index=0
)

# Date range
start = st.sidebar.date_input("📅 Start Date", date(2012, 1, 1))
end = st.sidebar.date_input("📅 End Date", date(2022, 12, 31))

# ---------- Download stock data ----------
data = yf.download(stock, start, end, auto_adjust=False)
st.subheader(f'📊 Stock Data for {stock}')
st.dataframe(data.tail())

# ---------- Moving Averages ----------
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

st.markdown("## 📈 Moving Average Analysis")

for ma, color, label in [(ma_50_days, 'r', 'MA50'), (ma_100_days, 'b', 'MA100'), (ma_200_days, 'purple', 'MA200')]:
    fig = plt.figure(figsize=(10, 4))
    plt.plot(ma, color, label=label)
    plt.plot(data.Close, 'g', label='Closing Price')
    plt.legend()
    st.pyplot(fig)

# ---------- Data Preparation ----------
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))
past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_full)

x, y = [], []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])
x, y = np.array(x), np.array(y)

# ---------- Prediction ----------
predicted = model.predict(x)
scale = 1 / scaler.scale_
predicted = predicted * scale
y = y * scale

# ---------- Accuracy ----------
rmse = np.sqrt(mean_squared_error(y, predicted))
mae = mean_absolute_error(y, predicted)
r2 = r2_score(y, predicted)

st.markdown("## 📐 Prediction Accuracy Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

# ---------- Final Plot ----------
st.markdown("## 🔮 Original vs Predicted Prices")
fig4 = plt.figure(figsize=(12, 5))
plt.plot(y, 'g', label='Actual Price')
plt.plot(predicted, 'r', label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig4)

# # ---------- Future Forecast ----------
# st.markdown("## 🔮 Future Forecast (Next 30 Days)")

# full_data = pd.concat([data_train, data_test], ignore_index=True)
# full_scaled = scaler.fit_transform(full_data)

# future_input = full_scaled[-100:].tolist()
# future_predictions = []

# for _ in range(30):
#     current_input = np.array(future_input[-100:]).reshape(1, 100, 1)
#     pred = model.predict(current_input, verbose=0)
#     future_predictions.append(pred[0][0])
#     future_input.append([pred[0][0]])

# # Convert predictions to original scale
# future_predictions = np.array(future_predictions).reshape(-1)
# last_actual_price = data.Close.iloc[-1]
# full_forecast = np.concatenate((np.array([last_actual_price]).ravel(), future_predictions))

# # Dates + Plotting
# last_date = data.index[-1]
# forecast_dates = [last_date + timedelta(days=i) for i in range(0, 31)]

# fig5 = plt.figure(figsize=(12, 5))
# plt.plot(forecast_dates, full_forecast, marker='o', linestyle='--', label="Forecast", color='orange')
# plt.title("📆 Next 30 Days Forecast (with Dates)")
# plt.xlabel("Date")
# plt.ylabel("Predicted Price")
# plt.xticks(rotation=45)
# plt.legend()
# st.pyplot(fig5)