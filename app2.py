import streamlit as st
import pandas as pd
import base64
import datetime
import io
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from pmdarima import auto_arima
import plotly.graph_objects as go

# Custom hash function for complex numbers
def hash_complex(c):
    return hash((c.real, c.imag))

# Load data
@st.cache_data(hash_funcs={complex: hash_complex})
def load_data(file_path):
    df_stock = pd.read_csv(file_path)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock.set_index('Date', inplace=True)
    return df_stock

df_stock = load_data('f_yahoo_stock.csv')

# Streamlit App
st.title('Stock Price Prediction')

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df_stock.index,
                open=df_stock['Open'], high=df_stock['High'],
                low=df_stock['Low'], close=df_stock['Close'],
                increasing_line_color='green', decreasing_line_color='red')])
fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price',
                  xaxis_rangeslider_visible=False)
st.plotly_chart(fig)

# Date selection widgets
start_date = st.date_input("Select start date:", datetime.datetime(2015, 11, 23), disabled=True)
end_date = st.date_input("Select end date:", datetime.datetime(2020, 11, 20))
num_days = st.selectbox("Select number of days for predictions:", [30, 60, 90, 120, 150, 180])

# Model fitting functions
@st.cache_data(hash_funcs={complex: hash_complex})
def fit_model(df, model):
    if model == 'SARIMA':
        # Define SARIMA parameters
        p = 1  # AR order
        d = 1  # I order
        q = 1  # MA order
        P = 1  # Seasonal AR order
        D = 1  # Seasonal I order
        Q = 1  # Seasonal MA order
        m = 12  # Seasonal period (daily)
        # Fit SARIMA model
        sarima_model = SARIMAX(df['Adj Close'], order=(p, d, q), seasonal_order=(P, D, Q, m))
        return sarima_model.fit()

    elif model == 'ARIMA':
        # Fit AutoARIMA model
        autoarima_model = auto_arima(df['Adj Close'], seasonal=True, m=12, max_order=5, stepwise=True, suppress_warnings=True)
        return autoarima_model

    elif model == 'Prophet':
        # Fit Prophet model
        prophet_model = Prophet()
        prophet_df = df.reset_index().rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        prophet_model.fit(prophet_df)
        return prophet_model

# Model selection widget
selected_model = st.radio('Select Model', ['SARIMA', 'ARIMA', 'Prophet'])

# Fit the selected model
model_fit = fit_model(df_stock.loc[start_date:end_date], selected_model)

# Generate predictions
if selected_model == 'SARIMA':
    # Fit SARIMA model
    prediction_start_date = end_date + datetime.timedelta(days=1)
    prediction_end_date = prediction_start_date + datetime.timedelta(days=num_days - 1)  # Predict for the selected number of days
    predictions = model_fit.predict(start=prediction_start_date, end=prediction_end_date)
elif selected_model == 'ARIMA':
    # Fit AutoARIMA model
    predictions = model_fit.predict(n_periods=num_days)
elif selected_model == 'Prophet':
    # Fit Prophet model
    future = pd.date_range(start=end_date + datetime.timedelta(days=1), periods=num_days)
    future_df = pd.DataFrame({'ds': future})
    predictions = model_fit.predict(future_df)



# Plot actual and predicted adjusted close prices using Plotly
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Adj Close'], mode='lines', name='Actual Adjusted Close'))
if selected_model == 'Prophet':
    fig2.add_trace(go.Scatter(x=predictions['ds'], y=predictions['yhat'], mode='lines', name=f'Predicted Adjusted Close ({selected_model})'))
else:
    fig2.add_trace(go.Scatter(x=predictions.index, y=predictions, mode='lines', name=f'Predicted Adjusted Close ({selected_model})'))
fig2.update_layout(title='Actual and Predicted Adjusted Close Price', xaxis_title='Date', yaxis_title='Adjusted Close Price', showlegend=True)
st.plotly_chart(fig2)
