from flask import Flask, render_template, request
import requests
import pandas as pd
import os
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Fetch and store gas data function
def fetch_gas_data():
    url = 'https://api.eia.gov/v2/seriesid/PET.EMM_EPM0U_PTE_NUS_DPG.W'
    api_key = '8v7KYPFapVsagpCVQqaclCMeRJ40yv2s21Hh3H14'
    params = {'api_key': api_key}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        tabular_data = [(x['period'], x['value']) for x in data['response']['data']]
        new_data_df = pd.DataFrame(tabular_data, columns=['date', 'price'])
        new_data_df['date'] = pd.to_datetime(new_data_df['date'], format='%Y-%m-%d')

        onedrive_path = os.path.expanduser('/Users/jackson/OneDrive - University of Waterloo/4A/msci436/')
        csv_file_path = os.path.join(onedrive_path, 'gasData.csv')

        if os.path.exists(csv_file_path):
            existing_data_df = pd.read_csv(csv_file_path, parse_dates=['date'])
            combined_data_df = pd.concat([existing_data_df, new_data_df]).drop_duplicates(subset=['date']).reset_index(drop=True)
            combined_data_df.to_csv(csv_file_path, index=False)
        else:
            new_data_df.to_csv(csv_file_path, index=False)
        
        return combined_data_df
    else:
        return None

# Predict and create forecast function
def predict_gas_prices(data_df):
    data_df.set_index('date', inplace=True)
    model = ARIMA(data_df['price'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=7)
    forecast_dates = pd.date_range(data_df.index[-1], periods=7, freq='D')
    forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast_price': forecast})
    forecast_df.set_index('date', inplace=True)
    return forecast_df

# Decision making function
def should_fill_up(current_miles, consumption_rate, data_df, forecast_df):
    days_until_empty = current_miles / consumption_rate
    today_price = data_df['price'].iloc[-1]
    future_prices = forecast_df['forecast_price'][:int(days_until_empty)]
    avg_future_price = future_prices.mean()
    if today_price <= avg_future_price:
        return "Fill up now"
    else:
        return "Wait to fill up"

@app.route('/', methods=['GET', 'POST'])
def index():
    decision = None
    graphJSON = None

    if request.method == 'POST':
        current_miles = float(request.form['current_miles'])
        consumption_rate = float(request.form['consumption_rate'])
        
        data_df = fetch_gas_data()
        forecast_df = predict_gas_prices(data_df)
        
        decision = should_fill_up(current_miles, consumption_rate, data_df, forecast_df)

        # Create Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_df.index, y=data_df['price'], mode='lines', name='Historical Prices'))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['forecast_price'], mode='lines', name='Forecasted Prices', line=dict(dash='dash')))
        fig.update_layout(title='Gas Prices: Historical and Forecasted', xaxis_title='Date', yaxis_title='Price')
        
        graphJSON = pio.to_json(fig)

    return render_template('index.html', decision=decision, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
