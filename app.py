from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask_cors import CORS
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.model_selection import train_test_split
from datetime import datetime

app = Flask(__name__)
CORS(app)

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
        new_data_df = new_data_df.sort_values(by='date', ascending=True)
        return new_data_df
    else:
        return None

# Preprocess data and create features
def preprocess_data(data_df):
    data_df['dayofweek'] = data_df['date'].dt.dayofweek
    data_df['quarter'] = data_df['date'].dt.quarter
    data_df['month'] = data_df['date'].dt.month
    data_df['year'] = data_df['date'].dt.year
    data_df['dayofyear'] = data_df['date'].dt.dayofyear
    data_df['dayofmonth'] = data_df['date'].dt.day
    data_df['weekofyear'] = data_df['date'].dt.isocalendar().week
    data_df = data_df.set_index('date')
    
    # Create lag features
    for lag in range(1, 8):
        data_df[f'lag_{lag}'] = data_df['price'].shift(lag)
        
    # Fill NaN values with the price from the most recent available data
    data_df = data_df.fillna(method='bfill')
    print("Preprocessed DataFrame:\n", data_df)
    return data_df

# Train the Random Forest model and store feature names
def train_random_forest(data_df):
    X = data_df.drop(['price'], axis=1)
    y = data_df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train.columns

# Predict future prices
def predict_future_prices(model, feature_names, data_df, forecast_days):
    last_row = data_df.iloc[-1]
    future_dates = pd.date_range(start=data_df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    
    for lag in range(1, 8):
        future_df[f'lag_{lag}'] = last_row[f'lag_{lag-1}'] if lag > 1 else last_row['price']
    
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['quarter'] = future_df.index.quarter
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['dayofmonth'] = future_df.index.day
    future_df['weekofyear'] = future_df.index.isocalendar().week
    
    # Ensure future_df has the same columns as the training data in the same order
    future_df = future_df[feature_names]
    
    forecast = model.predict(future_df)
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast_price': forecast})
    
    return forecast_df

# Decision making function
def should_fill_up(current_miles, daily_driving_distance, data_df, forecast_df):
    days_until_empty = current_miles / daily_driving_distance
    today_price = data_df['price'].iloc[-1]
    future_prices = forecast_df['forecast_price'][:int(days_until_empty)]
    optimal_future_price = min(future_prices)
    optimal_date = forecast_df[forecast_df['forecast_price'] == optimal_future_price]['date'].values[0]
    
    # Convert numpy datetime64 to datetime
    optimal_date = pd.to_datetime(optimal_date).strftime('%B %d')
    
    if today_price <= optimal_future_price and days_until_empty > 0:
        return "Fill up now"
    else:
        return f"Fill up on {optimal_date}"

@app.route('/', methods=['GET', 'POST'])
def index():
    decision = None
    graphJSON = None

    if request.method == 'POST':
        current_miles = float(request.form['current_miles'])
        daily_driving_distance = float(request.form['daily_driving_distance'])
        
        # Calculate days until empty
        days_until_empty = int(current_miles / daily_driving_distance)
        
        data_df = fetch_gas_data()
        data_df = preprocess_data(data_df)
        
        model, feature_names = train_random_forest(data_df)
        
        forecast_df = predict_future_prices(model, feature_names, data_df, days_until_empty)
        
        decision = should_fill_up(current_miles, daily_driving_distance, data_df, forecast_df)

        # Create Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_df.index, y=data_df['price'], mode='lines', name='Historical Prices'))
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast_price'], mode='lines', name='Forecasted Prices', line=dict(dash='dash')))
        fig.update_layout(title='Gas Prices: Historical and Forecasted', xaxis_title='Date', yaxis_title='Price')
        
        graphJSON = pio.to_json(fig)

        print(forecast_df)
        print(data_df)

    return render_template('index.html', decision=decision, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
