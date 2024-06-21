import logging
import os
import requests
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render
from django.http import HttpResponse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.seasonal import STL
from xgboost import XGBRegressor



# Get an instance of a logger
logger = logging.getLogger(__name__)


def get_image(request, image_name):
    """
    Retrieves and returns the image specified by image_name.
    """
    with open(image_name, "rb") as f:
        return HttpResponse(f.read(), content_type="image/png")

# Function to load weather data
def load_weather_data():
    data = pd.read_csv('kanpur.csv')
    data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
    data['date'] = data['date_time'].dt.date
    daily_data = data.groupby('date').agg({'maxtempC': 'max', 'mintempC': 'min'}).reset_index()
    daily_data['date'] = pd.to_datetime(daily_data['date'])
    daily_data['year'] = daily_data['date'].dt.year
    daily_data['month'] = daily_data['date'].dt.month
    daily_data['day_of_year'] = daily_data['date'].dt.dayofyear
    daily_data['sin_day'] = np.sin(2 * np.pi * daily_data['day_of_year'] / 365.25)
    daily_data['cos_day'] = np.cos(2 * np.pi * daily_data['day_of_year'] / 365.25)
    return daily_data

# Train the Random Forest Regressor models
def train_models(data):
    features = ['year', 'month', 'day_of_year', 'sin_day', 'cos_day']
    X = data[features]
    y_max = data['maxtempC']
    y_min = data['mintempC']

    X_train, X_test, y_max_train, y_max_test, y_min_train, y_min_test = train_test_split(
        X, y_max, y_min, test_size=0.2, random_state=42)

    rf_max = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_min = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_max.fit(X_train, y_max_train)
    rf_min.fit(X_train, y_min_train)

    y_max_pred = rf_max.predict(X_test)
    y_min_pred = rf_min.predict(X_test)

    mae_max = mean_absolute_error(y_max_test, y_max_pred)
    mae_min = mean_absolute_error(y_min_test, y_min_pred)

    print(f'Mean Absolute Error for Max Temp: {mae_max}')
    print(f'Mean Absolute Error for Min Temp: {mae_min}')

    return rf_max, rf_min

# Function to predict future temperatures
def predict_temperature(date, rf_max, rf_min):
    date = pd.to_datetime(date, format='%Y-%m-%d')
    features = {
        'year': date.year,
        'month': date.month,
        'day_of_year': date.dayofyear,
        'sin_day': np.sin(2 * np.pi * date.dayofyear / 365.25),
        'cos_day': np.cos(2 * np.pi * date.dayofyear / 365.25),
    }
    features_df = pd.DataFrame(features, index=[0])
    max_temp = rf_max.predict(features_df)[0]
    min_temp = rf_min.predict(features_df)[0]
    return max_temp, min_temp

# Forecast view
def forecast(request):
    if request.method == 'POST':
        selected_date_str = request.POST.get('selected_date')
        selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d')

        # Load and preprocess weather data
        data = load_weather_data()

        # Train the models
        rf_max, rf_min = train_models(data)

        # Predict temperatures for the future date
        max_temp, min_temp = predict_temperature(selected_date_str, rf_max, rf_min)

        # Plot the results
        plt.figure(figsize=(14, 7))

        # Plot historical data
        plt.plot(data['date'], data['maxtempC'], color='blue', label='Historical Max Temp')
        plt.plot(data['date'], data['mintempC'], color='green', label='Historical Min Temp')

        # Plot the predicted data
        plt.axvline(x=selected_date, color='red', linestyle='--', label='Prediction Date')
        plt.scatter([selected_date], [max_temp], color='red', label='Predicted Max Temp')
        plt.scatter([selected_date], [min_temp], color='orange', label='Predicted Min Temp')

        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.title('Weather Forecast')
        plt.legend()
        plt.tight_layout()

        # Save the plot to a file
        plot_filename = 'static/images/Prediction1.png'
        plt.savefig(plot_filename)
        plt.close()

        # Prepare the context for rendering
        context = {
            'selected_date': selected_date_str,
            'min_temperature': min_temp,
            'max_temperature': max_temp,
            'plot_filename': plot_filename
        }

        return render(request, 'forecast.html', context)

    # Handle GET request
    else:
        return render(request, 'forecast.html')
    
def index(request):
    """
    Render the index page.
    """
    logger.info('Rendering index page')
    return render(request, "index.html")


def about(request):
    """
    Render the about page.
    """
    return render(request, "about.html")


def predict(request):
    """
    Predict view: Loads weather data, performs data analysis and predictions,
    and renders the predict.html page with the results.
    """
    try:
        logger.info('Loading weather data')

        # Load weather data
        weather_df = pd.read_csv('kanpur.csv', parse_dates=['date_time'], index_col='date_time')

        # Select relevant columns
        weather_df_num = weather_df[['maxtempC', 'mintempC', 'cloudcover', 'humidity', 'tempC', 'sunHour', 'HeatIndexC', 'precipMM', 'pressure', 'windspeedKmph']]

        # Resample to weekly data to reduce the size of the dataset
        weekly_weather_df_num = weather_df_num.resample('W').mean()

        # Plot all features over time (weekly data)
        weekly_weather_df_num.plot(subplots=True, figsize=(25, 20))
        plt.savefig('All1.png')
        plt.close()

        # Plot features for a specific time period (daily data for 2019-2020)
        weather_df_num['2019':'2020'].resample('D').ffill().plot(subplots=True, figsize=(25, 20))
        plt.savefig('All2.png')
        plt.close()

        # Plot histograms of features (weekly data)
        weekly_weather_df_num.hist(bins=10, figsize=(15, 15))
        plt.savefig('Hist1.png')
        plt.close()

        # Select data for scatter plots (weekly data for 2019-2020)
        weth = weather_df_num['2019':'2020'].resample('W').mean()
    

        # Scatter plot: Minimum Temperature vs Temperature
        plt.scatter(weth.mintempC, weth.tempC)
        plt.xlabel("Minimum Temperature")
        plt.ylabel("Temperature")
        plt.savefig('Scatter1.png')
        plt.close()

        # Scatter plot: Heat Index vs Temperature
        plt.scatter(weth.HeatIndexC, weth.tempC)
        plt.xlabel("Heat Index")
        plt.ylabel("Temperature")
        plt.savefig('Scatter2.png')
        plt.close()

        # Scatter plot: Pressure vs Temperature
        plt.scatter(weth.pressure, weth.tempC)
        plt.xlabel("Pressure")
        plt.ylabel("Temperature")
        plt.savefig('Scatter3.png')
        plt.close()

        # Scatter plot: Maximum Temperature vs Temperature
        plt.scatter(weth.maxtempC, weth.tempC)
        plt.xlabel("Maximum Temperature")
        plt.ylabel("Temperature")
        plt.savefig('Scatter4.png')
        plt.close()

        # Compute and plot the correlation matrix (weekly data)
        corr_matrix = weekly_weather_df_num.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.savefig('Correlation_Matrix.png')
        plt.close()

        # Time Series Decomposition (weekly data)
        result = seasonal_decompose(weekly_weather_df_num['tempC'].dropna(), model='additive', period=52)
        result.plot()
        plt.savefig('Time_Series_Decomposition.png')
        plt.close()

        # Pairplot of features (weekly data)
        sns.pairplot(weekly_weather_df_num)
        plt.savefig('Pairplot.png')
        plt.close()

        # Random Forest Prediction

        # Load your weather data
        data = pd.read_csv('kanpur.csv')

        # Convert 'date_time' column to datetime
        data['date_time'] = pd.to_datetime(data['date_time'])

        # Set 'date_time' as index
        data.set_index('date_time', inplace=True)

        # Select only the 'maxtempC' column and resample to weekly frequency
        weekly_data = data['maxtempC'].resample('W').mean()

        # Handle missing values if any
        weekly_data.ffill(inplace=True)

        # Create lagged features
        def create_lagged_features(data, num_lags):
            df = pd.DataFrame(data)
            columns = [df.shift(i) for i in range(1, num_lags+1)]
            columns.append(df)
            df = pd.concat(columns, axis=1)
            df.dropna(inplace=True)
            return df

        # Create lagged features for weekly data
        num_lags = 52  # Use one year of weekly lags
        lagged_data = create_lagged_features(weekly_data, num_lags)

        # Split the data into features (X) and target (y)
        X = lagged_data.iloc[:, :-1].values
        y = lagged_data.iloc[:, -1].values

        # Split the data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-train_size/len(X)), shuffle=False)

        # Fit the Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict future values
        future_steps = int(0.5 * len(weekly_data))
        future_predictions = []

        # Use the last available data point to start predictions
        last_data_point = X[-1, :]

        for _ in range(future_steps):
            pred = model.predict(last_data_point.reshape(1, -1))[0]
            future_predictions.append(pred)
            last_data_point = np.roll(last_data_point, -1)
            last_data_point[-1] = pred

        # Generate future date range
        last_date = weekly_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=future_steps, freq='W')

        # Create a DataFrame for future predictions
        predicted_series = pd.Series(future_predictions, index=future_dates)

        # Plot the results
        plt.figure(figsize=(14, 7))

        # Plot the historical data
        plt.plot(weekly_data.index, weekly_data, color='blue', label='Historical Data')

        # Plot the future predictions
        plt.plot(predicted_series.index, predicted_series, color='red', linestyle='--', label='Future Predictions')

        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Max Temperature (C)')
        plt.title('Future Weather Prediction using Random Forest (Weekly Data)')
        plt.legend()
        plt.savefig('Prediction.png')
        plt.close()

        return render(request, 'predict.html')
    
    except Exception as e:
        logger.error(f'Error in predict view: {str(e)}')
        return render(request, 'error.html', {'message': 'An error occurred. Please try again later.'})



def top(request):
    """
    Finds the top hottest and coldest cities using OpenWeatherMap API and renders the top.html page.
    """
    try:
        logger.info('Finding top hottest and coldest cities')

        # Function to get current temperature of a city using OpenWeatherMap API
        def get_temperature(city):
            api_key = ""  # Replace with your own API key
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
            response = requests.get(url)
            data = response.json()
            temperature = data["main"]["temp"]
            return temperature

        # Function to find the top hottest cities
        def find_hottest_cities(cities):
            city_temperatures = {}
            for city in cities:
                temperature = get_temperature(city)
                city_temperatures[city] = temperature
            hottest_cities = sorted(city_temperatures, key=city_temperatures.get, reverse=True)[:5]
            return hottest_cities

        # Function to find the top coldest cities
        def find_coldest_cities(cities):
            city_temperatures = {}
            for city in cities:
                temperature = get_temperature(city)
                city_temperatures[city] = temperature
            coldest_cities = sorted(city_temperatures, key=city_temperatures.get)[:5]
            return coldest_cities

        # List of cities to analyze
        cities = ["Mumbai", "Bhopal", "Chennai", "Bengaluru", "Hyderabad", "Nagpur", "Shillong", "Nagpur", "Churu", "Bilaspur", "Banda", "Jhansi", "Phalodi", "Kargil", "Srinagar", "Agra", "Sri Ganganagar", "Surat","Imphal","Aizwal","Gangtok"]
        hottest_cities = find_hottest_cities(cities)
        coldest_cities = find_coldest_cities(cities)
        hot_cities={}
        for city in hottest_cities:
            temperature = get_temperature(city)
            hot_cities[city]=temperature


        cold_cities={}
        for city in coldest_cities:
            temperature = get_temperature(city)
            cold_cities[city]=temperature
        return render(request, "top.html",{"cold_cities1": cold_cities,"hot_cities1": hot_cities})
    
    except Exception as e:
        logger.error(f'Error in get_image view: {str(e)}')
        return render(request, 'error.html', {'message': 'An error occurred. Please try again later.'})
