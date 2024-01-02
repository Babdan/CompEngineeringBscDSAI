# DSAI302 - Capstone Project by Bogdan Itsam Dorantes-Nikolaev 042101002
import subprocess
import sys
import multiprocessing
from pmdarima import auto_arima
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to install missing packages if necessary
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages
required_packages = ["pandas", "matplotlib", "seaborn", "pmdarima", "numpy", "scikit-learn"]

# Iterate through the list and install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Importing necessary libraries
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Registering matplotlib converters
register_matplotlib_converters()

# Function to load the dataset from a different path if the initial path is incorrect
def load_data(initial_path):
    while True:
        try:  # Try to load the dataset
            data_frame = pd.read_csv(initial_path)
            return data_frame
        except FileNotFoundError:  # If the file is not found, ask the user to enter the correct path
            print("File not found at", initial_path)
            initial_path = input("Please enter the correct file path: ")

# Function to apply ARIMA model to a chunk of data
def apply_arima(data_chunk):
    model = auto_arima(data_chunk['DAYTON_MW'], start_p=1, start_q=1,
                       max_p=3, max_q=3, m=12, start_P=0, seasonal=True,
                       d=1, D=1, trace=True, error_action='ignore',
                       suppress_warnings=True, stepwise=True)
    return model.summary()

def main():
    # Load the dataset
    file_path = 'C:/Users/Bog/Downloads/DAYTON_hourly.csv'
    data = load_data(file_path)

    # Convert 'Datetime' to datetime object
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    # Optimize memory usage by converting numerical columns to float32
    data['DAYTON_MW'] = data['DAYTON_MW'].astype('float32')

    # Check for missing values and basic statistical summary of the data
    missing_values = data.isnull().sum()
    statistical_summary = data.describe()
    print("Statistical Summary: \n", statistical_summary, "\nMissing Values:", missing_values, "\n")

    # Initial visualization of the time series data
    plt.figure(figsize=(12, 6))
    plt.plot(data['Datetime'], data['DAYTON_MW'])
    plt.title('Energy Usage Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('DAYTON_MW')
    plt.grid(True)
    plt.show()

    # Feature Engineering
    data['hour'] = data['Datetime'].dt.hour
    data['day_of_week'] = data['Datetime'].dt.dayofweek
    data['day_of_month'] = data['Datetime'].dt.day
    data['month'] = data['Datetime'].dt.month
    data['year'] = data['Datetime'].dt.year

    # Plotting the distribution of energy usage
    plt.figure(figsize=(12, 6))
    sns.histplot(data['DAYTON_MW'], bins=50, kde=True)
    plt.title('Distribution of Energy Usage')
    plt.xlabel('DAYTON_MW')
    plt.ylabel('Frequency')
    plt.show()

    # Plotting energy usage by hour of the day with a different color palette
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='hour', y='DAYTON_MW', data=data, hue='hour', palette='coolwarm', legend=False)
    plt.title('Energy Usage by Hour of the Day')
    plt.xlabel('Hour')
    plt.ylabel('DAYTON_MW')
    plt.show()

    # Plotting energy usage by day of the week with a different color palette
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='day_of_week', y='DAYTON_MW', data=data, hue='day_of_week', palette='coolwarm', legend=False)
    plt.title('Energy Usage by Day of the Week')
    plt.xlabel('Day of Week')
    plt.ylabel('DAYTON_MW')
    plt.show()

    # Plotting energy usage by month with a different color palette
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='month', y='DAYTON_MW', data=data, hue='month', palette='coolwarm', legend=False)
    plt.title('Energy Usage by Month')
    plt.xlabel('Month')
    plt.ylabel('DAYTON_MW')
    plt.show()

    # ARIMA Model Section with Automated Parameter Selection
    # Splitting the data into training and testing sets
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]

    # Splitting the data into chunks for parallel processing
    num_partitions = multiprocessing.cpu_count()
    data_split = np.array_split(train, num_partitions)

    # Parallel processing
    pool = multiprocessing.Pool(num_partitions)
    models = pool.map(apply_arima, data_split)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # models now contains a list of summaries from fitted ARIMA models on different data chunks
    for model_summary in models:
        print(model_summary)

# Ensure the script is run directly
if __name__ == '__main__':
    main()
