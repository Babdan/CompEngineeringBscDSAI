# DSAI302 - Capstone Project by Bogdan Itsam Dorantes-Nikolaev 042101002

# Importing register_matplotlib_converters to avoid warnings when plotting
from pandas.plotting import register_matplotlib_converters
# Importing pandas for data manipulation
import pandas as pd
# Importing matplotlib for visualisation
import matplotlib.pyplot as plt
# Importing seaborn for better visualisation
import seaborn as sns

# Registering matplotlib converters to avoid warnings
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


# Load the dataset
file_path = 'C:/Users/Bog/Downloads/DAYTON_hourly.csv'
data = load_data(file_path)

# Convert 'Datetime' to datetime object
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Check for missing values
missing_values = data.isnull().sum()

# Basic statistical summary of the data
statistical_summary = data.describe()

# Displaying the statistical summary
print(statistical_summary)

# Initial visualisation of the time series data
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
