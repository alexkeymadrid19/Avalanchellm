import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Constants
num_samples = 1000

# Generate synthetic weather data
data = {
    'temperature': np.random.uniform(-10, 10, num_samples),  # in Celsius
    'precipitation': np.random.uniform(0, 200, num_samples),  # in mm
    'wind_speed': np.random.uniform(0, 30, num_samples),      # in km/h
    'humidity': np.random.uniform(0, 100, num_samples),      # in percentage
    'snow_depth': np.random.uniform(0, 300, num_samples)     # in cm
}

# Create a DataFrame
weather_data = pd.DataFrame(data)

# Generate risk labels based on conditions
# Example conditions: higher precipitation, wind speed, and snow depth increase risk
conditions = (weather_data['precipitation'] > 50) & (weather_data['wind_speed'] > 15) & (weather_data['snow_depth'] > 100)
weather_data['risk_label'] = np.where(conditions, 'high', 'low')

# Save to CSV
weather_data.to_csv('synthetic_avalanche_data.csv', index=False)