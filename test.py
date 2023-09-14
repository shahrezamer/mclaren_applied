import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
training_data = pd.read_csv('data/training_data.csv')
club_data = pd.read_csv('data/club_data.csv')
planet_data = pd.read_csv('data/planet_data.csv')

# Constants for golf ball parameters
diameter = 0.043  # meters
mass = 0.045  # kg
drag_coefficient = 0.3
lift_coefficient = 0.2

# Function to calculate initial ball speed
def calculate_initial_speed(swing_speed):
    return 1.5 * swing_speed

# Function to calculate air resistance force
def calculate_air_resistance(v, air_density):
    return 0.5 * drag_coefficient * air_density * (np.pi * diameter**2 / 4) * v**2

# Function to calculate gravitational force
def calculate_gravitational_force(planet_gravity, mass):
    return planet_gravity * mass

# Calculate initial ball speeds for each shot
training_data['initial_speed'] = calculate_initial_speed(training_data['club_data.swing_speed (m/s)'])

# Calculate forces for each shot
training_data['air_resistance_force'] = calculate_air_resistance(training_data['initial_speed'], 
                                                                 planet_data['air_density (kg/m^3)'])
training_data['gravitational_force'] = calculate_gravitational_force(planet_data['gravity (m/s^2)'], mass)

# Calculate net force on the golf ball
training_data['net_force'] = training_data['gravitational_force'] - training_data['air_resistance_force']

# Prepare data for linear regression
X = training_data[['net_force']]
y = training_data['shot_distance (m)']

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# # Make club recommendations for each planet
# planet_names = planet_data['planet'].unique()
# for planet_name in planet_names:
#     planet_info = planet_data[planet_data['planet'] == planet_name].iloc[0]
#     planet_gravity = planet_info['gravity (m/s^2)']
#     planet_air_density = planet_info['air_density (kg/m^3)']
    
#     # Calculate net force for the given planet
#     net_force = calculate_gravitational_force(planet_gravity, mass) - \
#                 calculate_air_resistance(training_data['initial_speed'], planet_air_density)
    
#     # Predict shot distance using the linear regression model
#     predicted_distance = model.predict([[net_force]])[0]
    
#     # Print club recommendation
#     print(f"Recommendation for {planet_name}:")
#     for index, row in club_data.iterrows():
#         club_name = row['club']
#         club_speed = row['swing_speed (m/s)']
#         initial_speed = calculate_initial_speed(club_speed)
#         net_force_for_club = calculate_gravitational_force(planet_gravity, mass) - \
#                              calculate_air_resistance(initial_speed, planet_air_density)
        
#         if net_force_for_club >= net_force:
#             print(f"Use {club_name} for an estimated distance of {predicted_distance:.2f} meters.")
#             break
#     print()
