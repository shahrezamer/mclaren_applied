import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

from lib.utils import calculate_initial_velocity


        
# Load the data from CSV files
club_data = pd.read_csv(Path.cwd() / 'data/club_data.csv')
planet_data = pd.read_csv(Path.cwd() / 'data/planet_data.csv')
training_data = pd.read_csv(Path.cwd() / 'data/training_data.csv')

test_data = pd.read_csv(Path.cwd() / 'data/test_data.csv')


data = training_data.merge(planet_data, on='planet').merge(club_data, on='club')
data['initial_velocity'] = data.apply(lambda row: calculate_initial_velocity(row['swing_speed (m/s)'], row['loft_angle (deg)']), axis=1)

t_data = test_data.merge(planet_data, on='planet').merge(club_data, on='club')
t_data['initial_velocity'] = t_data.apply(lambda row: calculate_initial_velocity(row['swing_speed (m/s)'], row['loft_angle (deg)']), axis=1)

X_train = data[['initial_velocity', 'air_density (kg/m^3)','gravity (m/s^2)', 'swing_speed (m/s)','loft_angle (deg)']]
y_train = data['shot_distance (m)']

X_test = t_data[['initial_velocity', 'air_density (kg/m^3)','gravity (m/s^2)', 'swing_speed (m/s)','loft_angle (deg)']]
y_test = t_data['shot_distance (m)']

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
model
print(r2, mse)


def prediction(data):
    count = 0
    for _, row in data.iterrows():
        planet_name = row['planet']
        actual_distance = row['shot_distance (m)']
        actual_club = row['club']

        club_data['club_velocity'] = club_data.apply(lambda x: calculate_initial_velocity(x['swing_speed (m/s)'], x['loft_angle (deg)']), axis=1)

        df = pd.DataFrame({
            'initial_velocity': club_data['club_velocity'],
            'air_density (kg/m^3)': row['air_density (kg/m^3)'],
            'gravity (m/s^2)': row['gravity (m/s^2)'],
            'swing_speed (m/s)': club_data['swing_speed (m/s)'],
            'loft_angle (deg)': club_data['loft_angle (deg)']
        })

        distances = model.predict(df)
        club_data['distance'] = abs(actual_distance - distances)

        min_club = club_data.loc[club_data['distance'].idxmin()]['club']

        print(planet_name, actual_distance, min_club, actual_club)

        if actual_club == min_club:
            count += 1

    print(count)
    
prediction(t_data)



