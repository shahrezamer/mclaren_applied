import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import math


class clubPredictor(object):
    def __init__(self, planet, shot_distance) -> None:
        self.planet = planet
        self.shot_distance = shot_distance
        self.__preprocess_data()

    def __calculate_initial_velocity(self, swing_speed, loft_angle_deg):
        loft_angle_rad = math.radians(loft_angle_deg)
        initial_velocity = 1.5 * swing_speed * (1 + math.sin(loft_angle_rad))
        return initial_velocity

    def __preprocess_data(self):
        self.club_data = pd.read_csv(Path.cwd() / 'data/club_data.csv')
        self.planet_data = pd.read_csv(Path.cwd() / 'data/planet_data.csv')
        self.training_data = pd.read_csv(Path.cwd() / 'data/training_data.csv')
        self.test_data = pd.read_csv(Path.cwd() / 'data/test_data.csv')

        data = self.training_data.merge(
            self.planet_data, on='planet').merge(self.club_data, on='club')
        data['initial_velocity'] = data.apply(lambda row: self.__calculate_initial_velocity(
            row['swing_speed (m/s)'], row['loft_angle (deg)']), axis=1)

        t_data = self.test_data.merge(
            self.planet_data, on='planet').merge(self.club_data, on='club')
        t_data['initial_velocity'] = t_data.apply(lambda row: self.__calculate_initial_velocity(
            row['swing_speed (m/s)'], row['loft_angle (deg)']), axis=1)
        self.t_data = t_data

        self.X_train = data[[
            'initial_velocity', 'air_density (kg/m^3)', 'gravity (m/s^2)', 'swing_speed (m/s)', 'loft_angle (deg)']]
        self.y_train = data['shot_distance (m)']

        self.X_test = t_data[[
            'initial_velocity', 'air_density (kg/m^3)', 'gravity (m/s^2)', 'swing_speed (m/s)', 'loft_angle (deg)']]
        self.y_test = t_data['shot_distance (m)']

    def predict(self):
        # Fit a linear regression model
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)

        planet_data = self.planet_data[self.planet_data['planet']
                                       == self.planet].reset_index()
        self.club_data['club_velocity'] = self.club_data.apply(
            lambda x: self.__calculate_initial_velocity(x['swing_speed (m/s)'], x['loft_angle (deg)']), axis=1)

        df = pd.DataFrame({
            'initial_velocity': self.club_data['club_velocity'],
            'air_density (kg/m^3)': planet_data.loc[0, 'air_density (kg/m^3)'],
            'gravity (m/s^2)': planet_data.loc[0, 'gravity (m/s^2)'],
            'swing_speed (m/s)': self.club_data['swing_speed (m/s)'],
            'loft_angle (deg)': self.club_data['loft_angle (deg)']
        })
        distances = model.predict(df)
        self.club_data['distance'] = abs(self.shot_distance - distances)
        min_club = self.club_data.loc[self.club_data['distance'].idxmin()]['club']
        return min_club

    def test_model(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        print("R-Squared Error:", r2)
        print("MSE:", mse)
        count = 0
        print('planet name, shot distance, best club, actual club')
        for _, row in self.t_data.iterrows():
            planet_name = row['planet']
            actual_distance = row['shot_distance (m)']
            actual_club = row['club']

            self.club_data['club_velocity'] = self.club_data.apply(
                lambda x: self.__calculate_initial_velocity(x['swing_speed (m/s)'], x['loft_angle (deg)']), axis=1)

            df = pd.DataFrame({
                'initial_velocity': self.club_data['club_velocity'],
                'air_density (kg/m^3)': row['air_density (kg/m^3)'],
                'gravity (m/s^2)': row['gravity (m/s^2)'],
                'swing_speed (m/s)': self.club_data['swing_speed (m/s)'],
                'loft_angle (deg)': self.club_data['loft_angle (deg)']
            })
            distances = model.predict(df)
            self.club_data['distance'] = abs(actual_distance - distances)

            min_club = self.club_data.loc[self.club_data['distance'].idxmin(
            )]['club']

            print(planet_name, actual_distance, min_club, actual_club)

            if actual_club == min_club:
                count += 1
