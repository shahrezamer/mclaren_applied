# Classification TESTING

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
training_data = pd.read_csv('data/training_data.csv')
test_data = pd.read_csv('data/test_data.csv')
club_data = pd.read_csv('data/club_data.csv')
planet_data = pd.read_csv('data/planet_data.csv')

# Merge planet data
training_data = training_data.merge(planet_data, on='planet', how='left')
training_data = training_data.merge(club_data, on='club', how='left')
test_data = test_data.merge(planet_data, on='planet', how='left')
test_data = test_data.merge(club_data, on='club', how='left')


label_encoder = LabelEncoder()
label_encoder.fit(training_data['club'])
training_data['club'] = label_encoder.transform(training_data['club'])

# Define features (X) and target variable (y) for training and test data
X_train = training_data[['gravity (m/s^2)', 'air_density (kg/m^3)', 'shot_distance (m)','loft_angle (deg)', 'swing_speed (m/s)']]
y_train = training_data['club']
X_test = test_data[['gravity (m/s^2)', 'air_density (kg/m^3)', 'shot_distance (m)','loft_angle (deg)', 'swing_speed (m/s)']]

# Create and train the Random Forest Classifier model
model = RandomForestClassifier(random_state=19, n_estimators=5)
model.fit(X_train, y_train)

# Make club predictions for the test data
y_pred_labels = model.predict(X_test)

# Decode the numerical club labels back to club names
y_pred = label_encoder.inverse_transform(y_pred_labels)

# Print recommended clubs for the test data
test_data['recommended_club'] = y_pred
print("Recommended Clubs for Test Data:")
print(test_data[['planet', 'shot_distance (m)','club', 'recommended_club']])


label_encoder.fit(test_data['club'])
test_data_encoded = label_encoder.transform(test_data['club'])
# Evaluate the model's performance (if needed)
mse = mean_squared_error(test_data_encoded, y_pred_labels)
r2 = r2_score(test_data_encoded, y_pred_labels)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
