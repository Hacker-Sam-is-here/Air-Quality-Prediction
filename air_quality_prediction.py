#Importing the Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Step 1: Load the Data
data = pd.read_csv('updated_pollution_dataset.csv')

# Step 2: Preprocess the Data
# Convert categorical labels into numerical values
label_encoder = LabelEncoder()
data['Air Quality'] = label_encoder.fit_transform(data['Air Quality'])

# Step 3: Feature Selection
X = data.drop('Air Quality', axis=1)  # Features
y = data['Air Quality']  # Target variable

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
# joblib.dump(model, 'air_quality_model.pkl')

# Load the model
loaded_model = joblib.load('air_quality_model.pkl')

# Predict Air Quality using the loaded model
def predict_air_quality(features):
    # Convert features to DataFrame with the same column names as the training data
    features_df = pd.DataFrame([features], columns=X.columns)
    prediction = loaded_model.predict(features_df)
    return label_encoder.inverse_transform(prediction)[0]

# Features: [Temperature,Humidity,PM2.5,PM10,NO2,SO2,CO,Proximity_to_Industrial_Areas,Population_Density]
features = [29.8,59.1,5.2,17.9,18.9,9.2,1.72,6.3,319]
print("Predicted Air Quality:", predict_air_quality(features))

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
