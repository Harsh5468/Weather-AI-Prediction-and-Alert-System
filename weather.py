import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from werkzeug.serving import run_simple
import threading
import csv

# Load dataset
df = pd.read_csv("weather.csv")

# Display first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Visualizing data relationships using pairplot
plt.figure(figsize=(12, 8))
sns.pairplot(df[['pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'rainfall', 'sunshine']], diag_kind='kde', corner=True)
plt.show()

# Preprocess data
X = df[['pressure', 'maxtemp', 'temperature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'rainfall', 'sunshine']]
y_temp = df['temperature']
y_rain = df['rainfall'].apply(lambda x: 1 if x > 0 else 0)

# Split data (70% training, 30% testing)
X_train, X_test, y_temp_train, y_temp_test, y_rain_train, y_rain_test = train_test_split(
    X, y_temp, y_rain, test_size=0.3, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
temp_model.fit(X_train, y_temp_train)

rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(X_train, y_rain_train)

# Evaluate models
temp_pred = temp_model.predict(X_test)
rain_pred = rain_model.predict(X_test)

# Evaluate temperature model
print("Temperature Model Evaluation:")
print("Mean Absolute Error:", mean_absolute_error(y_temp_test, temp_pred))
print("R2 Score:", r2_score(y_temp_test, temp_pred))

# Evaluate rainfall model
print("Rainfall Model Evaluation:")
print("Accuracy:", rain_model.score(X_test, y_rain_test))

# Create a DataFrame for actual vs predicted values
results_df = pd.DataFrame({
    'Actual Temperature': y_temp_test.values,
    'Predicted Temperature': temp_pred
})

# Display actual vs predicted values
print("First 10 rows of actual vs predicted temperatures:")
print(results_df.head(10))

# Plot training vs testing data distribution
plt.figure(figsize=(10, 5))
sns.histplot(y_temp_train, color="blue", label="Training Data", kde=True, bins=30)
sns.histplot(y_temp_test, color="red", label="Testing Data", kde=True, bins=30)
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.title("Training vs Testing Data Distribution")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Plot actual vs predicted temperatures
plt.figure(figsize=(10, 5))
sns.lineplot(data=results_df, x=results_df.index, y="Actual Temperature", label="Actual Temperature", color="blue")
sns.lineplot(data=results_df, x=results_df.index, y="Predicted Temperature", label="Predicted Temperature", color="red", linestyle="dashed")
plt.xlabel("Data Points")
plt.ylabel("Temperature")
plt.title("Actual vs Predicted Temperature")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Disaster recommendation system
def recommend_disaster(temperature, rainfall):
    recommendations = []

    temp = round(temperature, 1)
    rain = round(rainfall, 1)

    if temp >= 38:
        recommendations.append("🌡️ Extreme Heat Warning! Stay indoors and drink water regularly.")
    elif temp <= 4:
        recommendations.append("❄️ Freezing Alert! Wear layers and avoid staying out long.")

    if rain >= 60:
        recommendations.append("🌊 Heavy Rainfall Alert! Flooding likely. Avoid low-lying areas.")
    elif rain >= 20:
        recommendations.append("🌧️ Moderate Rainfall expected. Roads may be slippery.")

    if not recommendations:
        recommendations.append("✅ Weather looks safe. No extreme conditions today!")

    return recommendations


# Flask Web Application Setup
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get input values
        pressure = float(request.form["pressure"])
        maxtemp = float(request.form["maxtemp"])
        temperature = float(request.form["temperature"])
        mintemp = float(request.form["mintemp"])
        dewpoint = float(request.form["dewpoint"])
        humidity = float(request.form["humidity"])
        cloud = float(request.form["cloud"])
        rainfall = float(request.form["rainfall"])
        sunshine = float(request.form["sunshine"])
        
        # Create input array
        new_data = np.array([[pressure, maxtemp, temperature, mintemp, dewpoint, humidity, cloud, rainfall, sunshine]])
        
        # Debug: Print input data before scaling
        print("Input Data (Before Scaling):", new_data)
        
        # Scale input data
        new_data_scaled = scaler.transform(new_data)
        
        # Debug: Print input data after scaling
        print("Input Data (After Scaling):", new_data_scaled)
        
        # Predict temperature and rainfall
        predicted_temp = temp_model.predict(new_data_scaled)[0]
        predicted_rain = rain_model.predict(new_data_scaled)[0]
        
        # Debug: Print predictions
        print("Predicted Temperature:", predicted_temp)
        print("Predicted Rainfall:", predicted_rain)
        
        # Get disaster recommendations
        recommendations = recommend_disaster(predicted_temp, predicted_rain)
        
        return render_template("index.html", 
                              temperature=predicted_temp, 
                              rain="Yes" if predicted_rain == 1 else "No", 
                              recommendations=recommendations)
    
    return render_template("index.html")

# Running Flask in a Jupyter Notebook with an option to visit the website
def run_flask():
    print("Visit http://localhost:5000 in your browser to access the web application.")
    run_simple('localhost', 5000, app, use_reloader=False)

t = threading.Thread(target=run_flask)
t.start()

@app.route("/history")
def show_history():
    try:
        with open("predictions.csv", newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
    except FileNotFoundError:
        data = [["No data available"]]
    return render_template("history.html", data=data)
