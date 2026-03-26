import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load Data
df = pd.read_csv('f1_dnf.csv')

# Compute age_at_race
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['age_at_race'] = (df['date'] - df['dob']).dt.days / 365.25

# Clean numeric columns to match app.py
df["lat"] = pd.to_numeric(df["lat"], errors="coerce").fillna(0.0)
df["lng"] = pd.to_numeric(df["lng"], errors="coerce").fillna(0.0)
df["alt"] = pd.to_numeric(df["alt"], errors="coerce").fillna(0.0)
df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
df["laps"] = pd.to_numeric(df["laps"], errors="coerce").fillna(0.0)

# Define features and target
features = ['year', 'round', 'grid', 'positionOrder', 'points', 'laps', 'circuitId', 'lat', 'lng', 'alt', 'age_at_race']
df = df.dropna(subset=features + ['target_finish'])

X = df[features]
y = df['target_finish']

# Load model
model = pickle.load(open('classifier.pkl', 'rb'))

# Predict and Evaluate
y_pred = model.predict(X)

print("\nAccuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=["FINISH", "DNF"]))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
