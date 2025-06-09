import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "heart_model.pkl")
print("✅ Model saved as heart_model.pkl")
