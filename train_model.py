import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Example training data (replace with real landmark dataset)
X = [
    [0,0,0,0,0],  # fist
    [1,1,0,0,0],  # peace
    [1,1,1,1,1]   # open hand
]

y = [
    "Fist",
    "Peace",
    "Open Hand"
]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("models/gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")