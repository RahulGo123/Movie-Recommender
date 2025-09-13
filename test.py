import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load trained model
movie_model = joblib.load("final_model/model.pkl")

# Load test data
test_df = pd.read_csv("Artifacts/09092025__201917/data_ingestion/ingested/test.csv")

# Split features and target
TARGET_COLUMN = "rating"   # or "rating" depending on what you set as target
X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]

# Predict
y_pred = movie_model.predict(X_test)

# Accuracy
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Detailed metrics
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))
