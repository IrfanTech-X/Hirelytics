import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -------------------- LOAD DATASET --------------------

DATASET_PATH = "dataset/resume_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print("Dataset Loaded Successfully!")
print(df.head())

# -------------------- CHECK NULL VALUES --------------------

df = df.dropna()

# -------------------- FEATURES & LABELS --------------------

X = df["Resume"]
y = df["Category"]

# -------------------- TRAIN TEST SPLIT --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------- NLP + ML PIPELINE --------------------

model_pipeline = Pipeline([

    # NLP Feature Extraction
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )),

    # ML Model
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# -------------------- TRAIN MODEL --------------------

print("\nTraining Model...\n")

model_pipeline.fit(X_train, y_train)

# -------------------- EVALUATION --------------------

predictions = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# -------------------- SAVE MODEL --------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model_pipeline, "models/resume_classifier.pkl")

print("\nModel Saved Successfully!")
