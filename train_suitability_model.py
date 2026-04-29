import pandas as pd
import joblib
import ast

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ============================================
# LOAD CSV
# ============================================

df = pd.read_csv("dataset/ranked_candidates.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ============================================
# CONVERT STRING LISTS TO REAL LISTS
# ============================================

def count_list_items(x):

    try:
        return len(ast.literal_eval(x))

    except:
        return 0

# Count matched skills
df["matched_count"] = df["matched_skills"].apply(
    count_list_items
)

# Count missing skills
df["missing_count"] = df["missing_skills"].apply(
    count_list_items
)

# ============================================
# CREATE SKILL RATIO
# ============================================

df["skill_ratio"] = (
    df["matched_count"] /
    (
        df["matched_count"] +
        df["missing_count"] + 1
    )
)

# ============================================
# FEATURES
# ============================================

X = df[[
    "similarity",
    "ats_score",
    "matched_count",
    "missing_count",
    "skill_ratio"
]]

# ============================================
# TARGET LABEL
# ============================================

y = df["suitability"]

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# MODEL
# ============================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# ============================================
# TRAIN MODEL
# ============================================

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ============================================
# TEST MODEL
# ============================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# SAVE MODEL
# ============================================

joblib.dump(
    model,
    "models/suitability_model.pkl"
)

print("\nSuitability Model Saved Successfully")