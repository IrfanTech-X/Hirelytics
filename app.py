from flask import Flask, render_template, request, send_file
from utils.text_processor import clean_text
from models.embedding_model import get_embedding
from models.skill_extractor import extract_skills

import os
import pandas as pd
import numpy as np
import uuid
import nltk

# 🔹 ML Imports
from sklearn.ensemble import RandomForestClassifier
import joblib

# -------------------- NLTK --------------------
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# -------------------- Paths --------------------
UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/outputs"
MODEL_PATH = "models/candidate_model.pkl"   # ML model save path

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Utils --------------------

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two embeddings"""
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def compute_ats_score(similarity, matched_skills_count, total_skills):
    """
    ATS score combines:
    - similarity (70%)
    - skill match ratio (30%)
    """
    skill_score = (matched_skills_count / total_skills) * 100 if total_skills > 0 else 0
    return round((similarity * 0.7 + skill_score * 0.3), 2)


# -------------------- ML MODEL --------------------

def load_or_train_model():
    """
    Loads existing model OR trains a small demo model.
    This model predicts candidate suitability.
    """

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # 🔹 Dummy dataset (for demonstration)
    data = pd.DataFrame({
        "similarity": [90, 80, 70, 60, 50],
        "matched": [8, 7, 5, 4, 2],
        "missing": [1, 2, 3, 5, 6],
        "ats": [88, 78, 68, 58, 48],
        "label": ["Highly Suitable", "Highly Suitable", "Moderately Suitable", "Moderately Suitable", "Low Fit"]
    })

    X = data[["similarity", "matched", "missing", "ats"]]
    y = data["label"]

    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)

    return model


# 🔹 Load model once (global)
ml_model = load_or_train_model()


# -------------------- Routes --------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files():

    if "job_description" not in request.files:
        return "Job description file missing"

    job_file = request.files["job_description"]
    resumes = request.files.getlist("resumes")

    if job_file.filename == "" or len(resumes) == 0:
        return "Please upload job description and at least one resume"

    # -------------------- SAVE JOB DESCRIPTION --------------------
    job_filename = f"{uuid.uuid4()}_{job_file.filename}"
    job_path = os.path.join(app.config["UPLOAD_FOLDER"], job_filename)
    job_file.save(job_path)

    # 🔹 NLP processing
    job_text = clean_text(job_path)
    job_emb = get_embedding(job_text)

    # 🔹 Extract job skills (used as reference)
    job_skills = extract_skills(job_text)
    total_skills = len(job_skills)

    results = []

    # -------------------- PROCESS RESUMES --------------------
    for resume in resumes:

        resume_filename = f"{uuid.uuid4()}_{resume.filename}"
        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_filename)
        resume.save(resume_path)

        # 🔹 NLP + Embedding
        resume_text = clean_text(resume_path)
        resume_emb = get_embedding(resume_text)

        similarity = cosine_similarity(job_emb, resume_emb) * 100

        # 🔹 Extract resume skills
        skills = extract_skills(resume_text)

        # 🔹 Skill comparison
        matched_skills = [s for s in skills if s in job_skills]
        missing_skills = [s for s in job_skills if s not in skills]

        # 🔹 ATS score
        ats_score = compute_ats_score(similarity, len(matched_skills), total_skills)
        ats_friendly = "Yes" if ats_score >= 60 else "No"

        # -------------------- ML PREDICTION --------------------
        # Features for ML model
        features = [[
            similarity,
            len(matched_skills),
            len(missing_skills),
            ats_score
        ]]

        # 🔹 ML predicts final suitability
        ml_prediction = ml_model.predict(features)[0]

        # -------------------- RULE-BASED LABEL (kept for comparison) --------------------
        if similarity >= 80:
            rule_label = "Highly Suitable"
        elif similarity >= 60:
            rule_label = "Moderately Suitable"
        else:
            rule_label = "Low Fit"

        # -------------------- STORE RESULT --------------------
        results.append({
            "name": resume.filename,
            "similarity": round(similarity, 2),
            "skills": skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,

            # 🔹 BOTH labels (important for analysis)
            "suitability": rule_label,
            "ml_label": ml_prediction,

            "ats_score": ats_score,
            "ats_friendly": ats_friendly
        })

        # Delete resume file after processing
        os.remove(resume_path)

    # Delete job file
    os.remove(job_path)

    # -------------------- SAVE CSV --------------------
    output_path = os.path.join(OUTPUT_FOLDER, "ranked_candidates.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    # -------------------- SMART SORTING --------------------
    # Uses ATS + similarity (stronger ranking)
    results_sorted = sorted(
        results,
        key=lambda x: (x['ats_score'], x['similarity']),
        reverse=True
    )

    return render_template("results.html",
                           results=results_sorted,
                           job_skills=job_skills)


@app.route("/download")
def download_csv():
    output_path = os.path.join(OUTPUT_FOLDER, "ranked_candidates.csv")
    return send_file(output_path, as_attachment=True)


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)