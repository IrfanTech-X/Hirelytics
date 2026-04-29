from flask import Flask, render_template, request, send_file

from utils.text_processor import clean_text
from models.embedding_model import get_embedding
from models.skill_extractor import extract_skills, compare_skills

import os
import pandas as pd
import numpy as np
import uuid
import nltk
import joblib

# -------------------- NLTK --------------------

nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- FLASK --------------------

app = Flask(__name__)

# -------------------- PATHS --------------------

UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/outputs"

MODEL_PATH = "models/resume_classifier.pkl"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- LOAD ML MODEL --------------------

ml_model = joblib.load(MODEL_PATH)

# -------------------- UTIL FUNCTIONS --------------------

def cosine_similarity(vec1, vec2):

    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


def compute_ats_score(similarity, matched_skills_count, total_skills):

    skill_score = (
        (matched_skills_count / total_skills) * 100
        if total_skills > 0 else 0
    )

    return round((similarity * 0.7 + skill_score * 0.3), 2)

# -------------------- ROUTES --------------------

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

    job_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        job_filename
    )

    job_file.save(job_path)

    # -------------------- NLP PROCESSING --------------------

    job_text = clean_text(job_path)

    job_emb = get_embedding(job_text)

    #  FIXED: CLEAN SKILLS FROM JD
    job_skills_raw = extract_skills(job_text)
    job_skills = list(set([s.strip().lower() for s in job_skills_raw]))

    total_skills = len(job_skills)

    results = []

    # -------------------- PROCESS RESUMES --------------------

    for resume in resumes:

        resume_filename = f"{uuid.uuid4()}_{resume.filename}"

        resume_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            resume_filename
        )

        resume.save(resume_path)

        # -------------------- CLEAN TEXT --------------------

        resume_text = clean_text(resume_path)

        # -------------------- EMBEDDING SIMILARITY --------------------

        resume_emb = get_embedding(resume_text)

        similarity = cosine_similarity(
            job_emb,
            resume_emb
        ) * 100

        # -------------------- SKILL EXTRACTION (FIXED) --------------------

        resume_skills_raw = extract_skills(resume_text)
        resume_skills = list(set([s.strip().lower() for s in resume_skills_raw]))

        matched_skills, missing_skills, skill_ratio = compare_skills(
            job_skills,
            resume_skills
        )

        # -------------------- ATS SCORE --------------------

        ats_score = compute_ats_score(
            similarity,
            len(matched_skills),
            total_skills
        )

        ats_friendly = (
            "Yes" if ats_score >= 60 else "No"
        )

        # -------------------- ML PREDICTION --------------------

        predicted_category = ml_model.predict([
            resume_text
        ])[0]

        # -------------------- SUITABILITY --------------------

        if similarity >= 80:
            suitability = "Highly Suitable"

        elif similarity >= 60:
            suitability = "Moderately Suitable"

        else:
            suitability = "Low Fit"

        # -------------------- STORE RESULTS --------------------

        results.append({

            "name": resume.filename,

            "similarity": round(similarity, 2),

            "skills": resume_skills,

            "matched_skills": matched_skills,

            "missing_skills": missing_skills,

            "ats_score": ats_score,

            "ats_friendly": ats_friendly,

            "predicted_category": predicted_category,

            "suitability": suitability
        })

        # DELETE RESUME

        os.remove(resume_path)

    # DELETE JOB DESCRIPTION

    os.remove(job_path)

    # -------------------- SAVE CSV --------------------

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "ranked_candidates.csv"
    )

    df = pd.DataFrame(results)

    df.to_csv(output_path, index=False)

    # -------------------- SORT RESULTS --------------------

    results_sorted = sorted(
        results,
        key=lambda x: (
            x['ats_score'],
            x['similarity']
        ),
        reverse=True
    )

    # -------------------- RENDER DASHBOARD --------------------

    return render_template(
        "results.html",
        results=results_sorted,
        job_skills=job_skills
    )


@app.route("/download")
def download_csv():

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "ranked_candidates.csv"
    )

    return send_file(
        output_path,
        as_attachment=True
    )

# -------------------- RUN APP --------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )