from flask import Flask, render_template, request
from utils.text_processor import clean_text
from models.embedding_model import get_embedding
from models.skill_extractor import extract_skills
import os
import pandas as pd
import numpy as np
import uuid

app = Flask(__name__)

# -------------------- Paths --------------------
UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -------------------- Utils --------------------
def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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

    # Save job description
    job_filename = f"{uuid.uuid4()}_{job_file.filename}"
    job_path = os.path.join(app.config["UPLOAD_FOLDER"], job_filename)
    job_file.save(job_path)

    job_text = clean_text(job_path)
    job_emb = get_embedding(job_text)

    results = []

    # Process resumes
    for resume in resumes:
        resume_filename = f"{uuid.uuid4()}_{resume.filename}"
        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_filename)
        resume.save(resume_path)

        resume_text = clean_text(resume_path)
        resume_emb = get_embedding(resume_text)

        sim_score = cosine_similarity(job_emb, resume_emb)
        skills = extract_skills(resume_text)

        if sim_score >= 0.8:
            label = "Highly Suitable"
        elif sim_score >= 0.6:
            label = "Moderately Suitable"
        else:
            label = "Low Fit"

        results.append({
            "name": resume.filename,
            "similarity": round(sim_score * 100, 2),
            "skills": ", ".join(skills),
            "suitability": label
        })

    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_FOLDER, "ranked_candidates.csv")
    df.to_csv(output_path, index=False)

    return render_template("results.html", results=results)


# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True)
