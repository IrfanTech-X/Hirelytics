from flask import Flask, render_template, request, send_file
from utils.text_processor import clean_text
from models.embedding_model import get_embedding  # Uses global cache
from models.skill_extractor import extract_skills
import os
import pandas as pd
import numpy as np
import uuid
import nltk

# Download NLTK data (runs only if missing)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# -------------------- Paths --------------------
UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Utils --------------------
def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def compute_ats_score(similarity, matched_skills_count, total_skills):
    skill_score = (matched_skills_count / total_skills) * 100 if total_skills > 0 else 0
    return round((similarity * 0.7 + skill_score * 0.3), 2)

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

    # -------------------- Save job description --------------------
    job_filename = f"{uuid.uuid4()}_{job_file.filename}"
    job_path = os.path.join(app.config["UPLOAD_FOLDER"], job_filename)
    job_file.save(job_path)

    job_text = clean_text(job_path)
    job_emb = get_embedding(job_text)

    # Extract skills from job description
    job_skills = extract_skills(job_text)
    total_skills = len(job_skills)

    results = []

    # -------------------- Process resumes --------------------
    for resume in resumes:
        resume_filename = f"{uuid.uuid4()}_{resume.filename}"
        resume_path = os.path.join(app.config["UPLOAD_FOLDER"], resume_filename)
        resume.save(resume_path)

        resume_text = clean_text(resume_path)
        resume_emb = get_embedding(resume_text)

        similarity = cosine_similarity(job_emb, resume_emb) * 100  # %
        skills = extract_skills(resume_text)

        matched_skills = [s for s in skills if s in job_skills]
        missing_skills = [s for s in job_skills if s not in skills]

        ats_score = compute_ats_score(similarity, len(matched_skills), total_skills)
        ats_friendly = "Yes" if ats_score >= 60 else "No"

        if similarity >= 80:
            label = "Highly Suitable"
        elif similarity >= 60:
            label = "Moderately Suitable"
        else:
            label = "Low Fit"

        results.append({
            "name": resume.filename,
            "similarity": round(similarity,2),
            "skills": skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "suitability": label,
            "ats_score": ats_score,
            "ats_friendly": ats_friendly
        })

        os.remove(resume_path)

    os.remove(job_path)

    # Save results CSV
    output_path = os.path.join(OUTPUT_FOLDER, "ranked_candidates.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    # Sort results for display
    results_sorted = sorted(results, key=lambda x: (x['ats_score'], x['similarity']), reverse=True)

    return render_template("results.html", results=results_sorted, job_skills=job_skills)

@app.route("/download")
def download_csv():
    output_path = os.path.join(OUTPUT_FOLDER, "ranked_candidates.csv")
    return send_file(output_path, as_attachment=True)

# -------------------- Run App --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
