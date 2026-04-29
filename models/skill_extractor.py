import re

# Lightweight skill dictionary (ONLY for detection, NOT matching source)
SKILL_KEYWORDS = [
    "python", "java", "c", "c++", "machine learning",
    "deep learning", "nlp", "flask", "django",
    "mysql", "mongodb", "docker", "git",
    "javascript", "html", "css", "react", "ai", "ml",
    "data science", "aws", "azure", "linux"
]

def extract_skills(text):
    """
    Extract skills from ANY text (Job Description OR Resume)
    """

    if isinstance(text, (list, tuple)):
        text = " ".join(text)

    text = str(text).lower()

    found_skills = set()

    # 1. keyword-based extraction (MAIN METHOD)
    for skill in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found_skills.add(skill)

    return list(found_skills)


def compare_skills(job_skills, resume_skills):
    """
    Compare JOB DESCRIPTION vs RESUME skills
    """

    job_set = set([s.lower() for s in job_skills])
    resume_set = set([s.lower() for s in resume_skills])

    matched = list(job_set.intersection(resume_set))
    missing = list(job_set - resume_set)

    skill_ratio = (
        (len(matched) / len(job_set)) * 100
        if len(job_set) > 0 else 0
    )

    return matched, missing, skill_ratio