import json
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_PATH = os.path.join(BASE_DIR, "skills.json")

with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    SKILLS = json.load(f)

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in SKILLS:
        if skill.lower() in text:
            found_skills.append(skill)

    return list(set(found_skills))
