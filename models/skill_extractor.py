import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- Load NLTK --------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -------------------- Load Skills --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_PATH = os.path.join(BASE_DIR, "skills.json")

with open(SKILLS_PATH, "r", encoding="utf-8") as f:
    SKILLS_DB = json.load(f)

# -------------------- Text Preprocessing --------------------
def preprocess_text(text):
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    words = text.split()

    # Remove stopwords + lemmatize
    processed = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(processed)

# -------------------- Skill Extraction --------------------
def extract_skills(text):
    clean_text = preprocess_text(text)

    found_skills = []

    for skill, keywords in SKILLS_DB.items():
        for keyword in keywords:
            if keyword in clean_text:
                found_skills.append(skill)
                break  # avoid duplicate match

    return list(set(found_skills))