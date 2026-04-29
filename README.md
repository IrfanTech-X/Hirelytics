
# Hirelytics 

Hirelytics is an AI-based recruitment system that automatically analyzes resumes against a job description using **NLP, Machine Learning, and Semantic Similarity models**. It ranks candidates based on ATS score, skill matching, and ML-based classification.

---

## 🚀 Features

- 📄 Upload Job Description & Multiple Resumes
- 🧠 NLP-based text preprocessing (cleaning, tokenization, lemmatization)
- 🔍 Skill extraction using keyword-based NLP
- 🤖 Semantic similarity using Sentence Transformers
- 📊 ATS scoring system (skill match + similarity)
- 🧮 Machine Learning model (Random Forest classifier)
- 📈 Interactive analytics dashboard (Charts + Tables)
- 📥 Download ranked CSV report
- 🏆 Smart candidate ranking system

---

## 🧠 Technologies Used

### Backend
- Python
- Flask

### NLP & AI
- NLTK
- SentenceTransformers (`all-MiniLM-L6-v2`)
- TF-IDF Vectorizer

### Machine Learning
- Scikit-learn (RandomForestClassifier)

### Data Processing
- Pandas
- NumPy

### Visualization
- Chart.js
- Bootstrap 5

---

## 📦 Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/hirelytics.git
cd hirelytics

```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt