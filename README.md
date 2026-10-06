# Hirelytics

**AI-Powered Resume Screening & Candidate Ranking System**

Hirelytics is an AI-based recruitment system that analyzes resumes against a job description and automatically ranks candidates using **NLP, semantic similarity, skill matching, and machine learning**.

The system combines traditional NLP techniques with transformer-based sentence embeddings to move beyond simple keyword matching and provide a more meaningful candidate-job relevance score.

## Why Hirelytics?

Recruiters often need to review a large number of resumes for a single position. Hirelytics automates the initial screening process by:

* Extracting relevant information from resumes
* Identifying skills related to the job requirements
* Measuring semantic similarity between resumes and job descriptions
* Combining multiple signals into an ATS-style score
* Using machine learning for candidate classification
* Ranking candidates for faster initial screening

The goal is to demonstrate how **NLP and machine learning can be applied to a practical recruitment workflow.**

---

## Core Capabilities

### Resume & Job Description Analysis

* Upload a job description and multiple resumes
* Preprocess and normalize resume text
* Extract relevant skills and keywords
* Compare candidate profiles against job requirements

### NLP-Based Matching

* Text preprocessing using **NLTK**
* TF-IDF-based textual similarity
* Skill matching using keyword-based NLP
* Semantic similarity using **Sentence Transformers**

### AI-Based Candidate Ranking

Hirelytics combines multiple signals to generate an overall candidate score:

**ATS Score = Skill Matching + Semantic Similarity + ML-Based Classification**

This allows candidates to be ranked based on more than exact keyword overlap.

### Machine Learning

The system uses a **Random Forest classifier** to perform candidate classification based on extracted features.

### Analytics & Reporting

The dashboard provides:

* Candidate rankings
* ATS scores
* Skill matching information
* Similarity scores
* Interactive charts and tables
* Downloadable CSV reports

---

## AI / ML Pipeline

```text
Job Description
       |
       v
Text Preprocessing
       |
       v
Feature Extraction
   |           |
   |           +--> TF-IDF Features
   |
   +--------------> Skill Extraction
   |
   +--------------> Sentence Embeddings
                       |
                       v
              Semantic Similarity
                       |
                       v
              Random Forest Model
                       |
                       v
                Candidate Scoring
                       |
                       v
              Candidate Ranking
```

---

## Technology Stack

| Category         | Technologies                |
| ---------------- | --------------------------- |
| Backend          | Python, Flask               |
| NLP              | NLTK, TF-IDF                |
| Semantic AI      | Sentence Transformers       |
| Embedding Model  | `all-MiniLM-L6-v2`          |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing  | Pandas, NumPy               |
| Frontend         | HTML, Bootstrap 5           |
| Visualization    | Chart.js                    |

---

## Key Engineering Concepts Demonstrated

This project demonstrates practical experience with:

* Natural Language Processing
* Text preprocessing
* Feature engineering
* TF-IDF vectorization
* Transformer-based embeddings
* Semantic similarity
* Machine learning classification
* Candidate ranking
* Data processing with Pandas
* Flask-based AI application development
* AI-assisted decision support
* Data visualization and reporting

---

## Project Workflow

### 1. Job Description Input

The recruiter provides the job description containing the required skills, qualifications, and responsibilities.

### 2. Resume Processing

Uploaded resumes are converted into structured text and processed using NLP techniques such as cleaning, tokenization, and lemmatization.

### 3. Skill Extraction

Relevant skills are extracted from candidate resumes and compared against the requirements of the job description.

### 4. Semantic Matching

Sentence Transformers generate embeddings for the job description and resumes.

The system calculates semantic similarity to identify candidates whose experience is conceptually relevant, even when the wording differs.

### 5. Machine Learning Classification

Extracted features are passed to a Random Forest classifier to support candidate classification.

### 6. Candidate Ranking

The system combines the available matching signals to generate candidate scores and produces a ranked list.

### 7. Recruiter Dashboard

Recruiters can review candidate scores, matching information, analytics, and export the results as a CSV report.

---

## Example Use Case

Suppose a company is hiring for an **AI/ML Engineer** requiring:

```text
Python
Machine Learning
NLP
Scikit-learn
TensorFlow
```

Hirelytics can analyze multiple resumes and identify:

```text
Candidate A
High skill match
High semantic similarity
Strong ML classification
        ↓
Higher ranking

Candidate B
Moderate skill match
High semantic similarity
        ↓
Medium ranking

Candidate C
Low skill match
Low semantic similarity
        ↓
Lower ranking
```

This helps recruiters prioritize candidates for further human review.

---

## Project Structure

```text
hirelytics/
│
├── app.py
├── requirements.txt
│
├── models/
│   └── ...
│
├── templates/
│   └── ...
│
├── static/
│   └── ...
│
├── utils/
│   └── ...
│
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/hirelytics.git
cd hirelytics
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Then open the local URL shown in the terminal.

---

## Project Focus

Hirelytics was built to explore how **NLP, transformer-based semantic representations, and machine learning can be combined into an end-to-end AI application**.

Rather than relying solely on keyword matching, the system combines multiple approaches to provide a broader view of candidate-job relevance.

---

## Future Improvements

Potential extensions include:

* LLM-based resume analysis
* Explainable candidate scoring
* Bias and fairness evaluation
* Named Entity Recognition for structured resume extraction
* Experience and education extraction
* Vector database integration
* Retrieval-Augmented Generation (RAG)
* Human-in-the-loop candidate review
* Model evaluation and monitoring
* Production deployment with scalable inference

---

## Disclaimer

Hirelytics is an experimental AI recruitment tool designed to assist with initial candidate screening. Its rankings should support, not replace, human decision-making. Recruitment decisions should be made using appropriate human review and fairness considerations.

---

## Author

**Irfan Ferdous Siam**

B.Sc. in Computer Science & Engineering
Green University of Bangladesh

* GitHub: https://github.com/IrfanTech-X
* Portfolio: https://irfanferdous.netlify.app
* LinkedIn: https://linkedin.com/in/irfan-ferdous-siam
