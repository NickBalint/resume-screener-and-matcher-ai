import re
from io import StringIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


JOB_ROLES: Dict[str, Dict[str, object]] = {
    "Data Analyst": {
        "description": "Analyze datasets, create dashboards, build reports, use SQL, Excel, Python, Tableau, Power BI, statistics, and data visualization.",
        "skills": ["python", "sql", "excel", "tableau", "power bi", "statistics", "data visualization", "pandas", "dashboard", "reporting"],
    },
    "Machine Learning Engineer": {
        "description": "Build, train, evaluate, and deploy machine learning models using Python, scikit-learn, TensorFlow, PyTorch, NLP, computer vision, APIs, and cloud tools.",
        "skills": ["python", "machine learning", "scikit-learn", "tensorflow", "pytorch", "nlp", "computer vision", "model deployment", "api", "cloud"],
    },
    "Software Engineer": {
        "description": "Design and develop applications using data structures, algorithms, Java, Python, JavaScript, React, backend APIs, databases, Git, testing, and system design.",
        "skills": ["java", "python", "javascript", "react", "api", "database", "git", "testing", "algorithms", "data structures"],
    },
    "Cybersecurity Analyst": {
        "description": "Monitor systems, detect threats, analyze vulnerabilities, use networking, Linux, SIEM, incident response, risk assessment, and security tools.",
        "skills": ["networking", "linux", "siem", "incident response", "vulnerability", "risk assessment", "security", "firewall", "python", "threat analysis"],
    },
    "Web Developer": {
        "description": "Build responsive websites and web apps using HTML, CSS, JavaScript, React, Node.js, APIs, databases, Git, UI design, and accessibility.",
        "skills": ["html", "css", "javascript", "react", "node", "api", "database", "git", "ui", "accessibility"],
    },
}

ACTION_VERBS = [
    "built", "created", "developed", "designed", "implemented", "analyzed", "automated",
    "improved", "optimized", "deployed", "trained", "evaluated", "managed", "led"
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(resume_text: str, all_skills: List[str]) -> List[str]:
    cleaned = clean_text(resume_text)
    found = []
    for skill in sorted(set(all_skills)):
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, cleaned):
            found.append(skill)
    return found


def score_resume(resume_text: str) -> pd.DataFrame:
    role_names = list(JOB_ROLES.keys())
    role_docs = [JOB_ROLES[role]["description"] for role in role_names]
    docs = [resume_text] + role_docs

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    rows = []
    for role, similarity in zip(role_names, similarities):
        required_skills = JOB_ROLES[role]["skills"]
        matched_skills = extract_skills(resume_text, required_skills)
        missing_skills = [s for s in required_skills if s not in matched_skills]
        skill_score = len(matched_skills) / len(required_skills)
        final_score = round((0.65 * similarity + 0.35 * skill_score) * 100, 1)
        rows.append({
            "Job Role": role,
            "Match Score": final_score,
            "Matched Skills": ", ".join(matched_skills) if matched_skills else "None found",
            "Missing Skills": ", ".join(missing_skills[:6]) if missing_skills else "None",
        })

    return pd.DataFrame(rows).sort_values("Match Score", ascending=False)


def resume_feedback(resume_text: str, best_role: str) -> List[str]:
    cleaned = clean_text(resume_text)
    feedback = []

    has_numbers = bool(re.search(r"\d+%|\$\d+|\d+\+|\b\d{2,}\b", resume_text))
    if not has_numbers:
        feedback.append("Add measurable impact, such as percentages, dollar savings, users served, or time saved.")

    if not any(verb in cleaned for verb in ACTION_VERBS):
        feedback.append("Start bullet points with stronger action verbs like built, analyzed, automated, optimized, or deployed.")

    role_skills = JOB_ROLES[best_role]["skills"]
    missing = [s for s in role_skills if s not in extract_skills(resume_text, role_skills)]
    if missing:
        feedback.append(f"For {best_role}, consider adding or learning: {', '.join(missing[:5])}.")

    if len(resume_text.split()) < 120:
        feedback.append("Your resume text looks short. Add project details, tools used, and outcomes achieved.")

    if not feedback:
        feedback.append("Strong resume match. Improve it further by tailoring the top skills and achievements to the job posting.")

    return feedback


def read_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        return " ".join(df.astype(str).fillna("").values.flatten())
    return uploaded_file.read().decode("utf-8", errors="ignore")


st.set_page_config(page_title="AI Resume Screener + Job Matcher", page_icon="📄", layout="wide")
st.title("📄 AI Resume Screener + Job Matcher")
st.write("Paste or upload resume text to predict suitable job roles and get improvement suggestions.")

with st.sidebar:
    st.header("Project Info")
    st.write("ML used: TF-IDF vectorization + cosine similarity + rule-based skill extraction.")
    st.write("Group extension ideas: PDF parsing, live job listings, user accounts, BERT embeddings, dashboards.")

uploaded_file = st.file_uploader("Upload a resume text file or CSV", type=["txt", "csv"])
text_from_file = read_uploaded_file(uploaded_file)

resume_text = st.text_area(
    "Resume text",
    value=text_from_file,
    height=280,
    placeholder="Paste your resume here..."
)

if st.button("Analyze Resume", type="primary"):
    if not resume_text.strip():
        st.warning("Please paste or upload resume text first.")
    else:
        results = score_resume(resume_text)
        best_role = results.iloc[0]["Job Role"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Best Match", best_role)
            st.metric("Top Score", f"{results.iloc[0]['Match Score']}%")
        with col2:
            st.subheader("Job Match Results")
            st.dataframe(results, use_container_width=True, hide_index=True)

        st.subheader("Improvement Suggestions")
        for item in resume_feedback(resume_text, best_role):
            st.write(f"- {item}")

        st.subheader("Detected Skills Across All Roles")
        all_skills = [skill for role in JOB_ROLES.values() for skill in role["skills"]]
        detected = extract_skills(resume_text, all_skills)
        st.write(", ".join(detected) if detected else "No listed skills detected yet.")
