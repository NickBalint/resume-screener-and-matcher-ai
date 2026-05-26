import re
import shutil
import math
from io import BytesIO
from typing import Dict, List, Tuple, cast

import fitz
import pandas as pd
import pytesseract
import requests
import streamlit as st
from PIL import Image
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

OCR_LANGUAGE_OPTIONS: Dict[str, str] = {
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Portuguese": "por",
    "Dutch": "nld",
}

REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"
REMOTEOK_API_URL = "https://remoteok.com/api"
HN_ALGOLIA_JOBS_API_URL = "https://hn.algolia.com/api/v1/search_by_date"


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


@st.cache_resource(show_spinner=False)
def get_role_classifier() -> Tuple[TfidfVectorizer, LogisticRegression]:
    train_texts: List[str] = []
    train_labels: List[str] = []

    for role, config in JOB_ROLES.items():
        description = str(config["description"])
        skills = [str(skill) for skill in cast(List[str], config["skills"])]

        templates = [
            description,
            f"{role} role focused on {', '.join(skills)}.",
            f"Experience with {', '.join(skills[:5])} and related projects.",
            f"Built solutions using {', '.join(skills)}.",
            f"Hands-on {role.lower()} background with {', '.join(skills[:6])}.",
        ]

        for text in templates:
            train_texts.append(text)
            train_labels.append(role)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    features = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(features, train_labels)
    return vectorizer, classifier


def score_resume(resume_text: str) -> pd.DataFrame:
    role_names = list(JOB_ROLES.keys())
    role_docs = [JOB_ROLES[role]["description"] for role in role_names]
    docs = [resume_text] + role_docs

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    similarities = cosine_similarity(matrix.getrow(0), matrix).flatten()[1:]

    ml_prob_by_role: Dict[str, float] = {role: 0.0 for role in role_names}
    try:
        clf_vectorizer, classifier = get_role_classifier()
        probs = classifier.predict_proba(clf_vectorizer.transform([resume_text]))[0]
        ml_prob_by_role = {role: float(prob) for role, prob in zip(classifier.classes_, probs)}
    except Exception:
        # If the classifier cannot run for any reason, keep deterministic fallback scores.
        pass

    rows = []
    for role, similarity in zip(role_names, similarities):
        required_skills = cast(List[str], JOB_ROLES[role]["skills"])
        matched_skills = extract_skills(resume_text, required_skills)
        missing_skills = [s for s in required_skills if s not in matched_skills]
        skill_score = len(matched_skills) / len(required_skills)
        ml_confidence = ml_prob_by_role.get(role, 0.0)
        final_score = round((0.5 * similarity + 0.25 * skill_score + 0.25 * ml_confidence) * 100, 1)
        rows.append({
            "Job Role": role,
            "Match Score": final_score,
            "ML Confidence": round(ml_confidence * 100, 1),
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

    role_skills = cast(List[str], JOB_ROLES[best_role]["skills"])
    missing = [s for s in role_skills if s not in extract_skills(resume_text, role_skills)]
    if missing:
        feedback.append(f"For {best_role}, consider adding or learning: {', '.join(missing[:5])}.")

    if len(resume_text.split()) < 120:
        feedback.append("Your resume text looks short. Add project details, tools used, and outcomes achieved.")

    if not feedback:
        feedback.append("Strong resume match. Improve it further by tailoring the top skills and achievements to the job posting.")

    return feedback


def extract_pdf_text(pdf_bytes: bytes, ocr_lang_codes: List[str]) -> Tuple[str, str]:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        direct_text = "\n".join(pages_text).strip()
        if direct_text:
            return direct_text, ""
    except Exception:
        pass

    if shutil.which("tesseract") is None:
        return "", "No selectable text was found in this PDF. OCR fallback requires Tesseract (macOS: brew install tesseract)."

    try:
        selected_langs = sorted(set(ocr_lang_codes))
        lang_arg = "+".join(selected_langs) if selected_langs else "eng"
        ocr_pages = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page in document:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                ocr_pages.append(pytesseract.image_to_string(image, lang=lang_arg))

        ocr_text = "\n".join(ocr_pages).strip()
        if ocr_text:
            return ocr_text, "No selectable text was found, so OCR was used to read this PDF."
        return "", "No selectable text was found and OCR could not detect readable text."
    except Exception as exc:
        message = str(exc)
        if "Failed loading language" in message or "Error opening data file" in message:
            return "", "OCR language data is missing for one or more selected languages. Install additional packs with: brew install tesseract-lang"
        return "", "No selectable text was found and OCR failed while processing this PDF."


def read_uploaded_file(uploaded_file, ocr_lang_codes: List[str]) -> Tuple[str, str]:
    if uploaded_file is None:
        return "", ""
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8", errors="ignore"), ""
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        return " ".join(df.astype(str).fillna("").values.flatten()), ""
    if uploaded_file.type == "application/pdf":
        return extract_pdf_text(uploaded_file.read(), ocr_lang_codes)
    return uploaded_file.read().decode("utf-8", errors="ignore"), ""


def strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def fetch_remotive_jobs(search_query: str, limit: int = 20) -> Tuple[List[Dict[str, str]], str]:
    try:
        response = requests.get(
            REMOTIVE_API_URL,
            params={"search": search_query},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return [], "Unable to fetch live jobs right now from Remotive API. Please try again in a moment."

    jobs = payload.get("jobs", []) if isinstance(payload, dict) else []
    normalized_jobs: List[Dict[str, str]] = []
    for job in jobs[:limit]:
        normalized_jobs.append(
            {
                "title": str(job.get("title", "")).strip(),
                "company": str(job.get("company_name", "")).strip(),
                "location": str(job.get("candidate_required_location", "Remote")).strip(),
                "published": str(job.get("publication_date", "")).strip(),
                "apply_url": str(job.get("url", "")).strip(),
                "description": strip_html(str(job.get("description", ""))),
                "source": "Remotive",
            }
        )

    if not normalized_jobs:
        return [], "No live jobs were found for that role right now."
    return normalized_jobs, ""


def fetch_arbeitnow_jobs(search_query: str, limit: int = 20) -> Tuple[List[Dict[str, str]], str]:
    try:
        response = requests.get(ARBEITNOW_API_URL, timeout=15)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return [], ""

    jobs = payload.get("data", []) if isinstance(payload, dict) else []
    query_tokens = [token for token in clean_text(search_query).split() if token]

    normalized_jobs: List[Dict[str, str]] = []
    for job in jobs:
        title = str(job.get("title", "")).strip()
        description = strip_html(str(job.get("description", "")))
        searchable = clean_text(f"{title} {description}")
        if query_tokens and not any(token in searchable for token in query_tokens):
            continue

        normalized_jobs.append(
            {
                "title": title,
                "company": str(job.get("company_name", "")).strip(),
                "location": str(job.get("location", "Remote")).strip(),
                "published": str(job.get("created_at", "")).strip(),
                "apply_url": str(job.get("url", "")).strip(),
                "description": description,
                "source": "Arbeitnow",
            }
        )

        if len(normalized_jobs) >= limit:
            break

    return normalized_jobs, ""


def fetch_jobs_from_public_apis(search_query: str, limit: int = 20) -> Tuple[List[Dict[str, str]], str]:
    remotive_jobs, remotive_error = fetch_remotive_jobs(search_query, limit=limit)
    arbeitnow_jobs, _ = fetch_arbeitnow_jobs(search_query, limit=limit)
    remoteok_jobs, _ = fetch_remoteok_jobs(search_query, limit=limit)
    hackernews_jobs, _ = fetch_hackernews_jobs(search_query, limit=limit)

    combined = remotive_jobs + arbeitnow_jobs + remoteok_jobs + hackernews_jobs
    deduped: List[Dict[str, str]] = []
    seen_urls = set()
    for job in combined:
        apply_url = job.get("apply_url", "")
        if not apply_url or apply_url in seen_urls:
            continue
        seen_urls.add(apply_url)
        deduped.append(job)
        if len(deduped) >= limit:
            break

    if deduped:
        return deduped, ""
    if remotive_error:
        return [], remotive_error
    return [], "No live jobs were found from currently available public APIs."


def fetch_remoteok_jobs(search_query: str, limit: int = 20) -> Tuple[List[Dict[str, str]], str]:
    try:
        response = requests.get(
            REMOTEOK_API_URL,
            headers={"User-Agent": "resume-screener-ai/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return [], ""

    if not isinstance(payload, list):
        return [], ""

    query_tokens = [token for token in clean_text(search_query).split() if token]
    normalized_jobs: List[Dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        title = str(item.get("position", "")).strip()
        company = str(item.get("company", "")).strip()
        description = strip_html(str(item.get("description", "")))
        searchable = clean_text(f"{title} {description}")
        if query_tokens and not any(token in searchable for token in query_tokens):
            continue

        apply_url = str(item.get("url", "")).strip()
        if not apply_url:
            continue

        normalized_jobs.append(
            {
                "title": title,
                "company": company,
                "location": str(item.get("location", "Remote")).strip() or "Remote",
                "published": str(item.get("date", "")).strip(),
                "apply_url": apply_url,
                "description": description,
                "source": "RemoteOK",
            }
        )

        if len(normalized_jobs) >= limit:
            break

    return normalized_jobs, ""


def fetch_hackernews_jobs(search_query: str, limit: int = 20) -> Tuple[List[Dict[str, str]], str]:
    try:
        response = requests.get(
            HN_ALGOLIA_JOBS_API_URL,
            params={"tags": "job,story", "query": search_query, "hitsPerPage": str(limit * 2)},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return [], ""

    hits = payload.get("hits", []) if isinstance(payload, dict) else []
    normalized_jobs: List[Dict[str, str]] = []
    for hit in hits:
        if not isinstance(hit, dict):
            continue

        title = str(hit.get("title") or hit.get("story_title") or "").strip()
        if not title:
            continue

        apply_url = str(hit.get("url") or hit.get("story_url") or "").strip()
        if not apply_url:
            hn_id = str(hit.get("objectID", "")).strip()
            if hn_id:
                apply_url = f"https://news.ycombinator.com/item?id={hn_id}"
            else:
                continue

        created_at = str(hit.get("created_at", "")).strip()
        normalized_jobs.append(
            {
                "title": title,
                "company": "Hacker News Post",
                "location": "Remote/Unknown",
                "published": created_at,
                "apply_url": apply_url,
                "description": title,
                "source": "Hacker News",
            }
        )

        if len(normalized_jobs) >= limit:
            break

    return normalized_jobs, ""


def rank_live_jobs(resume_text: str, jobs: List[Dict[str, str]], target_role: str = "") -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame()

    all_skills = sorted({skill for role in JOB_ROLES.values() for skill in cast(List[str], role["skills"])})
    resume_skills = set(extract_skills(resume_text, all_skills))

    role_docs = []
    for job in jobs:
        description = job.get("description", "")
        title = job.get("title", "")
        role_docs.append(clean_text(f"{description} {title}"))

    docs = [clean_text(resume_text)] + role_docs
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(docs)
        similarities = cosine_similarity(matrix.getrow(0), matrix).flatten()[1:]
    except ValueError:
        # Fallback when all docs collapse to empty vocabulary after cleaning.
        similarities = [0.0] * len(jobs)

    resume_role_probs: Dict[str, float] = {role: 0.0 for role in JOB_ROLES.keys()}
    job_role_probs_list: List[Dict[str, float]] = [{role: 0.0 for role in JOB_ROLES.keys()} for _ in jobs]
    try:
        clf_vectorizer, classifier = get_role_classifier()
        resume_probs = classifier.predict_proba(clf_vectorizer.transform([resume_text]))[0]
        resume_role_probs = {role: float(prob) for role, prob in zip(classifier.classes_, resume_probs)}

        job_input_texts = [f"{job.get('title', '')} {job.get('description', '')}" for job in jobs]
        job_prob_vectors = classifier.predict_proba(clf_vectorizer.transform(job_input_texts))
        job_role_probs_list = [
            {role: float(prob) for role, prob in zip(classifier.classes_, probs)}
            for probs in job_prob_vectors
        ]
    except Exception:
        # Keep ranking deterministic even if classifier inference fails.
        pass

    rows = []
    for job, similarity, job_role_probs in zip(jobs, similarities, job_role_probs_list):
        job_skill_targets = set(extract_skills(job.get("description", ""), all_skills))
        if job_skill_targets:
            matched = sorted(resume_skills.intersection(job_skill_targets))
            skill_score = len(matched) / len(job_skill_targets)
            matched_preview = ", ".join(matched[:6]) if matched else "None"
        else:
            skill_score = 0.0
            matched_preview = "No common tracked skills found"

        target_role_confidence = job_role_probs.get(target_role, 0.0) if target_role else 0.0
        resume_vector = [resume_role_probs.get(role, 0.0) for role in JOB_ROLES.keys()]
        job_vector = [job_role_probs.get(role, 0.0) for role in JOB_ROLES.keys()]
        numerator = sum(a * b for a, b in zip(resume_vector, job_vector))
        denominator = math.sqrt(sum(a * a for a in resume_vector)) * math.sqrt(sum(b * b for b in job_vector))
        role_alignment = (numerator / denominator) if denominator else 0.0

        final_score = round(
            (0.55 * similarity + 0.2 * skill_score + 0.15 * target_role_confidence + 0.1 * role_alignment) * 100,
            1,
        )
        rows.append(
            {
                "Match Score": final_score,
                "Job Title": job.get("title", "Unknown Role"),
                "Company": job.get("company", "Unknown Company"),
                "Location": job.get("location", "Remote"),
                "Role Fit": round(target_role_confidence * 100, 1),
                "Matched Skills": matched_preview,
                "Source": job.get("source", "API"),
                "Apply Link": job.get("apply_url", ""),
            }
        )

    return pd.DataFrame(rows).sort_values("Match Score", ascending=False).reset_index(drop=True)


def source_counts_text(jobs: List[Dict[str, str]]) -> str:
    if not jobs:
        return ""
    counts: Dict[str, int] = {}
    for job in jobs:
        source = str(job.get("source", "Unknown")).strip() or "Unknown"
        counts[source] = counts.get(source, 0) + 1
    parts = [f"{source}: {count}" for source, count in sorted(counts.items())]
    return " | ".join(parts)


st.set_page_config(page_title="AI Resume Screener + Job Matcher", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --accent-blue: #2563eb;
        --accent-blue-dark: #1d4ed8;
        --accent-blue-soft: #dbeafe;
        --accent-blue-border: #93c5fd;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent-blue) !important;
        border-color: var(--accent-blue) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-blue-dark) !important;
        border-color: var(--accent-blue-dark) !important;
    }

    /* Blue style for OCR language selector */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        border-color: var(--accent-blue-border);
    }

    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] {
        background-color: var(--accent-blue-soft) !important;
        color: #1e3a8a !important;
        border-color: var(--accent-blue-border) !important;
    }

    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="tag"] * {
        color: #1e3a8a !important;
        fill: #1e3a8a !important;
    }

    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 1px var(--accent-blue) !important;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] input {
        caret-color: var(--accent-blue);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 AI Resume Screener + Job Matcher")
st.write("Paste or upload resume text to predict suitable job roles and get improvement suggestions.")

with st.sidebar:
    st.header("Project Info")
    st.write("ML used: TF-IDF + cosine similarity + Logistic Regression role classifier + rule-based skill extraction.")
    st.write("Group extension ideas: PDF parsing, live job listings, user accounts, BERT embeddings, dashboards.")
    selected_ocr_languages = st.multiselect(
        "OCR languages for scanned PDFs",
        options=list(OCR_LANGUAGE_OPTIONS.keys()),
        default=["English"],
        help="Use one or more languages when OCR is needed for image-based PDFs.",
    )

ocr_lang_codes = [OCR_LANGUAGE_OPTIONS[name] for name in selected_ocr_languages]

uploaded_file = st.file_uploader("Upload a resume file (TXT, CSV, or PDF)", type=["txt", "csv", "pdf"])

text_from_file, upload_notice = read_uploaded_file(uploaded_file, ocr_lang_codes)

if upload_notice:
    st.info(upload_notice)

resume_text = st.text_area(
    "Resume text",
    value=text_from_file,
    height=280,
    placeholder="Paste your resume here..."
)

if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False

if st.button("Analyze Resume", type="primary"):
    if not resume_text.strip():
        st.warning("Please paste or upload resume text first.")
        st.session_state["analysis_ready"] = False
    else:
        results = score_resume(resume_text)
        best_role = results.iloc[0]["Job Role"]
        suggestions = resume_feedback(resume_text, best_role)
        all_skills = [skill for role in JOB_ROLES.values() for skill in cast(List[str], role["skills"])]
        detected_skills = extract_skills(resume_text, all_skills)

        with st.spinner("Fetching current job listings and ranking best applications..."):
            live_jobs, jobs_error = fetch_jobs_from_public_apis(best_role, limit=25)

        ranked_jobs = pd.DataFrame()
        if not jobs_error:
            ranked_jobs = rank_live_jobs(resume_text, live_jobs, target_role=best_role)

        st.session_state["analysis_ready"] = True
        st.session_state["analysis_results"] = results
        st.session_state["analysis_best_role"] = best_role
        st.session_state["analysis_suggestions"] = suggestions
        st.session_state["analysis_detected_skills"] = detected_skills
        st.session_state["analysis_live_jobs"] = live_jobs
        st.session_state["analysis_jobs_error"] = jobs_error
        st.session_state["analysis_ranked_jobs"] = ranked_jobs

if st.session_state.get("analysis_ready", False):
    results = st.session_state["analysis_results"]
    best_role = st.session_state["analysis_best_role"]
    suggestions = st.session_state["analysis_suggestions"]
    detected_skills = st.session_state["analysis_detected_skills"]
    live_jobs = st.session_state["analysis_live_jobs"]
    jobs_error = st.session_state["analysis_jobs_error"]
    ranked_jobs = st.session_state["analysis_ranked_jobs"]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Best Match", best_role)
        st.metric("Top Score", f"{results.iloc[0]['Match Score']}%")
    with col2:
        st.subheader("Job Match Results")
        st.dataframe(results, width="stretch", hide_index=True)

    st.subheader("Improvement Suggestions")
    for item in suggestions:
        st.write(f"- {item}")

    st.subheader("Detected Skills Across All Roles")
    st.write(", ".join(detected_skills) if detected_skills else "No listed skills detected yet.")

    st.subheader("Live Job Recommendations")
    st.caption("Pulled from public APIs: Remotive, Arbeitnow, RemoteOK, and Hacker News Jobs")
    st.caption("Ranking uses role-aware ML (text similarity + skill overlap + role fit).")

    if jobs_error:
        st.info(jobs_error)
    else:
        st.caption(f"Sources loaded -> {source_counts_text(live_jobs)}")
        if ranked_jobs.empty:
            st.info("No ranked jobs available yet.")
        else:
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

            with filter_col1:
                location_options = ["All"] + sorted(ranked_jobs["Location"].dropna().astype(str).unique().tolist())
                selected_location = st.selectbox(
                    "Location",
                    options=location_options,
                    index=0,
                    key="live_jobs_location_filter",
                )

            with filter_col2:
                remote_only = st.checkbox(
                    "Remote only",
                    value=False,
                    key="live_jobs_remote_only",
                )

            with filter_col3:
                min_score = st.slider(
                    "Minimum score",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=1.0,
                    key="live_jobs_min_score",
                )

            with filter_col4:
                sort_option = st.selectbox(
                    "Sort by",
                    options=[
                        "Match Score (high to low)",
                        "Match Score (low to high)",
                        "Company (A-Z)",
                        "Location (A-Z)",
                        "Source (A-Z)",
                    ],
                    index=0,
                    key="live_jobs_sort",
                )

            filtered_jobs = ranked_jobs.copy()
            if selected_location != "All":
                filtered_jobs = filtered_jobs[filtered_jobs["Location"] == selected_location]

            if remote_only:
                filtered_jobs = filtered_jobs[
                    filtered_jobs["Location"].str.contains("remote", case=False, na=False)
                ]

            filtered_jobs = filtered_jobs[filtered_jobs["Match Score"] >= min_score]

            sort_map = {
                "Match Score (high to low)": ("Match Score", False),
                "Match Score (low to high)": ("Match Score", True),
                "Company (A-Z)": ("Company", True),
                "Location (A-Z)": ("Location", True),
                "Source (A-Z)": ("Source", True),
            }
            sort_col, ascending = sort_map[sort_option]
            filtered_jobs = filtered_jobs.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

            st.caption(f"Showing {len(filtered_jobs)} of {len(ranked_jobs)} jobs")

            if filtered_jobs.empty:
                st.info("No jobs match the selected filters. Try relaxing the filters.")
            else:
                st.dataframe(
                    filtered_jobs,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Apply Link": st.column_config.LinkColumn(
                            "Apply Link",
                            help="Open the direct job application page",
                            display_text="Apply",
                        ),
                    },
                )
