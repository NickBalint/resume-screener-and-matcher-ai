# AI Resume Screener + Job Matcher

A simple machine learning group project that analyzes resume text, predicts matching job roles, detects skills, and gives improvement suggestions.

## Features

- Resume upload or paste input
- Job-role match scoring
- Skill extraction
- Missing skill recommendations
- Resume improvement feedback
- Streamlit web interface

## Machine Learning Used

- TF-IDF vectorization
- Cosine similarity
- Rule-based skill matching

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Good Group Role Split

- Frontend/UI: Streamlit layout and design
- ML Engineer: matching model and scoring logic
- Data Engineer: job-role dataset and skills list
- Backend/Integration: file upload, PDF support, deployment
- Presenter/Tester: demo script, test resumes, slides

## Future Improvements

- Add PDF resume parsing with PyPDF2 or pdfplumber
- Add live job listings
- Use BERT sentence embeddings
- Add login and saved resume history
- Add dashboards and charts
- Export feedback as PDF
