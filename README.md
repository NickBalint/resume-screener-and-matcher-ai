# AI Resume Screener + Job Matcher

A simple machine learning group project that analyzes resume text, predicts matching job roles, detects skills, and gives improvement suggestions.

## Features

- Resume upload or paste input (TXT, CSV, PDF)
- OCR fallback for scanned/image-based PDF resumes
- OCR language selection for scanned PDFs (multi-language support)
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

## Chrome Extension (Realistic Apply Assistant)

This repo now includes a Chrome extension in `extension/` for the realistic workflow:

- Analyze this job on LinkedIn, Indeed, and company career pages
- Generate tailored resume bullets and a cover letter draft per posting
- Autofill common application fields using your saved profile

Extraction presets are included for:

- LinkedIn
- Indeed
- Greenhouse
- Lever

If a page structure does not match a known preset, the extension falls back to a generic extractor.

### Load the Extension in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder from this repo

### Set Up Your Candidate Profile

1. Click the extension icon
2. Open **Edit Candidate Profile**
3. Save your name, email, phone, links, and resume master text

### Use on Job Pages

1. Open a job posting on LinkedIn, Indeed, or a company careers page
2. Click the floating **Analyze this job** button (bottom-right) or use the popup
3. Review top role match, tailored bullets, and generated cover letter
4. Click **Autofill Form** to fill common fields (name, email, phone, links, summary)

### Notes

- This is an assisted workflow: you still review and submit manually.
- Autofill reliability varies by ATS/site markup.

### OCR Setup for Scanned PDFs

For image-only PDFs, install Tesseract OCR on your machine:

```bash
brew install tesseract
```

If you want OCR beyond English (for example Spanish or French), install extra language packs:

```bash
brew install tesseract-lang
```

Then pick one or more OCR languages in the app sidebar.

## Good Group Role Split

- Frontend/UI: Streamlit layout and design
- ML Engineer: matching model and scoring logic
- Data Engineer: job-role dataset and skills list
- Backend/Integration: file upload, PDF support, deployment
- Presenter/Tester: demo script, test resumes, slides

## Future Improvements

- Add live job listings
- Use BERT sentence embeddings
- Add login and saved resume history
- Add dashboards and charts
- Export feedback as PDF
