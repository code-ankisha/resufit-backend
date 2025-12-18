import os
import re
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from docx import Document

# ---------------- App Setup ----------------
app = Flask(__name__)
CORS(app, supports_credentials=True)

# HuggingFace API
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.environ.get("HF_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

# ---------------- File Readers ----------------
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


def read_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


# ---------------- HuggingFace Embedding ----------------
def get_embedding(text):
    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json={"inputs": text}
    )

    if response.status_code != 200:
        raise Exception("HuggingFace API error")

    emb = np.array(response.json())
    return emb.mean(axis=0)


def cosine(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------------- ATS Utilities ----------------
def extract_skills(text):
    SKILLS = [
        "python", "java", "javascript", "react", "node", "express", "flask",
        "mongodb", "mysql", "html", "css", "tailwind", "git", "github",
        "rest api", "jwt", "firebase", "data structures", "oops", "dbms"
    ]
    text = text.lower()
    return list({s for s in SKILLS if s in text})


def has_email(text):
    return bool(re.search(r"\S+@\S+\.\S+", text))


def has_phone(text):
    return bool(re.search(r"\+?\d[\d\s\-()]{8,}", text))


def has_address(text):
    return any(x in text.lower() for x in ["india", "road", "street", "sector", "city"])


# ---------------- API Route ----------------
@app.route("/match", methods=["POST"])
def match_resume():
    resume_text = ""
    jd = request.form.get("jd", "")

    file = request.files.get("file")
    if file:
        name = file.filename.lower()
        if name.endswith(".pdf"):
            resume_text = read_pdf(file)
        elif name.endswith(".docx"):
            resume_text = read_docx(file)
        else:
            resume_text = file.read().decode("utf-8")
    else:
        resume_text = request.form.get("resume", "")

    if not resume_text or not jd:
        return jsonify({"error": "Missing resume or JD"}), 400

    # ---- Embeddings via HuggingFace ----
    emb_r = get_embedding(resume_text)
    emb_j = get_embedding(jd)

    score = round(cosine(emb_r, emb_j) * 100)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd)
    missing_skills = list(set(jd_skills) - set(resume_skills))

    ats = {
        "contactInformation": [
            {
                "label": "Address",
                "status": "pass" if has_address(resume_text) else "fail",
                "message": "Address found." if has_address(resume_text) else "Address not found."
            },
            {
                "label": "Email",
                "status": "pass" if has_email(resume_text) else "fail",
                "message": "Email detected." if has_email(resume_text) else "Email missing."
            },
            {
                "label": "Phone",
                "status": "pass" if has_phone(resume_text) else "fail",
                "message": "Phone number detected." if has_phone(resume_text) else "Phone number missing."
            }
        ],
        "summary": {
            "status": "pass" if "summary" in resume_text.lower() else "fail",
            "message": "Summary found." if "summary" in resume_text.lower() else "Add a professional summary."
        },
        "education": {
            "status": "pass" if "bachelor" in resume_text.lower() else "fail",
            "message": "Education matches." if "bachelor" in resume_text.lower() else "Bachelor degree missing."
        },
        "fileType": {
            "status": "pass",
            "message": "ATS friendly resume format."
        }
    }

    suggestions = []
    if missing_skills:
        suggestions.append("Add missing skills from job description.")
    if not has_email(resume_text):
        suggestions.append("Add an email address.")
    if not has_phone(resume_text):
        suggestions.append("Add a phone number.")
    if "summary" not in resume_text.lower():
        suggestions.append("Add a professional summary.")

    return jsonify({
        "overall": score,
        "missingKeywords": missing_skills,
        "atsAnalysis": ats,
        "suggestions": suggestions
    })


# ---------------- Render Entry ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


