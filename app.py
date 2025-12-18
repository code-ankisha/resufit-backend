import os
import requests
import numpy as np
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from docx import Document

app = Flask(__name__)
CORS(app)

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"
}

# ---------- Utils ----------
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


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(text):
    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={"inputs": text}
    )
    response.raise_for_status()
    return response.json()


def extract_skills(text):
    skills = [
        "python", "java", "javascript", "react", "node", "flask",
        "mongodb", "mysql", "html", "css", "git", "firebase"
    ]
    return [s for s in skills if s in text.lower()]


# ---------- API ----------
@app.route("/match", methods=["POST"])
def match_resume():
    try:
        jd = request.form.get("jd", "")
        file = request.files.get("file")

        if not jd:
            return jsonify({"error": "Job description missing"}), 400

        resume_text = ""
        if file:
            if file.filename.endswith(".pdf"):
                resume_text = read_pdf(file)
            elif file.filename.endswith(".docx"):
                resume_text = read_docx(file)
            else:
                resume_text = file.read().decode("utf-8")

        if not resume_text:
            return jsonify({"error": "Resume missing"}), 400

        emb_r = get_embedding(resume_text)
        emb_j = get_embedding(jd)

        score = round(cosine(emb_r, emb_j) * 100)

        return jsonify({
            "overall": score,
            "missingKeywords": list(set(extract_skills(jd)) - set(extract_skills(resume_text))),
            "atsAnalysis": {},
            "suggestions": []
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/")
def home():
    return "ResuFit Backend is running 🚀"
