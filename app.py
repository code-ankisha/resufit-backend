from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import pdfplumber
from docx import Document


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
kw_model = KeyBERT(model="sentence-transformers/all-MiniLM-L6-v2")

app = Flask(__name__)
CORS(app)


def extract_text_from_pdf(file):
    """Extract text from PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def extract_text_from_docx(file):
    """Extract text from DOCX using python-docx."""
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.route("/match", methods=["POST"])
def match_resume():
    try:
        resume_text = ""
        jd = ""

       
        uploaded_file = request.files.get("file")
        jd = request.form.get("jd", "")

        if uploaded_file:
            filename = uploaded_file.filename.lower()

            if filename.endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)

            elif filename.endswith(".docx"):
                resume_text = extract_text_from_docx(uploaded_file)

            else:
                resume_text = uploaded_file.read().decode("utf-8")

        else:
          
            resume_text = request.form.get("resume", "")
            jd = request.form.get("jd", "")

        if not resume_text or not jd:
            return jsonify({"error": "Resume or JD missing"}), 400

        # ------------------ EMBEDDINGS ------------------
        emb_resume = embed_model.encode(resume_text)
        emb_jd = embed_model.encode(jd)

        similarity = cosine_similarity(emb_resume, emb_jd)
        score = round(similarity * 100)

        # ------------------ KEYWORDS ------------------
        jd_keywords = kw_model.extract_keywords(
            jd, keyphrase_ngram_range=(1, 2), top_n=10
        )
        keyword_list = [kw[0] for kw in jd_keywords]

        missing = [kw for kw in keyword_list if kw.lower() not in resume_text.lower()]

        breakdown = {
            "skills": max(40, 100 - len(missing) * 4),
            "experience": round(score * 0.9),
            "keywords": max(35, score - len(missing) * 3),
            "education": round(score * 0.8)
        }

        return jsonify({
            "overall": score,
            "suggestions": missing[:8],
            "breakdown": breakdown
        })

    except Exception as e:
        print("BACKEND ERROR:", e)
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(port=5000)
