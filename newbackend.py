import os
import re
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process

# Flask setup
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
VECTORS_FOLDER = "vectors"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTORS_FOLDER, exist_ok=True)

# Load fine-tuned SBERT model
model = SentenceTransformer("unsupervised_finetuned_sbert_new")

# Load LinkedIn skill list
with open("linkedin skill", "r", encoding="utf-8") as f:
    linkedin_skills = set(line.strip().lower() for line in f if line.strip())

# NLP model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc).replace('\xa0', ' ').lower()

def clean_text(text):
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_skills_from_text(text):
    section_headers = ['skills', 'technical skills', 'experience', 'technical expertise', 'work experience', 'projects']
    section_matches = {}

    for header in section_headers:
        pattern = rf"{header}[\s:]*\n?(.*?)(?=\n[A-Z][^\n]{{2,40}}\n|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            section_matches[header] = match.group(1).strip()

    combined_text = " ".join(section_matches.values())

    # Exact matches
    exact_matches = {skill for skill in linkedin_skills if re.search(rf"\b{re.escape(skill)}\b", combined_text)}

    # Fuzzy matches
    doc = nlp(combined_text)
    phrases = {
        chunk.text.lower().strip()
        for chunk in doc.noun_chunks
        if len(chunk.text.strip()) > 3 and ' ' in chunk.text
    }

    fuzzy_matches = set()
    for phrase in phrases:
        match, score, _ = process.extractOne(phrase, linkedin_skills, scorer=fuzz.token_sort_ratio)
        if score >= 92 and re.search(rf"\b{re.escape(match)}\b", phrase):
            fuzzy_matches.add(match)

    return sorted(exact_matches.union(fuzzy_matches))

# API route
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    if "resume" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["resume"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"status": "error", "message": "Only PDF files are allowed"}), 400

    filename = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # Extract and clean text
    text = extract_text_from_pdf(save_path)
    cleaned_text = clean_text(text)

    # Extract skills
    skills = extract_skills_from_text(cleaned_text)

    # Generate embedding
    skill_text = " ".join(skills)
    skill_vector = model.encode(skill_text)

    # Save to shared pickle
    vector_path = os.path.join(VECTORS_FOLDER, "all_resume_vectors.pkl")

    if os.path.exists(vector_path):
        with open(vector_path, "rb") as f:
            all_vectors = pickle.load(f)
    else:
        all_vectors = {}

    all_vectors[os.path.splitext(filename)[0]] = {
        "skills": skills,
        "vector": skill_vector.tolist()
    }

    with open(vector_path, "wb") as f:
        pickle.dump(all_vectors, f)

    return jsonify({
        "status": "success",
        "message": f"Skills extracted and vector stored for {filename}.",
        "skills": skills
    })

# Path to the stored resume vector data
VECTOR_PATH = "vectors/all_resume_vectors.pkl"

# Load resume vectors from file
def load_resume_vectors():
    if not os.path.exists(VECTOR_PATH):
        return {}
    with open(VECTOR_PATH, "rb") as f:
        return pickle.load(f)

# Function to rank resumes by similarity to job description
def rank_resumes(job_description):
    resume_data = load_resume_vectors()
    if not resume_data:
        return []

    jd_embedding = model.encode(job_description)

    ranked = []
    for resume_name, data in resume_data.items():
        vector = data["vector"]
        similarity = util.cos_sim(jd_embedding, [vector])[0][0].item()
        ranked.append({
            "resume": resume_name,
            "score": similarity,
            "skills": data["skills"]
        })

    # Sort by score descending
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

# API endpoint to rank resumes
@app.route("/submit", methods=["POST"])
def submit_job_data():
    data = request.get_json()
    job_description = data.get("jobDescription", "")

    if not job_description:
        return jsonify({"status": "error", "message": "Job description is required"}), 400

    ranked_resumes = rank_resumes(job_description)

        # Inject actual file names
    for resume in ranked_resumes:
        name = resume.get("resume")
        resume["file_name"] = f"{name}.pdf"  # or however the file is saved on disk

    return jsonify({
        "status": "success",
        "ranked_resumes": ranked_resumes
    })

import os
from werkzeug.utils import secure_filename, safe_join
from flask import send_from_directory
UPLOAD_FOLDER = "uploads"  # Update if different

@app.route("/download_resume/<filename>", methods=["GET"])
def download_resume(filename):
    try:
        # Safely join path to prevent path traversal
        file_path = safe_join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
        else:
            abort(404, description="File not found")
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
