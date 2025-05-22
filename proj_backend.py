from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

# Initialize Flask app and allow CORS
app = Flask(__name__)
CORS(app)

# Load the fine-tuned SBERT model
model = SentenceTransformer("unsupervised_finetuned_sbert_new")

# Function to rank skills by relevance to job description
def rank_skills(job_description, skills):
    jd_embedding = model.encode(job_description)
    skill_embeddings = model.encode(skills)

    # Compute cosine similarities
    scores = util.cos_sim(jd_embedding, skill_embeddings)[0]
    
    # Sort skills by score (descending)
    ranked_skills = sorted(zip(skills, scores.tolist()), key=lambda x: x[1], reverse=True)

    return ranked_skills

# API endpoint
@app.route("/submit", methods=["POST"])
def submit_job_data():
    data = request.get_json()
    
    job_description = data.get("jobDescription", "")
    skills = data.get("skills", [])

    # Rank the skills
    ranked = rank_skills(job_description, skills)

    # Format response
    results = [{"skill": skill, "score": float(score)} for skill, score in ranked]

    return jsonify({
        "status": "success",
        "ranked_skills": results
    })

if __name__ == "__main__":
    app.run(debug=True)
