from sentence_transformers import SentenceTransformer, util

# Load fine-tuned SBERT model
model = SentenceTransformer("unsupervised_finetuned_sbert")

def rank_resumes(job_description, resumes):
    # Encode JD and resumes
    jd_embedding = model.encode(job_description)
    resume_embeddings = model.encode(resumes)

    # Compute similarity scores
    scores = util.cos_sim(jd_embedding, resume_embeddings)[0]

    # Rank resumes by similarity
    ranked_resumes = sorted(zip(resumes, scores.tolist()), key=lambda x: x[1], reverse=True)

    return ranked_resumes

# Example usage
job_desc = "Software Developer"
resume_list = [
    "Java, Python, JavaScript, Git, REST APIs, SQL, Object-Oriented Programming, Agile Methodology, HTML, CSS",
    "Manual Testing, Automation Testing, Test Case Design, Bug Tracking, Selenium, JIRA, Agile Methodology, Regression Testing, SQL, API Testing",
    "Python, R, Machine Learning, Data Analysis, Pandas, NumPy, SQL, Data Visualization, Scikit-learn, TensorFlow",
    "AWS, Docker, Kubernetes, Jenkins, Linux, Terraform, CI/CD, Bash Scripting, Git, Monitoring Tools",
    "HTML, CSS, JavaScript, React, Responsive Design, Git, Webpack, TypeScript, UI/UX Principles, REST APIs",
    "Node.js, Express.js, SQL, MongoDB, REST APIs, Authentication, Git, Docker, Testing Frameworks, Scalability"
]

ranked_results = rank_resumes(job_desc, resume_list)
for res, score in ranked_results:
    print(f"Resume: {res} | Score: {score:.4f}")
