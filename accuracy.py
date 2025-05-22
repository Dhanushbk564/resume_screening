from sentence_transformers import SentenceTransformer, util
import pandas as pd
import random

# Load the fine-tuned SBERT model
model = SentenceTransformer("unsupervised_finetuned_sbert_new")

# Load test data
test_df = pd.read_csv("test_data.csv")

# Create a corpus of all resume skills to sample distractors
all_resumes = test_df["skills"].tolist()

def compute_mrr(model, test_df, num_distractors=9):
    reciprocal_ranks = []

    for _, row in test_df.iterrows():
        query = row["Job Description"]
        correct_resume = row["skills"]

        # Pick k distractors (excluding the correct one)
        distractors = random.sample([r for r in all_resumes if r != correct_resume], num_distractors)
        candidates = [correct_resume] + distractors
        random.shuffle(candidates)

        # Encode and score
        query_emb = model.encode(query)
        candidate_embs = model.encode(candidates)
        scores = util.cos_sim(query_emb, candidate_embs)[0]

        # Rank and get reciprocal rank
        sorted_indices = scores.argsort(descending=True)
        rank = sorted_indices.tolist().index(0) + 1  # index of correct_resume in candidates
        reciprocal_ranks.append(1 / rank)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mrr

# Run MRR evaluation
mrr_score = compute_mrr(model, test_df, num_distractors=9)
print(f"\n MRR: {mrr_score:.4f} ({mrr_score * 100:.2f}%)")
