from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

# Load SBERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load dataset
columns_to_read = ['Job Description', 'skills']

#df = pd.read_csv("job_descriptions.csv")

df = pd.read_csv(
    'job_descriptions.csv',   # Replace with your file path
    usecols=columns_to_read,  # Select only these columns
    nrows=100000       # Read only the first 1lakh rows
)


# Convert dataset into SBERT format
train_samples = [InputExample(texts=[row["Job Description"], row["skills"]]) for _, row in df.iterrows()]

# DataLoader for training
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

# Use Unsupervised Multiple Negatives Ranking Loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

# Save the trained model
model.save("1lakh_trained")
