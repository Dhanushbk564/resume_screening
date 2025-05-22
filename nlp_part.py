import fitz  # PyMuPDF
import re
import spacy
from rapidfuzz import fuzz, process

# -----------------------
# Load LinkedIn Skill List
# -----------------------
with open("linkedin skill", "r", encoding="utf-8") as f:
    linkedin_skills = set(line.strip().lower() for line in f if line.strip())

# -----------------------
# Extract Resume Text (PDF)
# -----------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc).replace('\xa0', ' ').lower()

# -----------------------
# Text Cleaning Function
# -----------------------
def clean_text(text):
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text

resume_text = extract_text_from_pdf("aravindraja.pdf")  # change path if needed
cleaned_resume_text = clean_text(resume_text)

# -----------------------
# Extract Relevant Sections Only
# -----------------------
section_headers = ['skills', 'technical skills', 'experience', 'technical expertise', 'work experience']
section_matches = {}

for header in section_headers:
    pattern = rf"{header}[\s:]*\n?(.*?)(?=\n[A-Z][^\n]{{2,40}}\n|$)"
    match = re.search(pattern, cleaned_resume_text, re.DOTALL)
    if match:
        section_matches[header] = match.group(1).strip()

# Combine matched section text
combined_text = " ".join(section_matches.values())

# -----------------------
# Exact Match First (Fast & Accurate)
# -----------------------
exact_matches = set()

# Use word boundaries (\b) to match exact skills
for skill in linkedin_skills:
    if re.search(rf"\b{re.escape(skill)}\b", combined_text):
        exact_matches.add(skill)

# -----------------------
# Fuzzy Match with spaCy Phrases (Only matching exact phrases within context)
# -----------------------
nlp = spacy.load("en_core_web_sm")
doc = nlp(combined_text)

# Only use meaningful noun phrases
phrases = set(
    chunk.text.lower().strip() 
    for chunk in doc.noun_chunks 
    if len(chunk.text.strip()) > 3 and ' ' in chunk.text
)

fuzzy_matches = set()
for phrase in phrases:
    match, score, _ = process.extractOne(phrase, linkedin_skills, scorer=fuzz.token_sort_ratio)
    if score >= 92:
        # Check if the match is an exact substring within the phrase
        match_found = re.search(rf"\b{re.escape(match)}\b", phrase)
        if match_found:  # Check if a match was found
            fuzzy_matches.add(match)

# -----------------------
# Final Result
# -----------------------
final_matches = sorted(set(exact_matches).union(fuzzy_matches))

print("\n Matched Skills in Resume (filtered + precise):")
for skill in final_matches:
    print("-", skill)
