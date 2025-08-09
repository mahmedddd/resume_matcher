import pandas as pd
import torch
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# Load model once
# =========================
# Must match the model used for job embeddings generation
model = SentenceTransformer("all-mpnet-base-v2")


# Globals for efficiency
tfidf_vectorizer = None
job_tfidf_matrix = None


def load_cleaned_data(resume_path, job_path):
    """Load preprocessed resumes and jobs."""
    resumes = pd.read_csv(resume_path)
    jobs = pd.read_csv(job_path)
    return resumes, jobs


def embed_text(text_list):
    """Generate semantic embeddings for a list of texts."""
    return model.encode(text_list, convert_to_tensor=True, show_progress_bar=False)


def load_job_embeddings(embedding_path):
    """Load precomputed job embeddings from disk."""
    data = torch.load(embedding_path, map_location=torch.device("cpu"))
    return data["job_titles"], data["job_descriptions"], data["embeddings"]


def build_tfidf_cache(job_descriptions):
    """Build TF-IDF matrix once for optional keyword filtering."""
    global tfidf_vectorizer, job_tfidf_matrix
    tfidf_vectorizer = TfidfVectorizer(lowercase=True)
    job_tfidf_matrix = tfidf_vectorizer.fit_transform(job_descriptions)


def normalize_scores(scores):
    """Normalize a list/array of scores to the range [0,1]."""
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def is_duplicate(title, seen_titles):
    """Check if job title is already in results."""
    title = title.lower().strip()
    if title in seen_titles:
        return True
    seen_titles.add(title)
    return False


def get_top_unique_matches(similarities, job_titles, job_descriptions, top_k=3):
    """Return top unique matches based on similarity."""
    top_indices = torch.topk(similarities, k=len(similarities)).indices
    seen_titles = set()
    results = []

    for idx in top_indices:
        idx = idx.item()
        title = job_titles[idx]
        if not is_duplicate(title, seen_titles):
            results.append({
                "score": round(float(similarities[idx]), 4),
                "job_title": title,
                "job_description": job_descriptions[idx][:300] + "..."
            })
        if len(results) >= top_k:
            break
    return results


def match_single_resume(
    resume_text,
    job_titles,
    job_descriptions,
    job_embeddings,
    top_k=3,
    skills=None,
    remove_duplicates=True,
    min_skill_matches=1,
    skill_weight=0.3
):
    """
    Match a single resume to jobs using embeddings + skill filtering.
    Steps:
      1. Filter jobs by skills (if provided).
      2. Embed resume.
      3. Compute semantic similarity.
      4. Optionally weight by skill match ratio.
      5. Return top matches.
    """
    # === Step 1: Filter jobs by skills ===
    if skills:
        print(f"[*] Filtering jobs requiring at least {min_skill_matches} skill matches...")
        skill_set = {skill.lower().strip() for skill in skills if skill.strip()}

        def skill_match_count(desc, title):
            text = (desc + " " + title).lower()
            return sum(bool(re.search(rf"\b{re.escape(skill)}\b", text)) for skill in skill_set)

        filtered_indices = [
            i for i, desc in enumerate(job_descriptions)
            if skill_match_count(desc, job_titles[i]) >= min_skill_matches
        ]
    else:
        print("[*] No skills provided — using all jobs.")
        filtered_indices = list(range(len(job_descriptions)))

    if not filtered_indices:
        print("[!] No jobs matched skill filter. Using all jobs.")
        filtered_indices = list(range(len(job_descriptions)))

    filtered_titles = [job_titles[i] for i in filtered_indices]
    filtered_descriptions = [job_descriptions[i] for i in filtered_indices]
    filtered_embeddings = job_embeddings[filtered_indices]

    # === Step 2: Embed resume ===
    print("[*] Embedding resume...")
    resume_embedding = embed_text([resume_text])[0]

    # === Step 3: Compute similarity ===
    print(f"[*] Computing similarities for {len(filtered_indices)} jobs...")
    similarities = util.cos_sim(resume_embedding, filtered_embeddings)[0]

    # === Step 4: Weighted scoring with skill match ratio ===
    if skills:
        print("[*] Applying skill match weighting...")
        skill_match_scores = []
        for i, desc in enumerate(filtered_descriptions):
            matches = sum(
                bool(re.search(rf"\b{re.escape(skill)}\b", (desc + " " + filtered_titles[i]).lower()))
                for skill in skill_set
            )
            skill_match_scores.append(matches / len(skill_set))
        skill_match_scores = normalize_scores(skill_match_scores)
        similarities = (1 - skill_weight) * similarities + skill_weight * torch.tensor(skill_match_scores)

    # === Step 5: Select top matches ===
    print("[*] Selecting top matches...")
    if remove_duplicates:
        return get_top_unique_matches(similarities, filtered_titles, filtered_descriptions, top_k)
    else:
        top_indices = torch.topk(similarities, k=top_k).indices
        return [
            {
                "score": round(float(similarities[i]), 4),
                "job_title": filtered_titles[i],
                "job_description": filtered_descriptions[i][:300] + "..."
            }
            for i in top_indices
        ]
