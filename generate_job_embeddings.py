#!/usr/bin/env python3
"""
generate_job_embeddings.py

Generates sentence embeddings for job postings from a CSV file.
Enhances text with extracted skills from a rich vocabulary and saves both embeddings and metadata.
Supports checkpointing/resume so long runs can continue after interruption.
"""

import argparse
import os
import re
import unicodedata
import random
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------
# Config & Defaults (change to your absolute paths if desired)
# -----------------------
DEFAULT_JOBS_CSV = r"D:\New folder\UNIVERSITY docs\resume_matcher\data\cleaned\cleaned_jobs.csv"
DEFAULT_OUT_PATH = r"D:\New folder\UNIVERSITY docs\resume_matcher\data\embeddings\job_embeddings.pt"
DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 64
RANDOM_SEED = 42
# how often to write a checkpoint (in batches). Tune lower if you want more frequent saves.
DEFAULT_SAVE_EVERY_N_BATCHES = 10

# -----------------------
# Utility Functions
# -----------------------
def log(message: str, level: str = "INFO"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}][{level}] {message}")

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    # remove high-unicode emojis / exotic characters
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def dedupe_jobs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__key"] = (
        df["job_title"].fillna("").str.strip().str.lower()
        + " ||| "
        + df["job_description"].fillna("").str.strip().str.lower()
    )
    df = df.drop_duplicates(subset="__key").drop(columns="__key")
    return df.reset_index(drop=True)

def load_skill_vocab() -> List[str]:
    """Return an expanded, diverse skill vocabulary. You can extend this list."""
    return sorted(set([
        # --- condensed but diverse skill list (you can expand) ---
        "python", "java", "javascript", "typescript", "go", "rust", "c", "c++", "c#", "swift", "kotlin",
        "html", "css", "sql", "nosql", "mongodb", "postgresql", "mysql",
        "node.js", "react", "angular", "vue.js", "django", "flask", "fastapi", "spring boot",
        "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd",
        "aws", "azure", "gcp", "cloud computing", "serverless",
        "machine learning", "deep learning", "nlp", "computer vision", "pytorch", "tensorflow", "keras",
        "pandas", "numpy", "scikit-learn", "spark", "hadoop", "tableau", "power bi",
        "sql", "excel", "data engineering", "etl", "mlops",
        "cybersecurity", "penetration testing", "ethical hacking",
        "seo", "content marketing", "google analytics", "facebook ads",
        "figma", "adobe", "photoshop", "illustrator", "ux design", "ui design",
        "project management", "agile", "scrum", "kanban", "product management",
        "financial analysis", "accounting", "sap", "quickbooks",
        "supply chain management", "logistics", "procurement",
        "clinical documentation", "nursing", "patient care",
        "technical writing", "content writing", "copywriting",
        "communication", "leadership", "teamwork", "problem solving"
    ]))

def simple_skill_extractor(text: str, skill_vocab: List[str]) -> List[str]:
    """
    Word-boundary-aware skill extractor. For symbol-heavy skills (c++, c#), also fallback to substring match.
    Returns lowercase skills found.
    """
    text_l = text.lower()
    found = set()
    for skill in skill_vocab:
        s = skill.lower()
        # word-boundary match (works for multi-word skills)
        try:
            if re.search(rf"\b{re.escape(s)}\b", text_l):
                found.add(s)
                continue
        except re.error:
            # fallback if regex fails for some pattern
            pass
        # substring fallback (useful for 'c++', 'c#', 'node.js' etc.)
        if s in text_l:
            found.add(s)
    return sorted(found)

def enrich_job_text(title: str, description: str, skills: List[str]) -> str:
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if description and description.strip():
        parts.append(description.strip())
    if skills:
        parts.append("Skills required: " + ", ".join(skills))
    return " . ".join(parts)

def batch_encode_and_return_cpu(model, texts: List[str], device: str):
    """
    Encode a list of texts and return a CPU tensor.
    (For batching we call this per-batch.)
    """
    embs = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=False,
        device=device
    )
    return embs.cpu()

def save_checkpoint(out_path: str, titles: List[str], descriptions: List[str],
                    enriched_texts: List[str], skills: List[List[str]], embeddings_cpu: torch.Tensor):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # ensure float32 on cpu
    payload = {
        "job_titles": titles,
        "job_descriptions": descriptions,
        "enriched_texts": enriched_texts,
        "skills": skills,
        "embeddings": embeddings_cpu.to(torch.float32)
    }
    torch.save(payload, out_path)
    log(f"Saved {len(titles)} job embeddings to {out_path}", "SUCCESS")

    # companion CSV for easy inspection
    skills_csv = out_path.replace(".pt", "_skills.csv")
    pd.DataFrame({
        "job_title": titles,
        "job_description": descriptions,
        "skills": [", ".join(s) for s in skills]
    }).to_csv(skills_csv, index=False)
    log(f"Saved extracted skills to {skills_csv}", "SUCCESS")

def load_checkpoint_if_exists(out_path: str) -> Optional[dict]:
    if not os.path.exists(out_path):
        return None
    try:
        data = torch.load(out_path, map_location="cpu")
        # basic validation
        if "job_titles" in data and "embeddings" in data:
            return data
        else:
            log("Existing checkpoint found but missing expected keys — ignoring.", "WARNING")
            return None
    except Exception as e:
        log(f"Failed to load existing checkpoint: {e}", "WARNING")
        return None

# -----------------------
# Main
# -----------------------
def main(jobs_csv: str, out_path: str, model_name: str, batch_size: int, no_gpu: bool, save_every_n_batches: int):
    set_seed()
    device = "cuda" if torch.cuda.is_available() and not no_gpu else "cpu"
    log(f"Using device: {device}")

    log(f"Loading model: {model_name} (this may take a while first run)")
    model = SentenceTransformer(model_name, device=device)

    if not os.path.exists(jobs_csv):
        log(f"Jobs CSV not found: {jobs_csv}", "ERROR")
        return

    log(f"Reading CSV: {jobs_csv}")
    df = pd.read_csv(jobs_csv)

    if "job_title" not in df.columns or "job_description" not in df.columns:
        log("CSV must contain 'job_title' and 'job_description' columns.", "ERROR")
        return

    # normalize and dedupe
    df["job_title"] = df["job_title"].astype(str).apply(normalize_text)
    df["job_description"] = df["job_description"].astype(str).apply(normalize_text)
    df = dedupe_jobs(df)
    total_jobs = len(df)
    log(f"{total_jobs} unique job postings after deduplication.")

    # prepare enriched texts and skills (we compute for all rows so indexes stable)
    skill_vocab = load_skill_vocab()
    enriched_texts = []
    skills_per_job = []
    log("Extracting skills and building enriched texts (this is quick)...")
    for _, row in df.iterrows():
        s = simple_skill_extractor(row["job_title"] + " " + row["job_description"], skill_vocab)
        skills_per_job.append(s)
        enriched_texts.append(enrich_job_text(row["job_title"], row["job_description"], s))

    # try to resume from existing out_path
    existing = load_checkpoint_if_exists(out_path)
    if existing:
        saved_count = len(existing["job_titles"])
        # quick sanity check: ensure saved_count <= total_jobs and first/last title match the CSV
        resume_ok = True
        if saved_count > total_jobs:
            log("Saved checkpoint contains more entries than current CSV — will restart from scratch.", "WARNING")
            resume_ok = False
        else:
            # check first and last saved item match the CSV to avoid silent mismatches
            if saved_count >= 1:
                if existing["job_titles"][0] != df.loc[0, "job_title"] or existing["job_titles"][-1] != df.loc[saved_count - 1, "job_title"]:
                    log("Checkpoint appears mismatched with current CSV (title mismatch). Will restart from scratch.", "WARNING")
                    resume_ok = False

        if resume_ok:
            log(f"Resuming from checkpoint with {saved_count} already processed jobs.", "INFO")
            processed_titles = list(existing["job_titles"])
            processed_descriptions = list(existing.get("job_descriptions", []))
            processed_enriched = list(existing.get("enriched_texts", []))
            processed_skills = list(existing.get("skills", []))
            # embeddings tensor on CPU
            processed_embeddings = existing["embeddings"].cpu()
            start_idx = processed_embeddings.shape[0]
            # If saved_count and start_idx mismatch, trust start_idx
            if start_idx != saved_count:
                log(f"Checkpoint count mismatch: saved_count={saved_count}, embeddings rows={start_idx}. Using embeddings rows={start_idx}.", "WARNING")
                saved_count = start_idx
            # If saved_count == total_jobs -> nothing to do
            if saved_count >= total_jobs:
                log("All jobs already embedded in checkpoint file. Exiting.", "INFO")
                return
        else:
            log("Removing mismatched checkpoint and restarting from scratch.", "INFO")
            # rename old checkpoint
            try:
                backup = out_path + ".old"
                os.replace(out_path, backup)
                log(f"Backed up mismatched checkpoint to {backup}", "INFO")
            except Exception:
                log("Could not backup checkpoint; continuing and overwriting.", "WARNING")
            processed_titles, processed_descriptions, processed_enriched, processed_skills = [], [], [], []
            processed_embeddings = None
            start_idx = 0
            saved_count = 0
    else:
        processed_titles, processed_descriptions, processed_enriched, processed_skills = [], [], [], []
        processed_embeddings = None
        start_idx = 0
        saved_count = 0

    # We'll gather per-batch embeddings in chunks (CPU tensors) and save periodically
    embeddings_chunks = []
    if processed_embeddings is not None:
        embeddings_chunks.append(processed_embeddings)

    # if resuming, populate processed_* lists from df up to start_idx (if not loaded from checkpoint)
    if saved_count > 0 and (not processed_titles):
        processed_titles = df["job_title"].tolist()[:saved_count]
        processed_descriptions = df["job_description"].tolist()[:saved_count]
        processed_enriched = enriched_texts[:saved_count]
        processed_skills = skills_per_job[:saved_count]

    # iterate batches
    batches = list(range(start_idx, total_jobs, batch_size))
    if not batches:
        log("Nothing to process (start index >= total jobs).", "INFO")
        return

    log(f"Processing batches starting at index {start_idx}. Number of batches: {len(batches)}")
    batch_counter = 0
    for batch_start in tqdm(batches, desc="Overall progress"):
        batch_counter += 1
        i = batch_start
        j = min(i + batch_size, total_jobs)
        batch_texts = enriched_texts[i:j]
        # encode on device and move to cpu
        batch_embs_cpu = batch_encode_and_return_cpu(model, batch_texts, device)
        embeddings_chunks.append(batch_embs_cpu)

        # append titles/descriptions/skills to processed lists
        processed_titles.extend(df["job_title"].tolist()[i:j])
        processed_descriptions.extend(df["job_description"].tolist()[i:j])
        processed_enriched.extend(enriched_texts[i:j])
        processed_skills.extend(skills_per_job[i:j])

        # periodic save
        if batch_counter % save_every_n_batches == 0 or j >= total_jobs:
            # concatenate chunks
            all_embs = torch.cat(embeddings_chunks, dim=0) if len(embeddings_chunks) > 1 else embeddings_chunks[0]
            save_checkpoint(out_path, processed_titles, processed_descriptions, processed_enriched, processed_skills, all_embs)
            # free memory by keeping a single chunk
            embeddings_chunks = [all_embs]

        # small safety: release batch_embs_cpu reference
        del batch_embs_cpu

    # final ensure saved
    final_embs = torch.cat(embeddings_chunks, dim=0) if len(embeddings_chunks) > 1 else embeddings_chunks[0]
    save_checkpoint(out_path, processed_titles, processed_descriptions, processed_enriched, processed_skills, final_embs)
    log("All done.", "INFO")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs_csv", type=str, default=DEFAULT_JOBS_CSV)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_PATH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU even if GPU is available.")
    parser.add_argument("--save_every_n_batches", type=int, default=DEFAULT_SAVE_EVERY_N_BATCHES,
                        help="How many batches between checkpoint saves.")
    args = parser.parse_args()

    main(args.jobs_csv, args.out, args.model, args.batch_size, args.no_gpu, args.save_every_n_batches)
