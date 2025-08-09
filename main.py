# main.py

import argparse
import torch
from extract_text import extract_resume_text
from entity_extractor import extract_skills, extract_experience
from matcher import match_single_resume

def load_job_data(embedding_path):
    """Load precomputed job embeddings."""
    data = torch.load(embedding_path, map_location=torch.device("cpu"))
    return data["job_titles"], data["job_descriptions"], data["embeddings"]

def run_cli(resume_path, embedding_path, top_k=3, min_skill_matches=1):
    """Run resume matching in CLI mode."""
    # Load job data
    job_titles, job_descriptions, job_embeddings = load_job_data(embedding_path)

    # Extract resume text
    print("[*] Extracting text from resume...")
    resume_text = extract_resume_text(resume_path)
    print("[✓] Resume text extracted.")

    # Extract skills and experience
    print("[*] Extracting skills and experience...")
    skills = extract_skills(resume_text)
    experience = extract_experience(resume_text)
    print(f"[✓] Skills found: {', '.join(skills) if skills else 'None'}")
    print(f"[✓] Estimated experience: {experience or 'Not detected'} years")

    # Match to jobs
    print(f"[*] Matching to jobs (top {top_k} results)...")
    matches = match_single_resume(
        resume_text,
        job_titles,
        job_descriptions,
        job_embeddings,
        top_k=top_k,
        skills=skills,
        min_skill_matches=min_skill_matches
    )

    # Display results
    print("\n=== TOP MATCHING JOBS ===\n")
    for idx, job in enumerate(matches, 1):
        print(f"{idx}. {job['job_title']} (Score: {job['score']:.2f})")
        print(f"   {job['job_description']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Matcher CLI")
    parser.add_argument("resume", help="C:\\Users\\ahmed\\Desktop\\ahmed's_Resume.pdf")
    parser.add_argument(
        "--embeddings",
        default="D:\\New folder\\UNIVERSITY docs\\resume_matcher\\data\\embeddings\\job_embeddings.pt",
        help="Path to job_embeddings.pt file"
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of top matches to display")
    parser.add_argument("--min_skill_matches", type=int, default=1, help="Minimum required skill matches")

    args = parser.parse_args()
    run_cli(args.resume, args.embeddings, top_k=args.top_k, min_skill_matches=args.min_skill_matches)
