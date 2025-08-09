import os
import re
import pandas as pd
import unicodedata

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Normalize to decompose accented characters
    text = unicodedata.normalize("NFKD", text)

    # Decode corrupted characters from encoding issues
    text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")

    # Remove known misencoding characters
    text = re.sub(r"[âÂÃ¢€¦™“”¢‘’]", "", text)

    # Remove brackets, excess punctuation, whitespace artifacts
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def clean_job_dataset(input_path, output_path):
    print("[*] Reading job dataset...")
    df = pd.read_csv(input_path)

    # Keep only necessary columns and drop missing descriptions
    df = df[['Job Title', 'Job Description']].dropna()

    print("[*] Cleaning job title and description fields...")
    df['job_title'] = df['Job Title'].apply(clean_text)
    df['job_description'] = df['Job Description'].apply(clean_text)

    # Keep cleaned columns only
    df = df[['job_title', 'job_description']]

    # Save cleaned jobs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[✓] Cleaned job descriptions saved to: {output_path}")
    print(f"[✓] Total cleaned jobs: {len(df)}")

if __name__ == "__main__":
    clean_job_dataset(
        r"D:\New folder\UNIVERSITY docs\resume_matcher\data\job_descriptions.csv",
        r"D:\New folder\UNIVERSITY docs\resume_matcher\data\cleaned_jobs.csv"
    )

