import os
import re
import pandas as pd
import unicodedata

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Normalize unicode (e.g., remove accented characters)
    text = unicodedata.normalize("NFKD", text)

    # Decode garbage sequences
    text = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")

    # Remove known junk characters
    text = re.sub(r'[âÂÃ¢€¦™“”¢‘’]', '', text)

    # Replace multiple dots and newlines
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)

    # Final cleanup
    text = text.strip()
    return text

def clean_resume_dataset(input_path, output_path):
    print("[*] Reading input file...")
    df = pd.read_csv(input_path)

    # Drop rows with very short or missing resume text
    df = df[df['Resume'].astype(str).str.len() > 50]

    print("[*] Cleaning resume text...")
    df['resume_text'] = df['Resume'].apply(clean_text)

    # Clean and standardize category column
    df['category'] = df['Category'].astype(str).str.strip()

    # Drop unnecessary columns
    df = df[['category', 'resume_text']]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[✓] Cleaned resumes saved to: {output_path}")
    print(f"[✓] Total cleaned resumes: {len(df)}")
    print(f"[✓] Unique categories: {df['category'].nunique()}")

if __name__ == "__main__":
    clean_resume_dataset(
        r"D:\New folder\UNIVERSITY docs\resume_matcher\data\UpdatedResumeDataSet.csv",
        r"D:\New folder\UNIVERSITY docs\resume_matcher\data\Ucleaned_resumes.csv"
    )
