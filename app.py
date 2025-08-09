import streamlit as st
import torch
from extract_text import extract_resume_text
from entity_extractor import extract_skills, extract_experience
from matcher import match_single_resume, build_tfidf_cache, load_job_embeddings


# Streamlit Config
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("Resume Matcher System")
st.write("Upload your resume to find the top matching job descriptions!")

# Load job embeddings once
@st.cache_resource(show_spinner="Loading job embeddings and descriptions...")
def load_job_data():
    embedding_path = r"D:\New folder\UNIVERSITY docs\resume_matcher\data\embeddings\job_embeddings.pt"
    job_titles, job_descriptions, job_embeddings = load_job_embeddings(embedding_path)
    build_tfidf_cache(job_descriptions)
    return job_titles, job_descriptions, job_embeddings

job_titles, job_descriptions, job_embeddings = load_job_data()

# Resume Upload
uploaded_file = st.file_uploader(
    "Upload your resume (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"]
)

# Main logic
if uploaded_file:
    try:
        # Step 1: Extract text
        with st.spinner("Extracting text from resume..."):
            resume_text = extract_resume_text(uploaded_file)

        if not resume_text.strip():
            st.error("The resume appears to be empty.")
            st.stop()

        st.success("Resume text extracted successfully!")

        # Step 2: Extract entities
        with st.spinner("🔍 Extracting skills and experience..."):
            skills = extract_skills(resume_text) or []
            experience = extract_experience(resume_text)

        # Display extracted details
        st.markdown("### Extracted Skills")
        st.write(", ".join(skills) if skills else "⚠️ No skills detected — matching will be purely semantic.")

        st.markdown("### Estimated Experience")
        st.write(f"{experience} years" if experience else "⚠️ Not detected.")

        st.divider()

        # Step 3: Matching settings
        top_k = st.slider("Number of top matches", 1, 10, 5)
        remove_duplicates = st.checkbox("🧹 Remove duplicate job titles", value=True)
        min_skill_matches = st.slider("Minimum required skill matches", 1, 5, 1)

        # Step 4: Match resume
        with st.spinner("🔗 Matching resume to jobs..."):
            matches = match_single_resume(
                resume_text=resume_text,
                job_titles=job_titles,
                job_descriptions=job_descriptions,
                job_embeddings=job_embeddings,
                top_k=top_k,
                skills=skills if skills else None,
                remove_duplicates=remove_duplicates,
                min_skill_matches=min_skill_matches
            )

        # Step 5: Display results
        if not matches:
            st.warning(" No matching jobs found. Try uploading a more detailed resume.")
        else:
            st.success(f"Found {len(matches)} matching job(s)!")
            for idx, job in enumerate(matches, 1):
                st.markdown(f"### {idx}. {job['job_title']}")
                st.write(f"**Match Score:** {job['score']:.2f}")
                st.write(job['job_description'])

    except RuntimeError as e:
        if "not enough memory" in str(e).lower():
            st.error("Memory Error: Try reducing model size or running on CPU.")
        else:
            st.error(f"Runtime Error: {e}")

    except Exception as e:
        st.error(f"Unexpected Error: {type(e).__name__} — {e}")
