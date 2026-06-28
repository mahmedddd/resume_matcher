import json
import tempfile
import os
import asyncio
from typing import List, Any
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Internship, get_db
from models import CVSkills, InternshipOut, MatchResponse
from nlp.cv_parser import parse_cv_pdf
from nlp.skill_extractor import extract_skills
from nlp.embedder import get_embedder
from nlp.matcher import rank_internships
from nlp.email_generator import generate_outreach_email

router = APIRouter()

class EmailDraftRequest(BaseModel):
    cv_skills: List[str]
    job_title: str
    company: str
    job_description: str

class EmailDraftResponse(BaseModel):
    draft: str


@router.post("/upload-cv", response_model=MatchResponse)
async def upload_cv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF CV file"),
    top_n: int = 10,
    db: AsyncSession = Depends(get_db),
):
    print(f"[DEBUG] Received upload request for file: '{file.filename}'", flush=True)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        print(f"[DEBUG] Rejecting file with invalid extension: '{file.filename}'", flush=True)
        raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Received: {file.filename}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode="wb") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: Extract text from PDF
        print(f"[CV Engine] Step 1: Parsing PDF {tmp_path}", flush=True)
        cv_text = await asyncio.to_thread(parse_cv_pdf, tmp_path)
        print(f"[CV Engine] Step 1 finished. Got {len(cv_text)} bytes", flush=True)
        if not cv_text.strip():
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

        # Step 2: Extract skills using LLM
        print("[CV Engine] Step 2: Extracting identity and skills...", flush=True)
        cv_skills: CVSkills = await extract_skills(cv_text)
        print(f"[CV Engine] Step 2 finished. Identity: {getattr(cv_skills, 'primary_role', '')}", flush=True)

        # Step 2.1: Mirror Extracted Identity to UserProfile
        from database import UserProfile
        profile = (await db.execute(select(UserProfile))).scalars().first()
        if not profile:
            profile = UserProfile()
            db.add(profile)
        
        if getattr(cv_skills, 'full_name', ''): profile.full_name = cv_skills.full_name
        if getattr(cv_skills, 'email', ''): profile.email = cv_skills.email
        if getattr(cv_skills, 'phone', ''): profile.phone = cv_skills.phone
        if getattr(cv_skills, 'linkedin_url', ''): profile.linkedin_url = cv_skills.linkedin_url
        if getattr(cv_skills, 'github_url', ''): profile.github_url = cv_skills.github_url
        if getattr(cv_skills, 'portfolio_url', ''): profile.portfolio_url = cv_skills.portfolio_url
        if getattr(cv_skills, 'summary', ''): profile.bio_summary = cv_skills.summary
        
        # New: Store experience and projects
        if hasattr(cv_skills, 'experience'):
            profile.experience = [exp.model_dump() if hasattr(exp, 'model_dump') else exp for exp in cv_skills.work_experience]
        if hasattr(cv_skills, 'projects'):
            profile.projects = [proj.model_dump() if hasattr(proj, 'model_dump') else proj for proj in cv_skills.projects]
        if hasattr(cv_skills, 'education'):
            # Some LLMs return a string, some a list. Safely store it in a list so it plays nice with frontend.
            edu_raw = cv_skills.education
            if isinstance(edu_raw, list):
                profile.education = [e.model_dump() if hasattr(e, 'model_dump') else e for e in edu_raw]
            elif isinstance(edu_raw, str) and edu_raw.strip():
                profile.education = [{"degree": edu_raw.strip(), "institution": "Extracted from CV", "year": ""}]
        if hasattr(cv_skills, 'skills'):
            profile.skills = cv_skills.skills or []
            
        await db.commit()

        # Step 2.5: Synchronously run scrapers for the exact role matched by the CV
        try:
            from routers.scraper import run_scrape
            if getattr(cv_skills, "primary_role", ""):
                print(f"[CV Scraper] Triggering background search for: {cv_skills.primary_role}", flush=True)
                background_tasks.add_task(
                    run_scrape,
                    sources=["rozee", "internshala", "linkedin", "mustakbil"], 
                    keywords=[cv_skills.primary_role]
                )
        except Exception as e:
            print(f"[CV Scraper] Soft fail initializing background scraper: {e}", flush=True)

        # Step 3: Embed rich CV representation
        rich_cv_text = f"Role: {getattr(cv_skills, 'primary_role', '') or ''}\n"
        rich_cv_text += f"Summary: {getattr(cv_skills, 'summary', '') or ''}\n"
        
        safe_skills = getattr(cv_skills, 'skills', []) or []
        rich_cv_text += f"Skills: {', '.join(safe_skills)}\n"
        
        safe_exp = getattr(cv_skills, 'work_experience', []) or []
        for exp in safe_exp:
            achv = getattr(exp, 'achievements', []) or []
            rich_cv_text += f"Experience: {getattr(exp, 'title', '')} at {getattr(exp, 'company', '')}. {' '.join(achv)}\n"
            
        safe_proj = getattr(cv_skills, 'projects', []) or []
        for proj in safe_proj:
            techs = getattr(proj, 'technologies', []) or []
            rich_cv_text += f"Project: {getattr(proj, 'name', '')}. {getattr(proj, 'description', '')}. Tech: {', '.join(techs)}\n"
            
        rich_cv_text += f"Education: {getattr(cv_skills, 'education', '') or ''}"

        print("[CV Engine] Step 3: Offloading Transformer Encoding to thread...", flush=True)
        embedder = get_embedder()
        cv_embedding = await asyncio.to_thread(
            embedder.encode,
            rich_cv_text[:3500],
            show_progress_bar=False,
        )
        cv_embedding = cv_embedding.tolist()
        print("[CV Engine] Step 3 finished executing neural encoding.", flush=True)

        # Step 4: Fetch all internships with embeddings
        print("[CV Engine] Step 4: Loading SQL DB Vectors...", flush=True)
        rows = (await db.execute(
            select(Internship).where(Internship.embedding != None)
        )).scalars().all()
        print(f"[CV Engine] Step 4 finished. Retrieved {len(rows)} matrix profiles.", flush=True)

        ranked = []
        if rows:
            # Step 5: Rank by cosine similarity
            print("[CV Engine] Step 5: Commencing Cosine Matrix Ranking...", flush=True)
            # Offload ranking to thread as it can be CPU heavy with many rows
            ranked = await asyncio.to_thread(
                rank_internships,
                cv_embedding,
                getattr(cv_skills, 'skills', []) or [],
                rows,
                top_n=top_n
            )
            print("[CV Engine] Step 5 finished ranking. Proceeding to reputation check.", flush=True)

            # Step 6: Anti-Scam Reputation Check
            try:
                print("[CV Engine] Step 6: Triggering Anti-Scam Reputation check", flush=True)
                from nlp.reputation import batch_reputation_check
                companies = [r.company for r in ranked]
                rep_map = await batch_reputation_check(companies)
                print(f"[CV Engine] Step 6: Completed Rep check, found {len(rep_map)} companies.", flush=True)
                
                for r in ranked:
                    if r.company in rep_map:
                        from models import ReputationAnalysis
                        r.reputation = ReputationAnalysis(**rep_map[r.company])
            except Exception as e:
                print(f"[Warning] Reputation check skipped/failed: {e}", flush=True)
        else:
            print("[CV Engine] Step 4: Warning - No internships in DB to match against.", flush=True)

        print("[CV Engine] ENDPOINT SUCCESS", flush=True)
        return MatchResponse(cv_skills=cv_skills, matches=ranked)

    finally:
        os.unlink(tmp_path)

@router.post("/draft-email", response_model=EmailDraftResponse)
async def draft_email(body: EmailDraftRequest):
    draft = await generate_outreach_email(
        cv_skills=body.cv_skills,
        job_title=body.job_title,
        company=body.company,
        job_description=body.job_description
    )
    return EmailDraftResponse(draft=draft)
