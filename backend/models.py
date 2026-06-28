"""
Pydantic schemas for API request / response models.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict


# ─── Internship Schemas ──────────────────────────────────────────────────────

class InternshipBase(BaseModel):
    title: str
    company: str
    city: str
    is_remote: bool = False
    description: Optional[str] = None
    skills_required: Optional[List[str]] = None
    url: Optional[str] = None
    source: str
    deadline: Optional[str] = None
    salary: Optional[str] = None


class ReputationAnalysis(BaseModel):
    status: str
    summary: str

class InternshipOut(InternshipBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scraped_at: datetime
    match_score: Optional[float] = None
    matching_skills: Optional[List[str]] = None
    missing_skills: Optional[List[str]] = None
    reputation: Optional[ReputationAnalysis] = None


class InternshipListResponse(BaseModel):
    total: int
    page: int
    per_page: int
    items: List[InternshipOut]


# ─── Stats Schema ─────────────────────────────────────────────────────────────

class CityCount(BaseModel):
    city: str
    count: int


class StatsResponse(BaseModel):
    total_internships: int
    total_companies: int
    remote_count: int
    onsite_count: int
    by_city: List[CityCount]
    last_scraped: Optional[datetime] = None


# ─── CV / Match Schemas ───────────────────────────────────────────────────────

class WorkExperience(BaseModel):
    title: str = ""
    company: str = ""
    duration: str = ""
    achievements: List[str] = []

class Project(BaseModel):
    name: str = ""
    description: str = ""
    technologies: List[str] = []

class CVSkills(BaseModel):
    full_name: Optional[str] = ""
    email: Optional[str] = ""
    phone: Optional[str] = ""
    linkedin_url: Optional[str] = ""
    github_url: Optional[str] = ""
    portfolio_url: Optional[str] = ""
    
    primary_role: str = ""
    skills: List[str] = []
    experience_level: str = "junior"  # student | junior | mid
    work_experience: Optional[List[WorkExperience]] = []
    projects: Optional[List[Project]] = []
    education: Optional[str] = ""
    summary: Optional[str] = ""


class MatchResponse(BaseModel):
    cv_skills: CVSkills
    matches: List[InternshipOut]


# ─── Scraper Schemas ──────────────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    sources: Optional[List[str]] = None   # ["rozee", "mustakbil", "internshala", "firecrawl"]
    keywords: Optional[List[str]] = None  # override default keywords


class ScrapeResponse(BaseModel):
    scraped: int
    new_entries: int
    sources_used: List[str]
    duration_seconds: float
