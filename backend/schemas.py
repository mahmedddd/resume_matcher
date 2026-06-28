from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserProfileSchema(BaseModel):
    full_name: str = ""
    email: str = ""
    phone: str = ""
    linkedin_url: Optional[str] = ""
    linkedin_email: Optional[str] = ""
    linkedin_password: Optional[str] = ""
    github_url: Optional[str] = ""
    portfolio_url: Optional[str] = ""
    bio_summary: Optional[str] = ""
    experience: Optional[List[dict]] = []
    projects: Optional[List[dict]] = []
    education: Optional[List[dict]] = []
    skills: Optional[List[str]] = []

    class Config:
        from_attributes = True

class ApplicationOut(BaseModel):
    id: int
    internship_id: int
    status: str
    applied_at: datetime
    company: str
    job_title: str
    notes: Optional[str] = ""
    apply_url: Optional[str] = None
    questions_pending: Optional[List[str]] = None
    is_cancelled: Optional[bool] = False
    is_paused: Optional[bool] = False

    class Config:
        from_attributes = True
