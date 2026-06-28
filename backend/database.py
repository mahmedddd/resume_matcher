"""
Database setup, models, and session management.
"""
from datetime import datetime
from typing import AsyncGenerator, List, Optional
import json

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, func, ForeignKey, JSON
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from config import settings


engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Internship(Base):
    __tablename__ = "internships"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(300), nullable=False)
    company = Column(String(200), nullable=False)
    city = Column(String(100), nullable=False, index=True)   # Lahore / Karachi / Remote etc.
    is_remote = Column(Boolean, default=False, index=True)
    description = Column(Text, nullable=True)
    skills_required = Column(Text, nullable=True)            # JSON list stored as string
    url = Column(String(1000), nullable=True)
    source = Column(String(100), nullable=False)             # rozee / mustakbil / internshala / firecrawl
    deadline = Column(String(100), nullable=True)
    salary = Column(String(100), nullable=True)
    embedding = Column(Text, nullable=True)                   # JSON float list
    scraped_at = Column(DateTime, default=datetime.utcnow, index=True)

    def get_skills(self) -> List[str]:
        if not self.skills_required:
            return []
        try:
            return json.loads(self.skills_required)
        except Exception:
            return [s.strip() for s in self.skills_required.split(",") if s.strip()]

    def get_embedding(self) -> Optional[List[float]]:
        if not self.embedding:
            return None
        try:
            return json.loads(self.embedding)
        except Exception:
            return None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


class UserProfile(Base):
    __tablename__ = "user_profile"
    id = Column(Integer, primary_key=True)
    full_name = Column(String)
    email = Column(String)
    phone = Column(String)
    linkedin_url = Column(String)
    linkedin_email = Column(String, nullable=True)     # For LinkedIn Easy Apply
    linkedin_password = Column(String, nullable=True)  # Stored locally, dev only
    github_url = Column(String)
    portfolio_url = Column(String)
    bio_summary = Column(Text)
    cv_path = Column(String)
    experience = Column(JSON, default=list) # List of dicts
    projects = Column(JSON, default=list)   # List of dicts
    education = Column(JSON, default=list)  # List of strings/dicts
    skills = Column(JSON, default=list)     # List of strings

class Application(Base):
    __tablename__ = "applications"
    id = Column(Integer, primary_key=True)
    internship_id = Column(Integer, ForeignKey("internships.id"))
    # Pending -> Applying -> Applied ✓
    #                    -> Awaiting You -> Applying -> Applied ✓
    #                    -> Needs LinkedIn -> Applying -> Applied ✓
    #                    -> Manual Required (instructions in notes)
    #                    -> Paused (user paused mid-run)
    #                    -> Cancelled
    #                    -> Failed
    status = Column(String, default="Pending")
    outreach_email = Column(Text)
    applied_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    apply_url = Column(String, nullable=True)          # Direct application URL for manual fallback
    questions_pending = Column(Text, nullable=True)    # JSON list of questions for the human
    human_answers = Column(Text, nullable=True)        # JSON dict of question -> answer
    is_cancelled = Column(Boolean, default=False)      # Agent checks this mid-run
    is_paused = Column(Boolean, default=False)         # Agent checks this mid-run

    internship = relationship("Internship", back_populates="applications")

Internship.applications = relationship("Application", back_populates="internship")

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
