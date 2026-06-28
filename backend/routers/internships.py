"""
Router: GET /api/v1/internships/internships, GET /api/v1/internships/stats
Includes smart deadline filtering: expired jobs are automatically excluded.
"""
import json
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import distinct, func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from database import Internship, get_db
from models import CityCount, InternshipListResponse, InternshipOut, StatsResponse
from deadline_utils import deadline_is_valid, parse_deadline

router = APIRouter()


def _build_item(row: Internship) -> InternshipOut:
    return InternshipOut(**{
        "id": row.id,
        "title": row.title,
        "company": row.company,
        "city": row.city,
        "is_remote": row.is_remote,
        "description": row.description,
        "skills_required": row.get_skills(),
        "url": row.url,
        "source": row.source,
        "deadline": row.deadline,
        "salary": row.salary,
        "scraped_at": row.scraped_at,
    })


@router.get("/internships", response_model=InternshipListResponse)
async def list_internships(
    city: Optional[str] = Query(None, description="Filter by city name or 'Remote'"),
    remote_only: bool = Query(False),
    source: Optional[str] = Query(None),
    search: Optional[str] = Query(None, description="Search in title, company, description"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    hide_expired: bool = Query(True, description="Hide internships whose deadline has passed"),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Internship)

    if city:
        if city.lower() == "remote":
            stmt = stmt.where(
                or_(Internship.is_remote == True,
                    Internship.city.ilike('%remote%'),
                    Internship.city.ilike('%anywhere%'))
            )
        else:
            stmt = stmt.where(Internship.city.ilike(f"%{city}%"))

    if remote_only:
        stmt = stmt.where(Internship.is_remote == True)

    if source:
        stmt = stmt.where(Internship.source == source)

    if search:
        term = f"%{search}%"
        stmt = stmt.where(
            or_(
                Internship.title.ilike(term),
                Internship.company.ilike(term),
                Internship.description.ilike(term),
            )
        )

    # Fetch all matching rows before deadline filter (SQLite can't parse dates reliably)
    stmt = stmt.order_by(Internship.scraped_at.desc())
    rows = (await db.execute(stmt)).scalars().all()

    # ---------- Deadline filter (Python-side) ----------
    today = date.today()
    if hide_expired:
        rows = [r for r in rows if deadline_is_valid(r.deadline, today)]

    total = len(rows)

    # Paginate in Python after filtering
    start = (page - 1) * per_page
    paginated = rows[start: start + per_page]

    items = [_build_item(r) for r in paginated]

    return InternshipListResponse(total=total, page=page, per_page=per_page, items=items)


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Stats are computed over ALL internships (including expired = historical data)."""
    total = (await db.execute(select(func.count(Internship.id)))).scalar_one()
    companies = (await db.execute(select(func.count(distinct(Internship.company))))).scalar_one()
    remote_count = (await db.execute(
        select(func.count(Internship.id)).where(Internship.is_remote == True)
    )).scalar_one()
    last_scraped = (await db.execute(select(func.max(Internship.scraped_at)))).scalar_one()

    city_rows = (await db.execute(
        select(Internship.city, func.count(Internship.id))
        .group_by(Internship.city)
        .order_by(func.count(Internship.id).desc())
    )).all()

    return StatsResponse(
        total_internships=total,
        total_companies=companies,
        remote_count=remote_count,
        onsite_count=total - remote_count,
        by_city=[CityCount(city=r[0], count=r[1]) for r in city_rows],
        last_scraped=last_scraped,
    )
