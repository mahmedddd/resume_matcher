"""
Router: POST /api/scrape — triggers scraper engine
"""
import asyncio
import time
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Internship, get_db
from models import ScrapeRequest, ScrapeResponse
from scrapers.normalizer import normalize

router = APIRouter()

# Track if a scrape is currently in-progress
_scraping = False
_last_result: dict = {}


async def run_scrape(sources: List[str], keywords: List[str]):
    global _scraping, _last_result
    _scraping = True
    start = time.time()
    scraped = []
    used = []

    tasks = []
    task_sources = []

    if "rozee" in sources:
        def _rozee_sync():
            import asyncio, sys
            loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from scrapers.rozee_scraper import RozeeScraper
                return loop.run_until_complete(RozeeScraper().scrape(keywords))
            finally:
                loop.close()
        import concurrent.futures
        tasks.append(asyncio.get_event_loop().run_in_executor(None, _rozee_sync))
        task_sources.append("rozee")

    if "mustakbil" in sources:
        async def run_mustakbil():
            from scrapers.mustakbil_scraper import MustakbilScraper
            return await MustakbilScraper().scrape(keywords)
        tasks.append(run_mustakbil())
        task_sources.append("mustakbil")

    if "internshala" in sources:
        def _internshala_sync():
            import asyncio, sys
            loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from scrapers.internshala_scraper import IntershalaScraper
                return loop.run_until_complete(IntershalaScraper().scrape(keywords))
            finally:
                loop.close()
        tasks.append(asyncio.get_event_loop().run_in_executor(None, _internshala_sync))
        task_sources.append("internshala")

    if "linkedin" in sources:
        def _linkedin_sync():
            import asyncio, sys
            loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from scrapers.linkedin_scraper import LinkedInScraper
                return loop.run_until_complete(LinkedInScraper().scrape(keywords))
            finally:
                loop.close()
        tasks.append(asyncio.get_event_loop().run_in_executor(None, _linkedin_sync))
        task_sources.append("linkedin")

    if "firecrawl" in sources:
        async def run_fc():
            from scrapers.firecrawl_scraper import FirecrawlScraper
            return await FirecrawlScraper().scrape(keywords)
        tasks.append(run_fc())
        task_sources.append("firecrawl")

    if "indeed" in sources:
        async def run_indeed():
            from scrapers.indeed_scraper import IndeedScraper
            return await IndeedScraper().scrape(keywords)
        tasks.append(run_indeed())
        task_sources.append("indeed")

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            src = task_sources[i]
            if isinstance(res, Exception):
                print(f"[Scraper] {src} failed: {res}")
            else:
                scraped.extend(res)
                used.append(src)

        # Normalize + save to DB
        normalized = [normalize(raw) for raw in scraped if raw]
        new_count = await save_internships(normalized)

        # Embed new internships in background
        asyncio.create_task(embed_all_unembedded())

        _last_result = {
            "scraped": len(scraped),
            "new_entries": new_count,
            "sources_used": used,
            "duration_seconds": round(time.time() - start, 2),
        }
    finally:
        _scraping = False


async def embed_all_unembedded():
    """Compute and store embeddings for internships that don't have one yet."""
    from nlp.embedder import get_embedder
    from database import AsyncSessionLocal
    import json as _json

    async with AsyncSessionLocal() as db:
        rows = (await db.execute(
            select(Internship).where(Internship.embedding == None)
        )).scalars().all()

        if not rows:
            return

        embedder = get_embedder()
        texts = [f"{r.title} {r.company} {r.description or ''} {r.skills_required or ''}" for r in rows]
        # Use a thread for the CPU-intensive encoding to avoid blocking the event loop
        embeddings_nd = await asyncio.to_thread(
            embedder.encode,
            texts,
            batch_size=32,
            show_progress_bar=False
        )
        embeddings = embeddings_nd.tolist()

        for row, emb in zip(rows, embeddings):
            row.embedding = _json.dumps(emb)

        await db.commit()
        print(f"[Embedder] Stored embeddings for {len(rows)} internships")


async def save_internships(normalized: list) -> int:
    """Insert new internships (skip duplicates by URL)."""
    from database import AsyncSessionLocal
    new_count = 0
    async with AsyncSessionLocal() as db:
        for data in normalized:
            if not data:
                continue
            # Check duplicate by URL
            if data.get("url"):
                exists = (await db.execute(
                    select(Internship).where(Internship.url == data["url"])
                )).scalar_one_or_none()
                if exists:
                    continue

            import json as _json
            skills_json = _json.dumps(data.get("skills_required", []))
            intern = Internship(
                title=data["title"],
                company=data["company"],
                city=data["city"],
                is_remote=data["is_remote"],
                description=data.get("description"),
                skills_required=skills_json,
                url=data.get("url"),
                source=data["source"],
                deadline=data.get("deadline"),
                salary=data.get("salary"),
            )
            db.add(intern)
            new_count += 1

        await db.commit()
    return new_count


@router.post("/scrape", response_model=ScrapeResponse)
async def trigger_scrape(
    body: ScrapeRequest,
    background_tasks: BackgroundTasks,
):
    global _scraping

    sources = body.sources or ["rozee", "mustakbil", "internshala", "linkedin", "indeed"]
    keywords = body.keywords or [
        # --- Tech / Software ---
        "software engineer intern Lahore",
        "software developer intern Karachi",
        "web developer intern Islamabad",
        "frontend developer intern Pakistan",
        "backend developer intern Pakistan",
        "full stack developer intern",
        "mobile app developer intern Pakistan",
        "android developer intern Pakistan",
        "ios developer intern Pakistan",
        "flutter developer intern Pakistan",
        # --- Data / AI ---
        "machine learning intern Pakistan",
        "data science intern Pakistan",
        "data analyst intern Lahore",
        "artificial intelligence intern Karachi",
        "NLP intern Pakistan",
        "computer vision intern Pakistan",
        # --- Design / Creative ---
        "UI UX designer intern Pakistan",
        "graphic designer intern Lahore",
        "video editor intern Pakistan",
        # --- Business / Marketing ---
        "digital marketing intern Pakistan",
        "social media intern Karachi",
        "business development intern Pakistan",
        "content writer intern Pakistan",
        "SEO intern Pakistan",
        "marketing intern Islamabad",
        # --- Finance / Operations ---
        "finance intern Lahore",
        "accounting intern Karachi",
        "HR intern Pakistan",
        # --- Remote ---
        "remote internship Pakistan",
        "work from home internship Pakistan",
        # --- City specific ---
        "internship Faisalabad",
        "internship Rawalpindi",
        "internship Multan",
        "internship Peshawar",
        "internship Sialkot",
    ]

    if _scraping:
        # Return last result if still running
        return ScrapeResponse(
            scraped=_last_result.get("scraped", 0),
            new_entries=_last_result.get("new_entries", 0),
            sources_used=_last_result.get("sources_used", []),
            duration_seconds=0.0,
        )

    background_tasks.add_task(run_scrape, sources, keywords)
    return ScrapeResponse(
        scraped=0,
        new_entries=0,
        sources_used=sources,
        duration_seconds=0.0,
    )


@router.get("/scrape/status")
async def scrape_status():
    return {
        "is_running": _scraping,
        "last_result": _last_result,
    }
