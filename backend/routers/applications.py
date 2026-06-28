from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from database import get_db, Application, Internship, UserProfile
from schemas import ApplicationOut, UserProfileSchema
from typing import List
import json

router = APIRouter()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

async def _app_or_404(app_id: int, db: AsyncSession) -> Application:
    app = await db.get(Application, app_id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    return app


async def _db_check_signals(app_id: int):
    """Called by FormFiller mid-run to check pause/cancel flags."""
    from database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        app = await db.get(Application, app_id)
        if app:
            return app.is_paused, app.is_cancelled
        return False, False


def run_autonomous_apply_sync(app_id: int, url: str, profile_dict: dict):
    """Sync wrapper to run Playwright in a dedicated Proactor event loop."""
    import asyncio
    import sys
    
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
        
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_autonomous_apply(app_id, url, profile_dict))
    finally:
        loop.close()


async def run_autonomous_apply(app_id: int, url: str, profile_dict: dict):
    from automation.form_filler import FormFiller, AgentCancelledError, AgentPausedError
    from database import AsyncSessionLocal
    import asyncio

    print(f"[Agent] run_autonomous_apply is using loop type: {type(asyncio.get_running_loop()).__name__}")

    try:
        applier = FormFiller(profile_dict, app_id=app_id, db_check_fn=_db_check_signals)
        res = await applier.apply(url)
    except Exception as crash:
        # Playwright / OS crash before any status was written — guarantee DB update
        print(f"[Agent] CRITICAL crash for app {app_id}: {crash}")
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(
                    status="Failed",
                    notes=f"Agent crashed unexpectedly: {str(crash)[:200]}. Please re-apply.",
                )
            )
            await db.commit()
        return

    reason = res.get("reason", "")

    async with AsyncSessionLocal() as db:
        # ── Cancelled ──────────────────────────────────────────────────
        if reason == "cancelled":
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(status="Cancelled", notes="Application was cancelled by you.")
            )

        # ── Paused ─────────────────────────────────────────────────────
        elif reason == "paused":
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(status="Paused", notes="Application was paused by you. Resume to continue.")
            )

        # ── Human Q&A required ─────────────────────────────────────────
        elif reason == "human_required":
            questions = res.get("questions", [])
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(
                    status="Awaiting You",
                    questions_pending=json.dumps(questions),
                    notes=f"Paused: {len(questions)} question(s) need your personal input.",
                )
            )
            print(f"[Agent] App {app_id} awaiting human answers: {questions}")

        # ── LinkedIn credentials needed ────────────────────────────────
        elif reason == "linkedin_credentials_required":
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(
                    status="Needs LinkedIn",
                    notes="Your LinkedIn credentials are needed to use Easy Apply.",
                )
            )

        # ── Manual fallback ────────────────────────────────────────────
        elif reason == "manual_required":
            apply_url = res.get("apply_url", url)
            instructions = res.get("instructions", "Please apply manually via the link.")
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(
                    status="Manual Required",
                    apply_url=apply_url,
                    notes=instructions,
                )
            )

        # ── Success ────────────────────────────────────────────────────
        elif res.get("success"):
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(status="Applied", notes=res.get("reason", "Application submitted."), questions_pending=None)
            )

        # ── Hard failure ───────────────────────────────────────────────
        else:
            await db.execute(
                update(Application).where(Application.id == app_id)
                .values(status="Failed", notes=reason or "Unknown error.")
            )

        await db.commit()
    print(f"[Agent] App {app_id} -> {res}")


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@router.get("", response_model=List[ApplicationOut])
async def get_applications(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Application, Internship.company, Internship.title)
        .join(Internship)
        .order_by(Application.applied_at.desc())
    )
    apps = []
    for row in result.all():
        app, company, title = row
        pending_qs = None
        if app.questions_pending:
            try:
                pending_qs = json.loads(app.questions_pending)
            except Exception:
                pending_qs = None
        apps.append(ApplicationOut(
            id=app.id,
            internship_id=app.internship_id,
            status=app.status,
            applied_at=app.applied_at,
            company=company,
            job_title=title,
            notes=app.notes or "",
            apply_url=app.apply_url,
            questions_pending=pending_qs,
            is_cancelled=app.is_cancelled or False,
            is_paused=app.is_paused or False,
        ))
    return apps


@router.get("/{app_id}/status")
async def get_application_status(app_id: int, db: AsyncSession = Depends(get_db)):
    app = await _app_or_404(app_id, db)
    pending_qs = None
    if app.questions_pending:
        try:
            pending_qs = json.loads(app.questions_pending)
        except Exception:
            pending_qs = None
    return {
        "id": app.id,
        "status": app.status,
        "notes": app.notes or "",
        "apply_url": app.apply_url,
        "questions_pending": pending_qs,
        "is_paused": app.is_paused or False,
        "is_cancelled": app.is_cancelled or False,
    }


@router.post("/apply/{internship_id}")
async def trigger_apply(
    internship_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    internship = await db.get(Internship, internship_id)
    if not internship:
        raise HTTPException(status_code=404, detail="Internship not found")

    profile = (await db.execute(select(UserProfile))).scalars().first()
    if not profile:
        raise HTTPException(status_code=400, detail="Please set up your User Profile first.")

    new_app = Application(
        internship_id=internship_id,
        status="Applying",
        apply_url=internship.url,
    )
    db.add(new_app)
    await db.commit()
    await db.refresh(new_app)

    profile_dict = {
        "full_name": profile.full_name or "",
        "email": profile.email or "",
        "phone": profile.phone or "",
        "linkedin_url": profile.linkedin_url or "",
        "linkedin_email": profile.linkedin_email or "",
        "linkedin_password": profile.linkedin_password or "",
        "github_url": profile.github_url or "",
        "portfolio_url": profile.portfolio_url or "",
        "skills": profile.skills or [],
    }
    background_tasks.add_task(run_autonomous_apply_sync, new_app.id, internship.url, profile_dict)
    return {"message": "Autonomous application started.", "application_id": new_app.id}


@router.post("/{app_id}/answer")
async def submit_human_answers(
    app_id: int,
    answers: dict,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    app = await _app_or_404(app_id, db)
    if app.status != "Awaiting You":
        raise HTTPException(status_code=400, detail=f"Not awaiting answers (status: {app.status})")

    app.human_answers = json.dumps(answers)
    app.status = "Applying"
    await db.commit()

    internship = await db.get(Internship, app.internship_id)
    profile = (await db.execute(select(UserProfile))).scalars().first()
    if not internship or not profile:
        raise HTTPException(status_code=400, detail="Missing internship or profile data")

    profile_dict = {
        "full_name": profile.full_name or "",
        "email": profile.email or "",
        "phone": profile.phone or "",
        "linkedin_url": profile.linkedin_url or "",
        "linkedin_email": profile.linkedin_email or "",
        "linkedin_password": profile.linkedin_password or "",
        "github_url": profile.github_url or "",
        "portfolio_url": profile.portfolio_url or "",
        "skills": profile.skills or [],
        "human_answers": answers,
    }
    background_tasks.add_task(run_autonomous_apply_sync, app.id, app.apply_url or internship.url, profile_dict)
    return {"message": "Answers submitted. Agent resuming.", "application_id": app_id}


@router.post("/{app_id}/set-credentials")
async def set_linkedin_credentials(
    app_id: int,
    body: dict,  # { "linkedin_email": "...", "linkedin_password": "..." }
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Receive LinkedIn credentials, save to profile, and resume the application."""
    app = await _app_or_404(app_id, db)
    if app.status != "Needs LinkedIn":
        raise HTTPException(status_code=400, detail=f"Not waiting for credentials (status: {app.status})")

    li_email = body.get("linkedin_email", "").strip()
    li_password = body.get("linkedin_password", "").strip()
    if not li_email or not li_password:
        raise HTTPException(status_code=422, detail="Both linkedin_email and linkedin_password are required.")

    # Save credentials to user profile
    profile = (await db.execute(select(UserProfile))).scalars().first()
    if not profile:
        raise HTTPException(status_code=400, detail="User profile not found.")
    profile.linkedin_email = li_email
    profile.linkedin_password = li_password

    app.status = "Applying"
    app.notes = "Credentials received. Resuming LinkedIn Easy Apply…"
    await db.commit()

    internship = await db.get(Internship, app.internship_id)
    if not internship:
        raise HTTPException(status_code=404, detail="Internship not found.")

    profile_dict = {
        "full_name": profile.full_name or "",
        "email": profile.email or "",
        "phone": profile.phone or "",
        "linkedin_url": profile.linkedin_url or "",
        "linkedin_email": li_email,
        "linkedin_password": li_password,
        "github_url": profile.github_url or "",
        "portfolio_url": profile.portfolio_url or "",
        "skills": profile.skills or [],
    }
    background_tasks.add_task(run_autonomous_apply_sync, app.id, app.apply_url or internship.url, profile_dict)
    return {"message": "Credentials saved. Agent resuming.", "application_id": app_id}


@router.post("/{app_id}/pause")
async def pause_application(app_id: int, db: AsyncSession = Depends(get_db)):
    app = await _app_or_404(app_id, db)
    if app.status not in ("Applying", "Pending"):
        raise HTTPException(status_code=400, detail=f"Cannot pause application in status: {app.status}")
    await db.execute(
        update(Application).where(Application.id == app_id)
        .values(is_paused=True)
    )
    await db.commit()
    return {"message": "Pause signal sent to agent.", "application_id": app_id}


@router.post("/{app_id}/resume")
async def resume_application(
    app_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    app = await _app_or_404(app_id, db)
    if app.status != "Paused":
        raise HTTPException(status_code=400, detail=f"Application is not paused (status: {app.status})")

    internship = await db.get(Internship, app.internship_id)
    profile = (await db.execute(select(UserProfile))).scalars().first()
    if not internship or not profile:
        raise HTTPException(status_code=400, detail="Missing internship or profile data.")

    human_answers = {}
    if app.human_answers:
        try:
            human_answers = json.loads(app.human_answers)
        except Exception:
            pass

    await db.execute(
        update(Application).where(Application.id == app_id)
        .values(status="Applying", is_paused=False, notes="Resumed by user.")
    )
    await db.commit()

    profile_dict = {
        "full_name": profile.full_name or "",
        "email": profile.email or "",
        "phone": profile.phone or "",
        "linkedin_url": profile.linkedin_url or "",
        "linkedin_email": profile.linkedin_email or "",
        "linkedin_password": profile.linkedin_password or "",
        "github_url": profile.github_url or "",
        "portfolio_url": profile.portfolio_url or "",
        "skills": profile.skills or [],
        "human_answers": human_answers,
    }
    background_tasks.add_task(run_autonomous_apply_sync, app.id, app.apply_url or internship.url, profile_dict)
    return {"message": "Agent resumed.", "application_id": app_id}


@router.post("/{app_id}/cancel")
async def cancel_application(app_id: int, db: AsyncSession = Depends(get_db)):
    app = await _app_or_404(app_id, db)
    if app.status in ("Applied", "Cancelled"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel application in status: {app.status}")
    await db.execute(
        update(Application).where(Application.id == app_id)
        .values(is_cancelled=True, status="Cancelled", notes="Cancelled by you.")
    )
    await db.commit()
    return {"message": "Cancellation signal sent.", "application_id": app_id}
