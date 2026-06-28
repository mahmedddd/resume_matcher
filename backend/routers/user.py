from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db, UserProfile
from schemas import UserProfileSchema

router = APIRouter()

@router.get("", response_model=UserProfileSchema)
async def get_profile(db: AsyncSession = Depends(get_db)):
    profile = (await db.execute(select(UserProfile))).scalars().first()
    if not profile:
        return {}
    return profile

@router.post("")
async def update_profile(data: UserProfileSchema, db: AsyncSession = Depends(get_db)):
    profile = (await db.execute(select(UserProfile))).scalars().first()
    if not profile:
        profile = UserProfile(**data.dict())
        db.add(profile)
    else:
        for key, value in data.dict().items():
            setattr(profile, key, value)
    
    await db.commit()
    return {"message": "Profile updated successfully"}
