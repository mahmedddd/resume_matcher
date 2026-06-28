"""
App-wide settings loaded from .env
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    FIRECRAWL_API_KEY: str = ""
    DATABASE_URL: str = "sqlite+aiosqlite:///./pakinter.db"
    FRONTEND_URL: str = "http://localhost:5173"
    SCRAPER_HEADLESS: bool = True
    SCRAPER_MAX_PAGES: int = 3

    model_config = {"env_file": "../.env", "extra": "ignore"}


settings = Settings()
