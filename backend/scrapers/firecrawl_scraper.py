import axios from 'axios';
from typing import List, Dict, Any
from .base import BaseScraper
from config import settings

class FirecrawlScraper(BaseScraper):
    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Uses Firecrawl /search endpoint to find internship listings on the public web.
        """
        if not settings.FIRECRAWL_API_KEY:
            return []

        results = []
        import httpx
        
        async with httpx.AsyncClient() as client:
            for keyword in keywords:
                query = f"remote {keyword} Pakistan 2026"
                try:
                    response = await client.post(
                        "https://api.firecrawl.dev/v1/search",
                        headers={"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}"},
                        json={
                            "query": query,
                            "limit": 5,
                            "scrapeOptions": {"formats": ["markdown", "json"]}
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get("data", []):
                            results.append({
                                "title": item.get("title", keyword),
                                "url": item.get("url"),
                                "company": "Search Result",
                                "location": "Remote / Pakistan",
                                "source": "firecrawl",
                                "description": item.get("markdown", "")[:500]
                            })
                except Exception as e:
                    print(f"Firecrawl error for {keyword}: {e}")
                    
        return results
