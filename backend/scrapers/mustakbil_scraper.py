import asyncio
import httpx
import random
from typing import List, Dict, Any
from .base import BaseScraper

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122 Safari/537.36",
]

class MustakbilScraper(BaseScraper):
    """
    Scrapes Mustakbil.com via HTTPX and BeautifulSoup.
    No login required, bypasses headless blocking.
    """
    
    BASE_URL = "https://mustakbil.com"

    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        results = []
        seen_urls = set()

        async with httpx.AsyncClient(follow_redirects=True, timeout=25.0) as client:
            for keyword in keywords:
                kw_encoded = keyword.replace(" ", "+")
                
                # Fetch up to 2 pages per keyword
                for page in range(1, 3):
                    ua = random.choice(_USER_AGENTS)
                    client.headers.update({"User-Agent": ua})
                    
                    search_url = f"{self.BASE_URL}/jobs?keyword={kw_encoded}&location=Pakistan&page={page}"
                    try:
                        resp = await client.get(search_url)
                        if resp.status_code != 200:
                            break
                            
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(resp.text, "html.parser")
                        
                        # Mustakbil uses article-card or simply anchors with href=/jobs/job/
                        job_links = [a for a in soup.select("a[href]") if "/jobs/job/" in a.get("href", "")]
                        
                        batch = []
                        for link in job_links:
                            href = link.get("href")
                            title = link.get_text(strip=True)
                            
                            if not href or not title or "View details" in title or len(title) < 5:
                                continue
                                
                            job_url = self.BASE_URL + href if href.startswith("/") else href
                            if job_url in seen_urls:
                                continue
                            seen_urls.add(job_url)
                            
                            batch.append({
                                "title": title,
                                "url": job_url,
                                "source": "mustakbil",
                                "company": "Pakistan Company", # Will fetch deep
                                "location": "Pakistan",
                                "description": ""
                            })
                            
                            # Limit to 15 per page to avoid overloading
                            if len(batch) >= 15:
                                break
                                
                        # Deep-fetch details in parallel
                        async def fetch_detail(job):
                            await asyncio.sleep(random.uniform(0.3, 1.0))
                            try:
                                client.headers.update({"User-Agent": random.choice(_USER_AGENTS)})
                                d_resp = await client.get(job["url"])
                                if d_resp.status_code == 200:
                                    d_soup = BeautifulSoup(d_resp.text, "html.parser")
                                    
                                    # Extract Company
                                    comp_el = d_soup.select_one(".company-name") or d_soup.select_one(".jd-hero__company")
                                    if comp_el:
                                        job["company"] = comp_el.get_text(strip=True)[:100]
                                        
                                    # Extract Description
                                    desc_el = d_soup.select_one(".job-description") or d_soup.select_one(".jd-content")
                                    if desc_el:
                                        job["description"] = desc_el.get_text(separator=" ", strip=True)[:2000]
                                        
                                    # Extract Salary
                                    salary_el = d_soup.select_one(".jd-stat--salary")
                                    if salary_el:
                                        job["salary"] = salary_el.get_text(strip=True)
                                        
                                    # Extract Location
                                    loc_el = d_soup.select_one(".jd-stat__icon--loc")
                                    if loc_el and loc_el.parent:
                                        job["location"] = loc_el.parent.get_text(strip=True)
                            except Exception:
                                pass
                            return job

                        enriched = await asyncio.gather(*[fetch_detail(j) for j in batch])
                        results.extend(enriched)
                        print(f"[Mustakbil] '{keyword}' page {page}: {len(batch)} listings")

                        # Stop paginating if no jobs found
                        if len(batch) == 0:
                            break
                            
                        await asyncio.sleep(random.uniform(1.5, 3.0))

                    except Exception as e:
                        print(f"[Mustakbil] Error for '{keyword}' p{page}: {e}")
                        break

        return results
