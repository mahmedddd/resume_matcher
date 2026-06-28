import asyncio
import random
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from .base import BaseScraper
from config import settings

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

_PAKISTAN_GEO_ID = "103469529"

class LinkedInScraper(BaseScraper):
    """
    Scrapes LinkedIn Pakistan job listings using Playwright.
    Bypasses the HTTP 999 block that httpx receives by using a real browser context.
    """

    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        results = []

        async def fetch_detail(job: Dict[str, Any], context):
            url = job.get("url")
            if not url:
                return
            new_page = await context.new_page()
            try:
                # Add delay to avoid aggressive rate limiting
                await asyncio.sleep(random.uniform(0.5, 1.5))
                resp = await new_page.goto(url, timeout=20000, wait_until="domcontentloaded")
                
                if resp and resp.status == 200:
                    html = await new_page.content()
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Extract description
                    desc_el = (
                        soup.select_one(".show-more-less-html__markup")
                        or soup.select_one(".description__text")
                        or soup.select_one(".job-view-layout")
                    )
                    job["description"] = desc_el.get_text(separator=" ", strip=True)[:2000] if desc_el else ""

                    # Company name
                    if job["company"] == "LinkedIn Company":
                        comp_detail = soup.select_one(".topcard__org-name-link") or soup.select_one(".company-name")
                        if comp_detail:
                            job["company"] = comp_detail.get_text(strip=True)

                    # Location
                    loc_detail = soup.select_one(".topcard__flavor--bullet") or soup.select_one(".job-details-jobs-unified-top-card__bullet")
                    if loc_detail:
                        job["location"] = loc_detail.get_text(strip=True)
                else:
                    job["description"] = ""
            except Exception:
                job["description"] = ""
            finally:
                await new_page.close()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=settings.SCRAPER_HEADLESS)
            context = await browser.new_context(
                user_agent=random.choice(_USER_AGENTS),
                viewport={"width": 1920, "height": 1080}
            )

            for keyword in keywords:
                kw_encoded = keyword.replace(" ", "%20")
                
                # Fetch up to 2 pages (50 jobs) per keyword
                for page_start in [0, 25]:
                    page = await context.new_page()
                    try:
                        # Don't double specify Pakistan if it's already in the keyword
                        search_kw = kw_encoded.replace("Pakistan", "").replace("%20%20", "%20").strip("%20")
                        
                        url = (
                            f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
                            f"?keywords={search_kw}"
                            f"&location=Pakistan"
                            f"&geoId={_PAKISTAN_GEO_ID}"
                            f"&f_TPR=r2592000"  # Past month
                            f"&start={page_start}"
                        )
                        
                        resp = await page.goto(url, timeout=25000)
                        print(f"DEBUG: LinkedIn URL {url} -> Status {resp.status if resp else 'None'}")
                        if not resp or resp.status != 200:
                            await page.close()
                            break

                        html = await page.content()
                        print(f"DEBUG: HTML length {len(html)}")
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, "html.parser")
                        
                        cards = soup.select(".base-card") or soup.select(".job-search-card") or soup.select("li")
                        print(f"DEBUG: LinkedIn cards found: {len(cards)}")
                        if not cards:
                            await page.close()
                            break

                        batch = []
                        for card in cards:
                            title_el = card.select_one(".base-search-card__title") or card.select_one("h3")
                            comp_el = card.select_one(".base-search-card__subtitle") or card.select_one("h4")
                            loc_el = card.select_one(".job-search-card__location")
                            link_el = card.select_one("a[href*='/jobs/view/']") or card.select_one("a")
                            
                            if title_el and link_el:
                                job_url = link_el.get("href", "").split("?")[0]
                                batch.append({
                                    "title": title_el.get_text(strip=True),
                                    "url": job_url,
                                    "company": comp_el.get_text(strip=True) if comp_el else "LinkedIn Company",
                                    "location": loc_el.get_text(strip=True) if loc_el else "Pakistan",
                                    "source": "linkedin",
                                    "description": ""
                                })

                        print(f"[LinkedIn] '{keyword}' page@{page_start}: {len(batch)} jobs")
                        
                        # Add to main results
                        results.extend(batch)
                        
                        await page.close()
                        
                        if len(batch) < 10:
                            break  # No more full pages
                            
                    except Exception as e:
                        print(f"[LinkedIn] Error for '{keyword}' page@{page_start}: {e}")
                        await page.close()
                        break

            # Deep fetch details in batches of 5 to not trigger blocks
            print(f"[LinkedIn] Deep-fetching {len(results)} jobs...")
            batch_size = 5
            for i in range(0, len(results), batch_size):
                batch_jobs = results[i:i + batch_size]
                await asyncio.gather(*[fetch_detail(job, context) for job in batch_jobs])
                await asyncio.sleep(random.uniform(1.0, 2.0))

            await browser.close()

        return results
