"""
jobs.com.pk scraper - Pakistan's major job board
Scrapes using httpx + BeautifulSoup (no login needed).
"""
import asyncio
import random
import httpx
from typing import List, Dict, Any
from .base import BaseScraper


_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/123 Safari/537.36",
]


class JobsPkScraper(BaseScraper):
    """
    Scrapes jobs.com.pk - Pakistan's #1 job marketplace.
    Exclusively Pakistan-based companies and positions.
    """

    BASE_URL = "https://www.jobs.com.pk"

    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        results = []
        seen_urls = set()

        async with httpx.AsyncClient(
            headers={"User-Agent": random.choice(_USER_AGENTS)},
            follow_redirects=True,
            timeout=25.0,
        ) as client:
            for keyword in keywords:
                kw_slug = keyword.replace(" ", "+")
                url = f"{self.BASE_URL}/search/?q={kw_slug}&l=Pakistan"

                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        print(f"[JobsPK] {keyword}: HTTP {resp.status_code}")
                        continue

                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")

                    # Try multiple card selectors
                    cards = (
                        soup.select(".job-listing")
                        or soup.select(".job-post")
                        or soup.select(".j-listing")
                        or soup.select("article.job")
                        or soup.select("[class*='job-item']")
                        or soup.select(".search-result")
                    )

                    if not cards:
                        # Fallback: find all job links
                        cards = soup.select("a[href*='/job/']")

                    batch = []
                    for card in cards[:20]:
                        try:
                            title_el = card.select_one("h2") or card.select_one("h3") or card.select_one(".title") or card.select_one("a")
                            comp_el = card.select_one(".company") or card.select_one(".employer") or card.select_one("[class*='company']")
                            loc_el = card.select_one(".location") or card.select_one(".city") or card.select_one("[class*='location']")
                            link_el = card.select_one("a[href]") or (card if card.name == "a" else None)

                            if not title_el or not link_el:
                                continue

                            title = title_el.get_text(strip=True)
                            href = link_el.get("href", "")
                            job_url = href if href.startswith("http") else f"{self.BASE_URL}{href}"

                            if not job_url or job_url in seen_urls or len(title) < 3:
                                continue
                            seen_urls.add(job_url)

                            batch.append({
                                "title": title,
                                "url": job_url,
                                "company": comp_el.get_text(strip=True) if comp_el else "Pakistan Company",
                                "location": loc_el.get_text(strip=True) if loc_el else "Pakistan",
                                "source": "jobs_pk",
                                "description": "",
                            })
                        except Exception:
                            continue

                    print(f"[JobsPK] '{keyword}': {len(batch)} listings")

                    # Enrich with descriptions
                    async def fetch_desc(job):
                        await asyncio.sleep(random.uniform(0.3, 0.8))
                        try:
                            d_resp = await client.get(job["url"])
                            if d_resp.status_code == 200:
                                d_soup = BeautifulSoup(d_resp.text, "html.parser")
                                desc_el = (
                                    d_soup.select_one(".job-description")
                                    or d_soup.select_one(".description")
                                    or d_soup.select_one("#job-desc")
                                    or d_soup.select_one("[class*='description']")
                                    or d_soup.select_one("main")
                                )
                                if desc_el:
                                    job["description"] = desc_el.get_text(separator=" ", strip=True)[:2000]

                                # Get company name from detail if missing
                                if job["company"] == "Pakistan Company":
                                    comp_detail = d_soup.select_one("[class*='company']") or d_soup.select_one("h2")
                                    if comp_detail:
                                        job["company"] = comp_detail.get_text(strip=True)[:100]
                        except Exception:
                            pass
                        return job

                    enriched = await asyncio.gather(*[fetch_desc(j) for j in batch])
                    results.extend(enriched)

                except Exception as e:
                    print(f"[JobsPK] Error for '{keyword}': {e}")

                await asyncio.sleep(random.uniform(1.0, 2.0))

        return results
