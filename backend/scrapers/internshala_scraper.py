import asyncio
import httpx
from bs4 import BeautifulSoup
from typing import List, Dict, Any
from .base import BaseScraper


class IntershalaScraper(BaseScraper):
    """
    Scrapes Internshala using their public search pages + BeautifulSoup.
    Extracts deadline and stipend directly from listing cards.
    """

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://internshala.com/",
    }

    @staticmethod
    def _extract_meta(card) -> Dict[str, Any]:
        """Extract deadline and stipend from internship meta items on the card."""
        deadline = None
        salary = None
        for item in card.select(".other_detail_item, .internship_other_details_container span"):
            label_el = item.select_one(".item_heading, .detail-header")
            value_el = item.select_one(".item_body, .detail-body")
            if not label_el or not value_el:
                continue
            label = label_el.get_text(strip=True).lower()
            value = value_el.get_text(strip=True)
            if "apply by" in label or "deadline" in label:
                deadline = value
            elif "stipend" in label or "salary" in label or "ctc" in label:
                salary = value
        return {"deadline": deadline, "salary": salary}

    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        results = []
        async with httpx.AsyncClient(headers=self.HEADERS, follow_redirects=True, timeout=25.0) as client:
            for keyword in keywords:
                try:
                    kw_slug = keyword.lower().replace(" ", "-")
                    url = f"https://internshala.com/internships/{kw_slug}-internship"
                    resp = await client.get(url)

                    if resp.status_code != 200:
                        print(f"[Internshala] Got {resp.status_code} for {url}")
                        continue

                    soup = BeautifulSoup(resp.text, "html.parser")

                    # Modern Internshala uses a different structure or loads items dynamically
                    # We'll use a more resilient approach by finding the detail links directly
                    detail_links = soup.select("a[href*='/internship/detail/']")
                    
                    if not detail_links:
                        print(f"[Internshala] No detail links found for {keyword}")
                        continue

                    # Process unique links to avoid duplicates
                    seen_hrefs = set()
                    for link in detail_links:
                        href = link.get("href", "")
                        if not href or href in seen_hrefs:
                            continue
                        seen_hrefs.add(href)
                        
                        # Find the container this link belongs to if possible
                        # Usually the link is inside a div that contains the company name too
                        container = link.find_parent("div", class_="individual_internship") or \
                                    link.find_parent("div", class_="internship_meta") or \
                                    link.find_parent("div")
                        
                        title = link.get_text(strip=True) or "Internship"
                        
                        # Try to find company name in parent container
                        company = "Internshala"
                        if container:
                            comp_el = container.select_one(".company_name") or container.select_one(".heading_6")
                            if comp_el:
                                company = comp_el.get_text(strip=True)
                        
                        meta = self._extract_meta(container) if container else {"deadline": None, "salary": None}
                        
                        results.append({
                            "title": title,
                            "url": f"https://internshala.com{href}" if href.startswith("/") else href,
                            "company": company,
                            "location": "Remote / Pakistan",
                            "source": "internshala",
                            "description": "",
                            "deadline": meta["deadline"],
                            "salary": meta["salary"],
                        })
                        
                        if len(results) >= 40: # Cap per keyword
                            break
                    continue

                except Exception as e:
                    print(f"[Internshala] Error for {keyword}: {e}")

            # Concurrently fetch descriptions
            async def fetch_desc(job):
                url = job.get("url")
                if not url: return
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        s = BeautifulSoup(r.text, "html.parser")
                        desc_div = s.select_one(".text-container") or s.select_one(".detail_view") or s.select_one(".internship_details")
                        if desc_div:
                            text_content = desc_div.get_text(separator='\n', strip=True)
                            job["description"] = text_content[:2000] if text_content else ""
                except Exception:
                    pass

            print(f"[Internshala] Deep-fetching descriptions for {len(results)} jobs...")
            tasks = [fetch_desc(job) for job in results]
            await asyncio.gather(*tasks, return_exceptions=True)

        return results
