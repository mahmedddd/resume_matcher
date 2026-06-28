import asyncio
import random
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from .base import BaseScraper
from config import settings


_ROZEE_CITIES = [
    "Lahore", "Karachi", "Islamabad", "Rawalpindi", "Peshawar",
    "Faisalabad", "Multan", "Sialkot", "Gujranwala", "Quetta",
    "Hyderabad", "Abbottabad",
]

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/122 Safari/537.36",
]


class RozeeScraper(BaseScraper):
    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        results = []

        async def fetch_desc(job: Dict[str, Any], context):
            url = job.get("url")
            if not url or url.startswith("javascript"):
                return
            new_page = await context.new_page()
            try:
                await new_page.goto(url, timeout=18000, wait_until="domcontentloaded")
                await asyncio.sleep(random.uniform(0.5, 1.2))

                # Description containers (ordered by specificity)
                desc_text = ""
                for selector in [".job-desc", ".j-desc", ".job-details", ".job-description", "#job-description", ".jd-content"]:
                    desc_el = await new_page.query_selector(selector)
                    if desc_el:
                        desc_text = (await desc_el.inner_text()).strip()
                        break

                if not desc_text:
                    try:
                        desc_text = (await new_page.locator("body").inner_text())[:2000]
                    except Exception:
                        desc_text = ""

                job["description"] = desc_text[:2000]

                # --- Extract structured metadata ---
                try:
                    # Salary (PKR amounts)
                    import re
                    salary_match = re.search(r"(?:PKR|Rs\.?)\s*[\d,]+(?:\s*[-–]\s*[\d,]+)?(?:\s*(?:per\s+month|/month|pm))?", desc_text, re.IGNORECASE)
                    if salary_match:
                        job["salary"] = salary_match.group(0).strip()[:60]
                except Exception:
                    pass

                try:
                    # Deadline from page meta
                    for label_text in ["Apply By", "Last Date", "Deadline", "Closing Date"]:
                        deadline_label = new_page.get_by_text(label_text, exact=False)
                        if await deadline_label.count() > 0:
                            handle = await deadline_label.first.element_handle()
                            if handle:
                                val = await new_page.evaluate("(el) => el.parentElement ? el.parentElement.innerText : ''", handle)
                                if label_text.lower() in val.lower():
                                    parts = val.split(label_text)[-1].strip().split("\n")
                                    job["deadline"] = parts[0].strip()[:40] if parts else ""
                                    break
                except Exception:
                    pass

                try:
                    # Experience level
                    exp_label = new_page.get_by_text("Experience", exact=False)
                    if await exp_label.count() > 0:
                        handle = await exp_label.first.element_handle()
                        if handle:
                            val = await new_page.evaluate("(el) => el.parentElement ? el.parentElement.innerText : ''", handle)
                            if "Experience" in val:
                                job["experience"] = val.split("Experience")[-1].strip().split("\n")[0][:80]
                except Exception:
                    pass

                # --- Extract city from location element ---
                try:
                    for loc_sel in [".loc", ".j-loc", ".location", "[data-location]"]:
                        loc_el = await new_page.query_selector(loc_sel)
                        if loc_el:
                            loc_text = (await loc_el.inner_text()).strip()
                            if loc_text:
                                job["location"] = loc_text
                                break
                except Exception:
                    pass

            except Exception as e:
                print(f"[Rozee] Warning: Could not deep-fetch {url}: {e}")
            finally:
                await new_page.close()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=settings.SCRAPER_HEADLESS)
            context = await browser.new_context(
                user_agent=random.choice(_USER_AGENTS),
                viewport={"width": 1366, "height": 768},
                locale="en-US",
            )

            # Scrape keyword x city combinations for diversity
            search_combos = []
            for kw in keywords:
                search_combos.append((kw, None))  # generic search
            for city in _ROZEE_CITIES[:6]:  # top 6 cities
                search_combos.append((keywords[0], city))  # primary keyword per city

            seen_urls = set()

            for kw, city in search_combos:
                if city:
                    search_url = f"https://www.rozee.pk/job/jsearch/q/{kw.replace(' ', '-')}/ct/{city.lower()}"
                else:
                    search_url = f"https://www.rozee.pk/job/jsearch/q/{kw.replace(' ', '-')}/"

                page = await context.new_page()
                try:
                    await page.goto(search_url, timeout=30000, wait_until="networkidle")
                    await page.evaluate("window.scrollBy(0, 1200)")
                    await asyncio.sleep(random.uniform(1.0, 2.0))

                    # Try multiple card selectors
                    job_elements = (
                        await page.query_selector_all(".job")
                        or await page.query_selector_all(".j-box")
                        or await page.query_selector_all(".job-row")
                        or await page.query_selector_all("[class*='job']")
                    )

                    count = 0
                    for job_el in job_elements[:20]:
                        title_elem = (
                            await job_el.query_selector(".jobt a")
                            or await job_el.query_selector("h3 a")
                            or await job_el.query_selector("a[href*='/job/detail/']")
                            or await job_el.query_selector("a[href*='/job/']")
                        )
                        comp_elem = (
                            await job_el.query_selector(".comp-name")
                            or await job_el.query_selector(".job-dtl b")
                            or await job_el.query_selector(".j-comp")
                            or await job_el.query_selector("span.comp")
                        )
                        loc_elem = (
                            await job_el.query_selector(".job-dtl .loc")
                            or await job_el.query_selector(".loc")
                            or await job_el.query_selector(".j-loc")
                            or await job_el.query_selector("[class*='loc']")
                        )

                        if title_elem:
                            title = (await title_elem.inner_text()).strip()
                            href = await title_elem.get_attribute("href") or ""
                            job_url = f"https://www.rozee.pk{href}" if href.startswith("/") else href
                            if not job_url or job_url in seen_urls:
                                continue
                            seen_urls.add(job_url)

                            company = (await comp_elem.inner_text()).strip() if comp_elem else "N/A"
                            location = (await loc_elem.inner_text()).strip() if loc_elem else (city or "Pakistan")

                            results.append({
                                "title": title,
                                "url": job_url,
                                "company": company,
                                "location": location,
                                "source": "rozee",
                                "description": "",
                            })
                            count += 1

                    print(f"[Rozee] '{kw}'{(' in ' + city) if city else ''}: {count} listings found")

                except Exception as e:
                    print(f"[Rozee] Error scraping '{kw}'{(' in ' + city) if city else ''}: {e}")
                finally:
                    await page.close()

            # Deep-fetch descriptions in parallel batches of 5
            print(f"[Rozee] Deep-fetching {len(results)} job descriptions...")
            batch_size = 5
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                await asyncio.gather(*[fetch_desc(job, context) for job in batch], return_exceptions=True)
                await asyncio.sleep(random.uniform(0.5, 1.5))

            await browser.close()

        return results
