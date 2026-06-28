import asyncio
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from .base import BaseScraper
from config import settings

class IndeedScraper(BaseScraper):
    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        results = []
        
        async def fetch_desc(job: Dict[str, Any], context):
            url = job.get("url")
            if not url or url.startswith("javascript"):
                return
            new_page = await context.new_page()
            try:
                # Add tiny timeout to prevent stalling
                await new_page.goto(url, timeout=15000, wait_until="domcontentloaded")
                
                # Fetch typical description containers
                desc_el = await new_page.query_selector('#jobDescriptionText')
                if desc_el:
                    text_content = await desc_el.inner_text()
                    job["description"] = text_content.strip() if text_content else ""
                else:
                    body_el = await new_page.locator('body').inner_text()
                    job["description"] = body_el.strip()[:1500] if body_el else ""

                # Fallback for Company
                if job.get("company") == "N/A":
                    comp_header = await new_page.query_selector('div[data-company-name="true"]') or await new_page.query_selector('.jobsearch-CompanyReview--heading')
                    if comp_header:
                        comp_text = await comp_header.inner_text()
                        if comp_text:
                            job["company"] = comp_text.strip()
                
                # Fetch detailed salary if empty
                if not job.get("salary"):
                    salary_header = await new_page.query_selector('#salaryInfoAndJobType')
                    if salary_header:
                        sal_text = await salary_header.inner_text()
                        if sal_text:
                            job["salary"] = sal_text.strip()

            except Exception as e:
                print(f"[Indeed] Warning: Could not deep-fetch description for {url} - {e}")
            finally:
                await new_page.close()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=settings.SCRAPER_HEADLESS)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36")
            
            for keyword in keywords:
                formatted_keyword = "+".join(keyword.split())
                search_url = f"https://pk.indeed.com/jobs?q={formatted_keyword}&l=Pakistan"
                page = await context.new_page()
                try:
                    await page.goto(search_url, timeout=30000, wait_until="networkidle")
                    await page.evaluate("window.scrollBy(0, 1000)")
                    await asyncio.sleep(2)
                    
                    # Extract list items
                    job_elements = await page.query_selector_all(".resultContent")
                    if not job_elements:
                        job_elements = await page.query_selector_all(".job_seen_beacon")
                        
                    for job_el in job_elements[:10]: # cap at 10
                        title_elem = await job_el.query_selector("h2.jobTitle a") or await job_el.query_selector("h2 a")
                        comp_elem = await job_el.query_selector('[data-testid="company-name"]') or await job_el.query_selector(".companyName") or await job_el.query_selector('span[data-testid="company-name"]')
                        loc_elem = await job_el.query_selector('[data-testid="text-location"]') or await job_el.query_selector(".companyLocation") or await job_el.query_selector('div[data-testid="text-location"]')
                        salary_elem = await job_el.query_selector("div.salary-snippet-container") or await job_el.query_selector('div[data-testid="attribute_snippet_testid"]')
                        
                        if title_elem:
                            title = (await title_elem.inner_text()).strip()
                            url = await title_elem.get_attribute("href")
                            company = (await comp_elem.inner_text()).strip() if comp_elem else "N/A"
                            location = (await loc_elem.inner_text()).strip() if loc_elem else "Pakistan"
                            salary = (await salary_elem.inner_text()).strip() if salary_elem else None
                            
                            full_url = f"https://pk.indeed.com{url}" if url and url.startswith("/") else url
                            
                            results.append({
                                "title": title,
                                "url": full_url,
                                "company": company,
                                "location": location,
                                "source": "indeed",
                                "description": "",
                                "salary": salary
                            })
                except Exception as e:
                    print(f"Error scraping Indeed listings for {keyword}: {e}")
                finally:
                    await page.close()
            
            print(f"[Indeed] Deep-fetching descriptions for {len(results)} jobs...")
            tasks = [fetch_desc(job, context) for job in results]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            await browser.close()
            
        return results
