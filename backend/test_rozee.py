import asyncio
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        page = await context.new_page()
        
        keyword = "software"
        search_url = f"https://www.rozee.pk/job/jsearch/q/{keyword}/"
        print(f"Navigating to {search_url}")
        
        try:
            await page.goto(search_url, timeout=30000, wait_until="networkidle")
            print(f"Page Title: {await page.title()}")
            
            # Check for current selectors
            job_elements = await page.query_selector_all(".job")
            print(f"Jobs found with .job: {len(job_elements)}")
            
            if len(job_elements) == 0:
                job_elements = await page.query_selector_all(".j-box text-container")
                print(f"Jobs found with .j-box text-container: {len(job_elements)}")
                
            for i, job_el in enumerate(job_elements[:3]):
                title_elem = await job_el.query_selector(".jobt a") or \
                             await job_el.query_selector("h3 a") or \
                             await job_el.query_selector("a[href*='/job/detail/']")
                if title_elem:
                    href = await title_elem.get_attribute("href")
                    text = await title_elem.inner_text()
                    print(f"Job {i}: {text.strip()} | {href}")
                else:
                    print(f"Job {i}: Title element NOT FOUND")
            
            # Check for generic a tags with detail in them
            links = await page.query_selector_all("a[href*='/job/detail/']")
            print(f"Generic links found with a[href*='/job/detail/']: {len(links)}")
            for i, link in enumerate(links[:3]):
                print(f"Link {i}: {await link.get_attribute('href')}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

asyncio.run(test())
