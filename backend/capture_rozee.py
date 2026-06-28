import asyncio
from playwright.async_api import async_playwright

async def test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = "https://www.rozee.pk/job/jsearch/q/software-intern/"
            await page.goto(url, timeout=30000, wait_until="networkidle")
            job = await page.query_selector(".job")
            if job:
                print(await job.inner_html())
            else:
                print("NOT FOUND")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

asyncio.run(test())
