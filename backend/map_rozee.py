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
                # Find all sub-elements and print their class and text
                elements = await job.query_selector_all("*")
                for el in elements:
                    cls = await el.get_attribute("class")
                    if cls:
                        text = (await el.inner_text()).strip()
                        if text and len(text) < 100:
                            print(f"  .{cls}: {text}")
            else:
                print("NOT FOUND")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

asyncio.run(test())
