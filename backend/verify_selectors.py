import asyncio
from playwright.async_api import async_playwright

async def verify_rozee():
    print("\n--- Verifying Rozee Selectors ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = "https://www.rozee.pk/job/jsearch/q/software-intern/"
            await page.goto(url, timeout=30000, wait_until="networkidle")
            
            job = await page.query_selector(".job")
            if job:
                print("✅ Found .job")
                title_el = await job.query_selector(".jobt")
                if title_el:
                    print(f"Title text: {await title_el.inner_text()}")
                
                details = await job.query_selector(".job-dtl")
                if details:
                    print(f"Details text: {await details.inner_text()}")
            else:
                print("❌ Could NOT find .job")
                # Try .jbox just in case
                jbox = await page.query_selector(".jbox")
                if jbox: print("✅ Found .jbox (no dot in name?)")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

async def verify_mustakbil():
    print("\n--- Verifying Mustakbil Selectors ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = "https://www.mustakbil.com/jobs/search?q=software-intern"
            await page.goto(url, timeout=30000, wait_until="networkidle")
            
            # The previous analysis showed .chbx (168)
            item = await page.query_selector(".chbx")
            if item:
                print("✅ Found .chbx")
                print(f"Content: {(await item.inner_text())[:200]}")
            
            # Try others
            item2 = await page.query_selector(".job-item")
            if item2: print("✅ Found .job-item")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

async def main():
    await verify_rozee()
    await verify_mustakbil()

if __name__ == "__main__":
    asyncio.run(main())
