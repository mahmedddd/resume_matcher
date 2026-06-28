import asyncio
from playwright.async_api import async_playwright

async def probe_url(name, url):
    print(f"\nProbing {name}: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        try:
            response = await page.goto(url, timeout=20000)
            print(f"  Status: {response.status}")
            print(f"  Final URL: {page.url}")
            if response.status == 200:
                text = await page.locator("body").inner_text()
                if "software" in text.lower() or "intern" in text.lower():
                    print("  ✅ Body contains keywords!")
                else:
                    print("  ❌ Body does NOT contain keywords.")
        except Exception as e:
            print(f"  Error: {e}")
        finally:
            await browser.close()

async def main():
    # Internshala hyphen check
    await probe_url("Internshala (Hyphen)", "https://internshala.com/internships/keywords-software-intern")
    # Mustakbil hyphen check
    await probe_url("Mustakbil (Hyphen)", "https://www.mustakbil.com/jobs/search-software-intern")

if __name__ == "__main__":
    asyncio.run(main())
