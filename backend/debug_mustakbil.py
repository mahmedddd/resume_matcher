import asyncio
from playwright.async_api import async_playwright

async def debug_mustakbil():
    print("\n--- Deep Debug Mustakbil ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            url = "https://www.mustakbil.com/jobs/search?q=software-intern"
            await page.goto(url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(5)
            
            # Print all div classes
            classes = await page.evaluate("() => Array.from(document.querySelectorAll('div')).map(d => d.className)")
            from collections import Counter
            counts = Counter([c for c in classes if c])
            print("Frequent DIV classes:")
            for c, f in counts.most_common(15):
                print(f"  .{c}: {f}")
                
            # Specifically check for common job board card classes
            targets = [".job-item", ".card", ".list-item", ".row", ".chbx"]
            for t in targets:
                count = await page.locator(t).count()
                print(f"Count of '{t}': {count}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

asyncio.run(debug_mustakbil())
