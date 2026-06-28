import asyncio
from playwright.async_api import async_playwright
from collections import Counter

async def analyze_selectors(name, url):
    print(f"\n--- Analyzing Selectors for {name} ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        try:
            await page.goto(url, timeout=30000, wait_until="networkidle")
            await asyncio.sleep(3)
            
            # Get all classes in the body
            classes = await page.evaluate("""() => {
                const results = [];
                document.querySelectorAll('*').forEach(el => {
                    if (el.className && typeof el.className === 'string') {
                        results.push(...el.className.split(/\\s+/));
                    }
                });
                return results;
            }""")
            
            count = Counter(classes)
            print("Top 20 most frequent classes:")
            for cls, freq in count.most_common(20):
                print(f"  .{cls}: {freq}")
                
            # Specifically check for job-like keywords in classes
            job_classes = [cls for cls in count if "job" in cls.lower() or "box" in cls.lower() or "item" in cls.lower() or "card" in cls.lower()]
            print("\nJob-related classes found:")
            for cls in job_classes:
                if count[cls] > 2: # Likely a list item
                    print(f"  .{cls}: {count[cls]}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

async def main():
    await analyze_selectors("Rozee", "https://www.rozee.pk/job/jsearch/q/software-intern/")
    await analyze_selectors("Mustakbil", "https://www.mustakbil.com/jobs/search?q=software-intern")

if __name__ == "__main__":
    asyncio.run(main())
