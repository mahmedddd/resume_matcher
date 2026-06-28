import asyncio
from playwright.async_api import async_playwright

async def check_search_form(name, url):
    print(f"\n--- Checking {name} ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        try:
            print(f"Navigating to {url}...")
            await page.goto(url, timeout=30000, wait_until="networkidle")
            
            # Look for search input and form
            search_input = await page.query_selector("input[type='text'], input[placeholder*='search' i]")
            if search_input:
                placeholder = await search_input.get_attribute("placeholder")
                print(f"Found search input: {placeholder}")
                
                # Try to find the form action
                form = await page.query_selector("form")
                if form:
                    action = await form.get_attribute("action")
                    method = await form.get_attribute("method")
                    print(f"Form action: {action}, method: {method}")
            
            # Just do a quick search and see where it lands
            if search_input:
                await search_input.fill("software intern")
                await page.keyboard.press("Enter")
                await page.wait_for_load_state("networkidle")
                print(f"Final URL after search: {page.url}")
                
                # Check results
                content = await page.content()
                print(f"Content length: {len(content)}")
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

async def main():
    await check_search_form("Rozee", "https://www.rozee.pk")
    await check_search_form("Mustakbil", "https://www.mustakbil.com")

if __name__ == "__main__":
    asyncio.run(main())
