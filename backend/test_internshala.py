import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36",
}

async def test():
    url = "https://internshala.com/internships/software-developer-internship"
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True) as client:
        resp = await client.get(url)
        print(f"Status: {resp.status_code}")
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Test original selectors
        cards = soup.select(".individual_internship") or soup.select(".internship_meta")
        print(f"Cards found with .individual_internship or .internship_meta: {len(cards)}")
        
        for i, card in enumerate(cards[:3]):
            title_el = (
                card.select_one(".heading_4_5 a") or
                card.select_one(".job-internship-name") or
                card.select_one("a[href*='/internship/detail/']")
            )
            href = title_el.get("href", "") if title_el else "NOT FOUND"
            print(f"Card {i} href: {href}")

        # Test fallback
        links = soup.select("a[href*='/internship/detail/']")
        print(f"Links found with a[href*='/internship/detail/']: {len(links)}")
        for i, link in enumerate(links[:3]):
            print(f"Link {i} href: {link.get('href')}")

import asyncio
asyncio.run(test())
