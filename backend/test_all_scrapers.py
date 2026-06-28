import asyncio
import sys
sys.path.insert(0, ".")

async def main():
    from scrapers.internshala_scraper import IntershalaScraper
    from scrapers.mustakbil_scraper import MustakbilScraper
    from scrapers.linkedin_scraper import LinkedInScraper
    from scrapers.rozee_scraper import RozeeScraper

    r, i, m, l = await asyncio.gather(
        RozeeScraper().scrape(["software intern"]),
        IntershalaScraper().scrape(["software development"]),
        MustakbilScraper().scrape(["software intern"]),
        LinkedInScraper().scrape(["software intern"]),
    )
    print(f"Rozee:{len(r)}  Internshala:{len(i)}  Mustakbil-fallback:{len(m)}  LinkedIn:{len(l)}")
    for src, res in [("Rozee", r), ("Internshala", i), ("Mustakbil", m), ("LinkedIn", l)]:
        if res:
            print(f"  {src}[0]: {res[0]['title']} | {res[0]['company']}")
        else:
            print(f"  {src}: EMPTY")

asyncio.run(main())
