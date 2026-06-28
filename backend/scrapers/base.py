import abc
from typing import List, Dict, Any

class BaseScraper(abc.ABC):
    @abc.abstractmethod
    async def scrape(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape internships based on keywords.
        Returns a list of raw internship dictionaries.
        """
        pass
