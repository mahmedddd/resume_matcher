import re
from typing import Dict, Any
from datetime import datetime


# Common deadline indicator patterns found on Pakistani job boards
_DEADLINE_PATTERNS = [
    # "Apply by June 30, 2025" / "Apply before June 30"
    r"(?:apply\s+(?:by|before|on)|deadline[:\s]+|last\s+date[:\s]+|closing\s+date[:\s]+|applications?\s+close[s]?\s+(?:on|by)?)\s*[:\-]?\s*(\d{1,2}[\/\-\s]\d{1,2}[\/\-\s]\d{2,4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+\w+)",
    # "30 June 2025" or "June 30 2025" standalone
    r"\b(\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{2,4})\b",
    # "30/06/2025" or "2025-06-30"
    r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b",
]

_SALARY_PATTERNS = [
    # "PKR 50,000 – 100,000" / "50k - 80k" / "Rs. 40,000"
    r"(?:PKR|Rs\.?|Salary[:\s]+)\s*[\d,k]+(?:\s*[-–]\s*[\d,k]+)?(?:\s*(?:per\s+month|/month|pm))?",
    r"\b\d+k\s*[-–]\s*\d+k\b",
    r"\$\s*\d+(?:,\d+)?(?:\s*[-–]\s*\$?\s*\d+(?:,\d+)?)?",
]

# Skills keywords commonly found in job descriptions
_SKILL_KEYWORDS = [
    "python", "javascript", "typescript", "react", "next.js", "vue.js", "angular", "node.js",
    "django", "flask", "fastapi", "java", "kotlin", "swift", "flutter", "dart", "c++", "c#",
    ".net", "go", "golang", "rust", "php", "laravel", "wordpress",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "docker", "kubernetes", "aws", "azure", "gcp", "linux", "bash", "git", "github",
    "html", "css", "tailwind", "bootstrap", "sass",
    "machine learning", "deep learning", "tensorflow", "pytorch", "nlp", "computer vision",
    "data analysis", "data science", "pandas", "numpy", "scikit-learn",
    "excel", "power bi", "tableau", "looker",
    "figma", "adobe xd", "photoshop", "illustrator", "canva", "ui/ux",
    "seo", "digital marketing", "google ads", "facebook ads",
    "content writing", "graphic design",
    "android", "ios", "firebase", "rest api", "graphql",
    "blockchain", "solidity", "web3",
    "cybersecurity", "penetration testing", "networking",
    "unity", "unreal engine", "game development",
    "tensorflow", "keras", "hugging face", "langchain",
]

# Known Pakistani cities (lowercase for matching)
_PK_CITIES = [
    "lahore", "karachi", "islamabad", "rawalpindi", "peshawar",
    "faisalabad", "multan", "quetta", "hyderabad", "gujranwala",
    "sialkot", "bahawalpur", "sargodha", "gujrat", "sheikhupura",
    "abbottabad", "rahim yar khan", "mardan", "sukkur", "larkana",
    "dera ghazi khan", "sahiwal", "gwadar", "mirpur", "muzaffarabad",
    "attock", "chakwal", "jhelum", "narowal", "hafizabad",
]

# Generic location strings that should NOT be treated as foreign (-> not forced remote)
_GENERIC_LOCATIONS = {
    "pakistan", "pk", "multiple locations", "multiple", "", "n/a",
    "anywhere", "all", "nationwide", "all over pakistan",
}


def _extract_deadline(text: str) -> str | None:
    """Try to find a deadline date string in a block of text."""
    if not text:
        return None
    text_lower = text.lower()
    for pattern in _DEADLINE_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            raw = match.group(1).strip().title()
            return raw[:40]  # cap length
    return None


def _extract_salary(text: str) -> str | None:
    """Try to find a salary mention in a block of text."""
    if not text:
        return None
    for pattern in _SALARY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()[:60]
    return None


def _extract_skills_from_text(text: str) -> list[str]:
    """Extract known tech skills from job description text."""
    if not text:
        return []
    text_lower = text.lower()
    found = []
    for skill in _SKILL_KEYWORDS:
        # Use word boundary matching for short skills to avoid false positives
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill.title() if len(skill) <= 4 else skill)
    return list(dict.fromkeys(found))  # deduplicate preserving order


def normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalises raw job data from different scrapers into a unified schema.
    Automatically extracts deadline, salary and skills from descriptions if not provided.
    """
    location = raw.get("location", "Pakistan").strip().lower()
    source = raw.get("source", "").lower()
    description = raw.get("description", "") or ""

    # --- City detection ---
    detected_city = "Pakistan"   # Sensible default (shows in All filter, not hidden)
    is_remote = False

    # Explicit remote signals in location string
    if "remote" in location or "work from home" in location or "wfh" in location:
        is_remote = True

    # Internshala is India-based, treat as remote unless a PK city is found
    if source == "internshala":
        is_remote = True

    # jobs_pk and rozee are Pakistan-native - never override to remote
    _pk_native_sources = {"rozee", "jobs_pk", "mustakbil"}
    is_pk_native = source in _pk_native_sources
    city_found = False
    
    # Check location, then title, then description for city mentions
    text_to_search = f"{location} {raw.get('title', '')} {description}".lower()
    
    for city in _PK_CITIES:
        if city in text_to_search:
            detected_city = city.capitalize()
            city_found = True
            # Revert to specific remote status if a true Pakistani city is matched
            is_remote = "remote" in location or "work from home" in location
            break

    # If it's a specific non-Pakistani city, strictly treat it as remote.
    # We consider it a specific non-Pakistani city if we didn't find a pk city,
    # and the location name isn't generic/Pakistan.
    generic_locations = ["pakistan", "pk", "multiple locations", "multiple", "", "n/a", "anywhere", "all"]
    if not city_found and not is_pk_native and not any(loc == location.strip() for loc in generic_locations) and not ("pakistan" in location or "pk" in location.split()):
        is_remote = True

    # For PK-native sources, default city is the source location, not Remote
    if not city_found and is_pk_native:
        detected_city = "Pakistan"
        is_remote = False

    if is_remote:
        detected_city = "Remote"

    description = raw.get("description", "") or ""

    # Try raw deadline first, fallback to regex extraction from description
    deadline = raw.get("deadline") or _extract_deadline(description)
    salary   = raw.get("salary")   or _extract_salary(description)
    # Extract skills from description if not provided
    skills   = raw.get("skills_required") or _extract_skills_from_text(description)

    return {
        "title":           raw.get("title", "Unknown Title"),
        "company":         raw.get("company", "Unknown Company"),
        "city":            detected_city,
        "is_remote":       is_remote,
        "description":     description,
        "skills_required": skills,
        "url":             raw.get("url"),
        "source":          raw.get("source"),
        "deadline":        deadline,
        "salary":          salary,
    }
