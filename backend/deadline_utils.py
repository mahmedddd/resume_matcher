"""
deadline_utils.py
-----------------
Utilities for parsing flexible human-readable deadline strings into
comparable datetime objects for filtering expired internships.

Supports the following formats extracted from Pakistani job boards:
  - "June 30, 2025"        (Month DD, YYYY)
  - "30 June 2025"         (DD Month YYYY)
  - "30/06/2025"           (DD/MM/YYYY)
  - "2025-06-30"           (YYYY-MM-DD)
  - "30 Jun"               (DD Mon — year assumed current/next)
  - "Jun 30"               (Mon DD — year assumed current/next)
  - "30/06/25"             (DD/MM/YY)
"""

import re
from datetime import date, datetime
from typing import Optional


# All strptime format patterns to attempt in order
_FORMATS = [
    "%B %d, %Y",   # June 30, 2025
    "%d %B %Y",    # 30 June 2025
    "%b %d, %Y",   # Jun 30, 2025
    "%d %b %Y",    # 30 Jun 2025
    "%d/%m/%Y",    # 30/06/2025
    "%Y-%m-%d",    # 2025-06-30
    "%m/%d/%Y",    # 06/30/2025
    "%d-%m-%Y",    # 30-06-2025
    "%d/%m/%y",    # 30/06/25
    "%d %B",       # 30 June   (no year — assume current/next)
    "%d %b",       # 30 Jun    (no year)
    "%B %d",       # June 30   (no year)
    "%b %d",       # Jun 30    (no year)
]


def parse_deadline(deadline_str: str | None) -> Optional[date]:
    """
    Try to parse a deadline string into a Python date.
    Returns None if unparseable.
    """
    if not deadline_str:
        return None

    # Clean up common noise
    text = deadline_str.strip().rstrip(".")
    # Normalise separators
    text = re.sub(r"[-–—]", "/", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    today = date.today()

    for fmt in _FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
            # If no year was embedded in the format, use the current or next year
            if "%Y" not in fmt and "%y" not in fmt:
                parsed = parsed.replace(year=today.year)
                # If that date has already passed this year, try next year
                if parsed < today:
                    parsed = parsed.replace(year=today.year + 1)
            return parsed
        except ValueError:
            continue

    return None


def deadline_is_valid(deadline_str: str | None, reference: date | None = None) -> bool:
    """
    Returns True if:
      - The deadline cannot be parsed (keep the job — better safe than sorry)
      - The deadline is today or in the future

    Returns False (filter OUT) if:
      - The deadline has clearly already passed
    """
    if not deadline_str:
        return True  # No deadline info -> keep the job

    ref = reference or date.today()
    parsed = parse_deadline(deadline_str)

    if parsed is None:
        return True  # Unparseable -> keep the job

    return parsed >= ref  # True = valid, False = expired
