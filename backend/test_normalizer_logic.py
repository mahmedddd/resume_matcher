from scrapers.normalizer import normalize

test_cases = [
    {"title": "Dev", "location": "India", "source": "internshala"},
    {"title": "Dev", "location": "Bangalore", "source": "linkedin"},
    {"title": "Dev", "location": "Lahore", "source": "rozee"},
    {"title": "Dev", "location": "Karachi, Pakistan", "source": "linkedin"},
    {"title": "Dev", "location": "Pakistan", "source": "linkedin"},
]

print("=== NORMALIZER TESTS ===")
for case in test_cases:
    result = normalize(case)
    print(f"Location: {case['location']:20} | Source: {case['source']:12} -> City: {result['city']:10} | Remote: {result['is_remote']}")
