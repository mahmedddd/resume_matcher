import sys
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

import nlp.cv_parser
nlp.cv_parser.parse_cv_pdf = lambda path: "I am a software engineer with 5 years of Python experience."
print("Sending POST request to /api/v1/cv/upload-cv...")
file_data = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
try:
    response = client.post("/api/v1/cv/upload-cv", files={"file": ("dummy.pdf", file_data, "application/pdf")})
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    import traceback
    traceback.print_exc()
