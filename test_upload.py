import requests, io
file_data = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
files = {"file": ("dummy.pdf", file_data, "application/pdf")}
try:
    print("Sending POST request to upload-cv...")
    r = requests.post("http://127.0.0.1:8000/api/v1/cv/upload-cv", files=files)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.text}")
except Exception as e:
    print(f"Connection Error: {e}")
