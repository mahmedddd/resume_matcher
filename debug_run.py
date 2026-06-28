import subprocess
import time
import requests

print("Starting backend for debugging...")
proc = subprocess.Popen(["python", "-m", "uvicorn", "main:app", "--port", "8000"], cwd="backend", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

time.sleep(4)

print("Sending POST request...")
file_data = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
try:
    r = requests.post("http://127.0.0.1:8000/api/v1/cv/upload-cv", files={"file": ("dummy.pdf", file_data, "application/pdf")})
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Request failed: {e}")

proc.terminate()
time.sleep(1)

out, _ = proc.communicate()
print("\n--- BACKEND LOGS ---")
print(out)
print("--- END OF LOGS ---")
