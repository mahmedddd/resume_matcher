import subprocess
import os
import sys
import time

def start_services():
    print("🚀 Starting Resume Matcher...")
    
    # 1. Start Backend
    print("📦 Starting Backend (FastAPI)...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"],
        cwd=os.path.join(os.getcwd(), "backend")
    )
    
    # 2. Wait for backend to warm up
    time.sleep(3)
    
    # 3. Start Frontend
    print("🌐 Starting Frontend (Next.js)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=os.path.join(os.getcwd(), "frontend_next"),
        shell=True
    )

    # 4. Start ngrok
    print("🔗 Skipping ngrok (running on localhost only)...")
    # ngrok_domain = "pisciform-unmonumental-bently.ngrok-free.dev"
    # ngrok_proc = subprocess.Popen(
    #     ["ngrok", "http", "3000", f"--domain={ngrok_domain}"],
    #     cwd=os.getcwd(),
    #     shell=True
    # )
    
    print("\n✅ Services are running!")
    print("👉 Backend: http://localhost:8000")
    print("👉 Frontend: http://localhost:3000")
    # print(f"🌍 Remote Access (ngrok): https://{ngrok_domain}")
    print("\nPress Ctrl+C to stop all services.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        # ngrok_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    start_services()
