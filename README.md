# 🎯 SkillSync PK

> AI-powered Pakistan internship & job discovery platform that automatically scrapes local job boards and matches them directly to your CV.

This platform leverages headless browsing and asynchronous scraping to pull live tech jobs across Pakistan, embedding them via local NLP models, and extracting CV skills via Gemini Flash LLM to find your perfect job match.

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| **Backend API** | FastAPI, Python 3.10+ |
| **Database** | SQLite + SQLAlchemy (Async) |
| **Scrapers** | Playwright (LinkedIn), HTTPX + BeautifulSoup (Mustakbil, Internshala) |
| **AI / NLP** | `sentence-transformers` (`all-MiniLM-L6-v2`), Gemini Flash LLM |
| **Frontend** | Next.js (React), TypeScript, Tailwind CSS / Vanilla CSS |

---

## ✨ Features

- **Multi-Source Local Scraping:** Seamlessly pulls jobs from Mustakbil, LinkedIn (bypassing anti-bot walls), and Internshala.
- **CV Smart Matching:** Upload a PDF CV, extract your unique skills via Google's Gemini, and map them to live job descriptions.
- **Semantic Search:** Uses advanced local text embedding (Sentence Transformers) to rank jobs not just by keyword, but by *meaning* and context.
- **Glassmorphic UI:** Modern, premium dashboard built on Next.js.
- **AI Agent Auto-Apply:** With 1-Click, our AI agent takes over, autonomously filling out tedious application forms for you across multiple sites.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- A free [Google Gemini API Key](https://ai.google.dev/)

### 2. Backend Setup
Navigate to the root directory and set up the Python environment:
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (required for LinkedIn scraper)
playwright install chromium

# Set up environment variables
copy .env.example .env
# Open .env and add your GEMINI_API_KEY

# Start the FastAPI backend
cd backend
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
Open a new terminal and set up the Next.js frontend:
```bash
cd frontend_next

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open your browser to [http://localhost:3000](http://localhost:3000) to access the platform.

---

## 📂 Architecture & Structure

```text
resume_matcher/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── database.py          # SQLite engine and ORM setup
│   ├── models.py            # Pydantic schemas and types
│   ├── routers/             # API routes (scraper, internships, cv)
│   ├── scrapers/            # Individual scraping logic (Mustakbil, LinkedIn, etc.)
│   └── nlp/                 # NLP pipelines (Embedder, Matcher, Gemini Skill Extraction)
├── frontend_next/
│   ├── src/app/             # Next.js App Router (Pages & Layouts)
│   ├── src/components/      # Reusable React components (Dashboard, CV Uploader)
│   └── src/lib/             # Axios API clients and utilities
├── requirements.txt         # Backend Python dependencies
├── .env.example             # Template for API keys
└── .gitignore               # Git exclusion rules
```

---

## 📝 License
This project is for educational and portfolio purposes. Data scraped belongs to the respective job boards.
