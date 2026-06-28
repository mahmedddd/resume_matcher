import os
import json
import asyncio
import aiohttp
from models import CVSkills

async def extract_skills(text: str) -> CVSkills:
    """
    Uses Groq LLaMA 3.3 for blazing fast deterministic extraction.
    """
    print(f"[DEBUG] Extracting skills from CV text (first 200 chars): {text[:200].strip()}...", flush=True)
    prompt = f"""
    You are a professional HR assistant. Analyze the provided CV text and extract the candidate's profile into the specified JSON format.
    
    IMPORTANT RULES:
    1. For fields like email, phone, linkedin_url, if not CLEARLY present in the text, use an empty string "". DO NOT guess or fill with other text.
    2. Ensure the JSON is valid and matches the schema exactly.
    3. Return ONLY the JSON object.

    JSON STRUCTURE:
    {{
      "full_name": "Applicant Name (if found, else empty string)",
      "email": "Applicant Email (if found, else empty string)",
      "phone": "Applicant Phone (if found, else empty string)",
      "linkedin_url": "LinkedIn Profile URL (if found, else empty string)",
      "github_url": "GitHub Profile URL (if found, else empty string)",
      "portfolio_url": "Personal website/portfolio URL (if found, else empty string)",
      "primary_role": "A short, descriptive job title (e.g. Software Engineer, Data Analyst)",
      "skills": ["List", "of", "all", "technical", "skills", "languages", "tools"],
      "experience_level": "student" | "junior" | "mid",
      "work_experience": [
        {{
          "title": "Job Title",
          "company": "Company Name",
          "duration": "e.g. June 2022 - Aug 2022",
          "achievements": ["Bullet 1", "Bullet 2"]
        }}
      ],
      "projects": [
        {{
          "name": "Project Name",
          "description": "Short project description",
          "technologies": ["React", "Python"]
        }}
      ],
      "education": "Degree and University info (if any)",
      "summary": "A 2-3 sentence dense psychological/structural summary of their overall profile and professional geometry."
    }}
    
    CV TEXT:
    {text[:5000]}
    """
    
    try:
        api_key = os.getenv("GROQ_API_KEY", "")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a world-class HR AI. You must output raw JSON only, matching the exact format requested. Use empty strings for missing information."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20) as resp:
                if resp.status != 200:
                    text_resp = await resp.text()
                    raise Exception(f"Groq API error: {resp.status} - {text_resp}")
                
                result = await resp.json()
                content = result['choices'][0]['message']['content']
                
        data = None
        # Clean markdown code blocks if the LLM wrapped the JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        return CVSkills(**data)
        
    except Exception as e:
        print(f"Error extracting skills with Groq: {e}")
        return CVSkills(
            primary_role="Software intern", 
            skills=[], 
            experience_level="student",
            work_experience=[],
            projects=[],
            education="",
            summary=""
        )
