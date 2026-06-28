import json
import google.generativeai as genai
from config import settings
import asyncio

async def generate_outreach_email(cv_skills: list, job_title: str, company: str, job_description: str) -> str:
    """
    Uses Gemini Flash to generate a hyper-personalized outreach email.
    """
    if not settings.GEMINI_API_KEY:
        return "Please configure GEMINI_API_KEY to generate personalized emails."

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')

    prompt = f"""
    Write a hyper-personalized, professional, and concise cold email for an internship application.
    
    CANDIDATE SKILLS/PROJECTS: {', '.join(cv_skills)}
    JOB TITLE: {job_title}
    COMPANY: {company}
    JOB DESCRIPTION SNIPPET: {job_description[:1000]}
    
    INSTRUCTIONS:
    1. Mention 1-2 specific overlapping technical skills or projects.
    2. Keep it under 150 words.
    3. Use a professional but enthusiastic tone.
    4. Mention that you found the opportunity via Resume Matcher.
    5. Include placeholders like [Your Name] and [Date/Time for meeting].
    
    Return ONLY the email text.
    """

    try:
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error generating email: {e}")
        return f"Hi {company} Team,\n\nI am interested in the {job_title} position. I have experience in {', '.join(cv_skills[:3])}. Looking forward to hearing from you."
