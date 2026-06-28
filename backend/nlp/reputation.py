import os
import json
import asyncio
import aiohttp

async def batch_reputation_check(companies: list[str]) -> dict:
    """
    Checks the reputation of a list of companies in a single LLM call.
    Returns a dict mapping company_name -> {"status": "green|yellow|red", "summary": "..."}
    """
    unique_companies = list(set([c for c in companies if c and c.lower() != "n/a" and c.strip()]))
    if not unique_companies:
        return {}
        
    prompt = f"""
    You are an AI career agent acting as a protector for Pakistani students seeking internships.
    Evaluate the reputation of these employers in Pakistan:
    {json.dumps(unique_companies)}
    
    Search your database for any Reddit, Quora, Glassdoor, or web sentiment about these specific companies.
    Pay close attention to severe red flags:
    - Asking interns to pay upfront fees or buy "training equipment" (SCAM).
    - Extremely poor conditions or unpaid exploitation with no learning value.
    - Highly toxic work culture or withholding stipends.
    
    If it's a known, safe company or startup, mark it "green". 
    If there are slight concerns or mixed reviews, mark it "yellow". 
    If it's a known scam or explicitly asks for fees, mark it "red".
    If you have absolutely zero data on the company, mark it "green" with the summary "No negative flags found."
    
    Return EXACTLY a raw JSON dictionary mapping the company name to its analysis.
    Schema:
    {{
        "Company Name": {{
            "status": "green"| "yellow"| "red",
            "summary": "1 brief sentence (max 15 words) explaining the reputation."
        }}
    }}
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
                {"role": "system", "content": "You must output raw JSON only, matching the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20) as resp:
                if resp.status != 200:
                    text_resp = await resp.text()
                    raise Exception(f"Groq API error: {resp.status} - {text_resp}")
                
                result = await resp.json()
                content = result['choices'][0]['message']['content']
                
        return json.loads(content)
        
    except Exception as e:
        print(f"[Reputation Engine] Check failed with Groq: {e}")
        return {}
