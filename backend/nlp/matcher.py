import numpy as np
from typing import List
from database import Internship
from models import InternshipOut

def cosine_similarity(v1, v2):
    """
    Computes cosine similarity between two vectors.
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def rank_internships(cv_embedding: List[float], cv_skills: List[str], internships: List[Internship], top_n: int = 10) -> List[InternshipOut]:
    """
    Ranks internships based on CV embedding and skills.
    """
    results = []
    
    cv_skills_set = set(s.lower() for s in cv_skills)
    
    for intern in internships:
        intern_embedding = intern.get_embedding()
        if not intern_embedding:
            continue
            
        score = cosine_similarity(cv_embedding, intern_embedding)
        
        # Skill matching
        intern_skills = intern.get_skills()
        intern_skills_set = set(s.lower() for s in intern_skills)
        
        matching_skills = list(cv_skills_set.intersection(intern_skills_set))
        missing_skills = list(intern_skills_set.difference(cv_skills_set))
        
        # Add to results
        results.append(InternshipOut(
            id=intern.id,
            title=intern.title,
            company=intern.company,
            city=intern.city,
            is_remote=intern.is_remote,
            description=intern.description,
            skills_required=intern_skills,
            url=intern.url,
            source=intern.source,
            deadline=intern.deadline,
            salary=intern.salary,
            scraped_at=intern.scraped_at,
            match_score=float(score * 100), # 0-100 scale
            matching_skills=matching_skills,
            missing_skills=missing_skills
        ))
        
    # Sort by match score descending
    results.sort(key=lambda x: x.match_score, reverse=True)
    return results[:top_n]
