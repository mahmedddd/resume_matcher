# entity_extractor.py

import re
import spacy
from sentence_transformers import SentenceTransformer, util

# Load SpaCy and Transformer Model once
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Comprehensive Skill List ---
COMMON_SKILLS = [
    # --- Programming & Software ---
    "python", "java", "javascript", "c++", "c#", "sql", "html", "css", "php",
    "typescript", "go", "ruby", "r", "scala", "bash", "matlab", "shell scripting",
    "git", "github", "api development", "rest api", "graphql", "vscode", "docker",
    "kubernetes", "jenkins", "ci/cd", "linux", "windows server", "flask", "django",
    "node.js", "react", "angular", "vue.js", "spring boot", "express.js", "firebase",

    # --- Data Science & Analytics ---
    "machine learning", "deep learning", "data science", "data analysis",
    "data engineering", "data mining", "big data", "pandas", "numpy", "scipy",
    "scikit-learn", "tensorflow", "keras", "pytorch", "matplotlib", "seaborn",
    "tableau", "power bi", "excel", "sql", "nosql", "hadoop", "spark",
    "predictive modeling", "data visualization", "statistical analysis",
    "data wrangling", "etl pipelines",

    # --- Networking & Security ---
    "network administration", "network security", "cybersecurity", "penetration testing",
    "ethical hacking", "firewall configuration", "dns", "tcp/ip", "vpn", "wireshark",
    "load balancing", "active directory", "linux server", "cloud security",
    "zero trust architecture", "zero-day mitigation",

    # --- Cloud, DevOps & IT ---
    "aws", "azure", "google cloud platform", "cloud computing", "virtualization",
    "docker", "kubernetes", "terraform", "ansible", "devops", "system administration",
    "helpdesk support", "it support", "vpn configuration", "sla management",

    # --- QA & Testing ---
    "manual testing", "automation testing", "selenium", "postman", "test case design",
    "regression testing", "unit testing", "integration testing", "bug tracking",
    "jira", "quality assurance", "load testing", "uat", "smoke testing",

    # --- Design & UX/UI ---
    "ui design", "ux design", "figma", "adobe xd", "sketch", "invision",
    "wireframing", "prototyping", "usability testing", "responsive design",
    "graphic design", "illustrator", "photoshop", "adobe creative suite",
    "design systems", "branding", "logo design", "visual hierarchy",

    # --- Marketing & Sales ---
    "seo", "sem", "content marketing", "email marketing", "google analytics",
    "facebook ads", "instagram marketing", "hubspot", "crm", "campaign management",
    "brand management", "market research", "adwords", "digital strategy",
    "salesforce", "copywriting", "social media marketing", "affiliate marketing",

    # --- Finance & Business ---
    "financial analysis", "budgeting", "forecasting", "investment strategies",
    "excel modeling", "accounting", "bookkeeping", "erp", "quickbooks", "oracle",
    "financial reporting", "financial planning", "p&l management", "revenue growth",
    "business intelligence", "business development", "cost analysis",

    # --- Operations & Supply Chain ---
    "supply chain management", "inventory management", "procurement", "logistics",
    "vendor management", "warehouse management", "demand forecasting",
    "purchasing", "shipping coordination", "order fulfillment", "rfq management",

    # --- Legal & Compliance ---
    "legal research", "contract drafting", "litigation", "paralegal work",
    "legal writing", "case management", "compliance", "document review",
    "intellectual property", "corporate law", "client intake",

    # --- Education & Healthcare ---
    "curriculum development", "lesson planning", "online teaching", "tutoring",
    "instructional design", "student assessment", "special education", "child psychology",
    "nursing", "patient care", "treatment planning", "occupational therapy",
    "speech therapy", "clinical documentation", "diagnosis", "medication management",

    # --- Soft Skills & Management ---
    "project management", "team management", "stakeholder communication",
    "time management", "critical thinking", "problem solving", "leadership",
    "adaptability", "emotional intelligence", "collaboration", "conflict resolution",
    "attention to detail", "presentation skills", "strategic planning", "scrum",
    "agile", "kanban", "okrs", "performance evaluation",

    # --- Writing & Content ---
    "technical writing", "content writing", "research writing", "blog writing",
    "copywriting", "proofreading", "editing", "creative writing", "documentation",

    # --- Miscellaneous ---
    "crm systems", "data entry", "sap", "zendesk", "service desk", "inventory control",
    "administrative support", "event planning", "hr policies", "recruitment",
    "customer service", "client onboarding", "training and development",
    "scheduling", "vendor negotiation", "travel coordination"
]


# Pre-encode skills for semantic matching
COMMON_SKILLS_LOWER = [skill.lower() for skill in COMMON_SKILLS]
SKILL_EMBEDDINGS = embedder.encode(COMMON_SKILLS_LOWER, convert_to_tensor=True)


# Rule-Based Match
def extract_skills_rule_based(text):
    tokens = [token.text.lower() for token in nlp(text)]
    return list(set(skill for skill in COMMON_SKILLS_LOWER if skill in tokens))


# Semantic Match
def extract_skills_semantic(text, threshold=0.6):
    text_embedding = embedder.encode([text], convert_to_tensor=True)
    cosine_scores = util.cos_sim(text_embedding, SKILL_EMBEDDINGS)[0]

    matched_skills = [
        COMMON_SKILLS[i]
        for i in range(len(COMMON_SKILLS))
        if cosine_scores[i] > threshold
    ]
    return list(set(matched_skills))


# Combined Approach
def extract_skills(text):
    rule_skills = extract_skills_rule_based(text)
    semantic_skills = extract_skills_semantic(text)
    return list(set(rule_skills + semantic_skills))


# Extract experience like "3+ years of experience"
def extract_experience(text):
    matches = re.findall(r"(\d+)\+?\s*(?:years|yrs)\s*(?:of)?\s*experience", text, re.IGNORECASE)
    return max(map(int, matches)) if matches else 0
