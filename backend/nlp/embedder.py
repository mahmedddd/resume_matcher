from sentence_transformers import SentenceTransformer
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedder():
    """
    Initializes and returns the sentence transformer model.
    Uses lru_cache to ensure it's only loaded once.
    """
    # all-MiniLM-L6-v2 is fast and small (~22MB)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
