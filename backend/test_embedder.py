import time
from sentence_transformers import SentenceTransformer
import os

print("Starting embedder diagnostic...")
start = time.time()
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded in {time.time() - start:.2f}s")
    
    test_text = "This is a test sentence for embedding speed."
    start = time.time()
    emb = model.encode([test_text] * 10)
    print(f"Encoded 10 sentences in {time.time() - start:.4f}s")
    print("SUCCESS")
except Exception as e:
    print(f"FAILURE: {e}")
