# rag_indexer.py
#import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import faiss
import numpy as np
import tiktoken
import os
import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, manually load .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

EMBED_MODEL = "text-embedding-3-large"
ENC = tiktoken.get_encoding("cl100k_base")


def extract_text_from_pdf(path):
    """Extract text page by page"""
    reader = PdfReader(path)
    text = []
    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            text.append(content)
    return text


def chunk_text(texts, max_tokens=800, overlap=80):
    """Split text into overlapping chunks for embeddings"""
    chunks = []
    for i, page in enumerate(texts):
        tokens = ENC.encode(page)
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = ENC.decode(chunk_tokens)
            chunks.append({"text": chunk_text, "meta": {"page": i + 1}})
            start += max_tokens - overlap
    return chunks

def embed_texts(chunks):
    """Call OpenAI embeddings API using direct HTTP requests"""
    texts = [c["text"] for c in chunks]
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": EMBED_MODEL,
        "input": texts
    }
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    result = response.json()
    embs = np.array([item["embedding"] for item in result["data"]]).astype("float32")
    return embs


def build_faiss_index(embs):
    """Create FAISS index and add vectors"""
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embs)
    index.add(embs)
    return index


def retrieve(index, embs, chunks, query, k=5):
    """Retrieve top-k relevant chunks for a query"""
    # Create embedding for query using direct HTTP request
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": EMBED_MODEL,
        "input": [query]
    }
    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=data
    )
    response.raise_for_status()
    result = response.json()
    q_emb = np.array([result["data"][0]["embedding"]]).astype("float32")
    
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)
    results = []
    for i, s in zip(ids[0], scores[0]):
        results.append({"text": chunks[i]["text"], "meta": chunks[i]["meta"], "score": float(s)})
    return results
