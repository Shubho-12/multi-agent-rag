# agents/retriever.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def normalize_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip()
    return s

class RetrieverAgent:
    def __init__(self):
        self.vectorizer = None
        self.doc_texts = []
        self.doc_vectors = None

    def add_documents(self, docs):
        self.doc_texts = [normalize_text(d) for d in docs]
        if not self.doc_texts:
            raise ValueError("No documents to index")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(self.doc_texts)

    def retrieve(self, query, top_k=3):
        if self.vectorizer is None or self.doc_vectors is None:
            return []
        q = normalize_text(query)
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(-sims)[:top_k]
        return [self.doc_texts[i] for i in top_indices]
