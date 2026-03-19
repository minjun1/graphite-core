"""
graphite/pipeline/retriever.py — Default evidence retrieval functionality.
"""
from typing import List, Dict, Any

from graphite.claim import Claim

class DocumentCorpus:
    """A minimal in-memory corpus providing chunking and search functionality."""
    
    def __init__(self, documents: List[Dict[str, str]]):
        """
        Expects a list of dicts: [{'document_id': 'doc1', 'text': '...'}]
        """
        self.documents = documents
        self.chunks = self._chunk_documents()
        
    def _chunk_documents(self) -> List[Dict[str, Any]]:
        chunks = []
        for doc in self.documents:
            # Baseline paragraph-level chunking
            paragraphs = doc["text"].split("\n\n")
            for i, p in enumerate(paragraphs):
                if p.strip():
                    chunks.append({
                        "document_id": doc["document_id"],
                        "chunk_index": i,
                        "text": p.strip()
                    })
        return chunks


class EvidenceRetriever:
    """Retrieves candidate evidence spans for a list of claims."""
    
    def __init__(self, corpus: DocumentCorpus):
        self.corpus = corpus
        
    def retrieve_evidence(self, claims: List[Claim], top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        For each claim, return the top_k candidate evidence chunks.
        Provides a default out-of-the-box lexical fallback retrieval.
        """
        results = {}
        for claim in claims:
            # Simple lexical overlap scoring for out-of-the-box usage
            keywords = set(claim.subject_entities + claim.object_entities)
            # Remove node prefixes (e.g., 'company:AAPL' -> 'aapl')
            keywords = {k.split(":")[-1].lower() for k in keywords}
            # Fallback to claim_text tokens if no specific entities
            if not keywords:
                keywords = set(claim.claim_text.lower().split())
                if len(keywords) > 10:
                    keywords = set(list(keywords)[:10])
            
            scored_chunks = []
            for chunk in self.corpus.chunks:
                score = sum(1 for kw in keywords if kw in chunk["text"].lower())
                if score > 0:
                    scored_chunks.append((score, chunk))
                    
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [c[1] for c in scored_chunks[:top_k]]
            results[claim.claim_id] = top_chunks
            
        return results

def retrieve_evidence(claims: List[Claim], corpus_docs: List[Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to run end-to-end default retrieval."""
    corpus = DocumentCorpus(corpus_docs)
    retriever = EvidenceRetriever(corpus)
    return retriever.retrieve_evidence(claims)
