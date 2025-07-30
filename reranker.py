# reranker.py
from sentence_transformers import CrossEncoder

class ReRanker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"✨ Initializing ReRanker with model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=4):
        # Create pairs of [query, document_content]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get scores from the CrossEncoder model
        scores = self.model.predict(pairs)
        
        # Combine documents with their scores and sort
        doc_scores = list(zip(documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top_k best documents
        return [doc for doc, score in doc_scores[:top_k]]