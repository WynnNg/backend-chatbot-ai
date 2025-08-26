from sentence_transformers import CrossEncoder

class Rerank():
    def __init__(self,):
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

    def rerank_docs(self, query, docs):
        if not docs:
            return []
        # Prepare the pairs for reranking
        pairs = [(query, doc) for doc in docs]
        
        # Get the scores from the reranker
        scores = self.reranker.predict(pairs)
        
        # Combine docs with their scores and sort them
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the sorted documents
        reranked_docs = [doc for doc, score in scored_docs]
        
        return reranked_docs