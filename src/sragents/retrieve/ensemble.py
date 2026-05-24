"""Ensemble retrieval using BGE (Dense) and BM25 (Sparse) with RRF fusion."""

from sragents.retrieve.bm25 import BM25Retriever
from sragents.retrieve.dense import DenseRetriever
from sragents.retrieve.base import register

@register("ensemble")
class EnsembleRetriever:
    """Ensemble Retriever combining BM25 and Dense retrieval via RRF."""

    def __init__(
        self,
        dense_model: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 256,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
    ):
        self._bm25 = BM25Retriever(k1=k1, b=b)
        
        # Determine prefix based on model name
        query_prefix = ""
        if "bge" in dense_model.lower():
            query_prefix = "Represent this sentence for searching relevant passages: "
        
        self._dense = DenseRetriever(
            model_name_or_path=dense_model,
            query_prefix=query_prefix,
            batch_size=batch_size,
        )
        self._rrf_k = rrf_k

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        """Build both sparse and dense indices."""
        print("  Ensemble: Building BM25 index...")
        self._bm25.build_index(corpus_ids, corpus_texts)
        print("  Ensemble: Building Dense index...")
        self._dense.build_index(corpus_ids, corpus_texts)

    def retrieve(
        self, queries: list[str], top_k: int = 50
    ) -> list[list[tuple[str, float]]]:
        """Retrieve from both and fuse using Reciprocal Rank Fusion (RRF)."""
        # We fetch more candidates for each to ensure better overlap for RRF
        fetch_k = max(top_k, 100)
        
        bm25_hits = self._bm25.retrieve(queries, top_k=fetch_k)
        dense_hits = self._dense.retrieve(queries, top_k=fetch_k)
        
        final_results = []
        for b_hits, d_hits in zip(bm25_hits, dense_hits):
            scores: dict[str, float] = {}
            
            # BM25 contribution
            for rank, (sid, _) in enumerate(b_hits, 1):
                scores[sid] = scores.get(sid, 0.0) + 1.0 / (self._rrf_k + rank)
                
            # Dense contribution
            for rank, (sid, _) in enumerate(d_hits, 1):
                scores[sid] = scores.get(sid, 0.0) + 1.0 / (self._rrf_k + rank)
            
            # Sort by RRF score
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            final_results.append(ranked[:top_k])
            
        return final_results
