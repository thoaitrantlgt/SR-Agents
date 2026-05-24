"""RAG-Fusion — Multi-Query Generation + Ensemble Retrieval + RRF Fusion."""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from sragents.llm import chat, create_llm_client, strip_think_tags
from sragents.retrieve.base import register
from sragents.retrieve.ensemble import EnsembleRetriever


@register("rag_fusion")
class RAGFusionRetriever:
    """RAG-Fusion Retriever.
    
    1. Generates multiple variations of the query using an LLM.
    2. Performs Ensemble (BGE + BM25) retrieval for each variation.
    3. Fuses all results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        expansion_model: str = "Qwen3-4B",
        num_variations: int = 3,
        api_base: str | None = None,
        dense_model: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 256,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        workers: int = 4,
    ):
        self._llm_client = create_llm_client(api_base=api_base)
        self._expansion_model = expansion_model
        self._num_variations = num_variations
        self._ensemble = EnsembleRetriever(
            dense_model=dense_model,
            batch_size=batch_size,
            k1=k1,
            b=b,
            rrf_k=rrf_k,
        )
        self._rrf_k = rrf_k
        self._workers = workers

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        """Build the underlying ensemble index."""
        self._ensemble.build_index(corpus_ids, corpus_texts)

    def _generate_variations(self, query: str) -> list[str]:
        """Generate variations of the query using the LLM."""
        prompt = (
            f"You are an expert in query optimization for an autonomous AI agent. Given the user's problem, your task is to rewrite it into {self._num_variations} different search queries to find the most relevant external tools, actionable skills, or procedural knowledge needed to solve it.\n\n"
            "Do NOT make assumptions about the specific domain of the problem (e.g., medicine, coding, math, business). Instead, focus purely on the structural and actionable needs of the text.\n\n"
            "Generate EXACTLY 3 versions focusing on these distinct aspects:\n"
            "1. A high-level summary of the core problem and the ultimate goal, deliberately filtering out background narrative or scene-setting noise.\n"
            "2. Extract the exact technical terms, constraints, mathematical formulas, acronyms, specific identifiers, or variable names from the prompt. Preserve their exact original formatting without ever converting or modifying them.\n"
            "3. Rephrase the problem into a 'How-to' or 'What tool/method can use...' question that specifically seeks a step-by-step workflow or an executable tool.\n\n"
            "Only list the new queries, one per line, without numbering.\n\n"
            f"Original problem: {query}\n\n"
            "New versions:"
    )
        resp = chat(
            self._llm_client,
            self._expansion_model,
            prompt,
            temperature=0.7,
            max_tokens=512,
        )
        lines = strip_think_tags(resp).strip().split("\n")
        variations = [line.strip("- ").strip() for line in lines if line.strip()]
        return variations[: self._num_variations]

    def retrieve(
        self, queries: list[str], top_k: int = 50
    ) -> list[list[tuple[str, float]]]:
        """Perform RAG-Fusion retrieval."""
        final_results: list[list[tuple[str, float]]] = []

        print(f"  RAG-Fusion: Generating variations for {len(queries)} queries...")
        
        # Tạo các biến thể truy vấn song song
        all_variations = [[] for _ in queries]
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {pool.submit(self._generate_variations, q): i for i, q in enumerate(queries)}
            for fut in as_completed(futures):
                idx = futures[fut]
                all_variations[idx] = fut.result()

        # Làm phẳng danh sách truy vấn để thực hiện retrieval hàng loạt (Batch)
        flattened_queries = []
        mapping = [] # Lưu index của query gốc
        
        for i, q in enumerate(queries):
            # Luôn bao gồm query gốc
            flattened_queries.append(q)
            mapping.append(i)
            # Thêm các biến thể
            for v in all_variations[i]:
                flattened_queries.append(v)
                mapping.append(i)
        
        print(f"  RAG-Fusion: Retrieving for {len(flattened_queries)} total queries...")
        all_hits = self._ensemble.retrieve(flattened_queries, top_k=top_k)
        
        # Gom nhóm kết quả theo query gốc
        grouped_hits: list[list[list[tuple[str, float]]]] = [[] for _ in queries]
        for hits, orig_idx in zip(all_hits, mapping):
            grouped_hits[orig_idx].append(hits)
            
        print(f"  RAG-Fusion: Fusing results via RRF for {len(queries)} queries...")
        for query_hits_list in grouped_hits:
            # Thuật toán Multi-way RRF fusion
            scores: dict[str, float] = {}
            for hits_list in query_hits_list:
                for rank, (sid, _) in enumerate(hits_list, 1):
                    # Công thức chuẩn RRF
                    scores[sid] = scores.get(sid, 0.0) + 1.0 / (self._rrf_k + rank)
            
            # Sắp xếp lại dựa trên điểm RRF
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            final_results.append(ranked[:top_k])
            
        return final_results