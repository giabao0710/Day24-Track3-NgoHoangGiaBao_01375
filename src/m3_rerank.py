from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._fallback_encoder = None

    def _load_model(self):
        if self._model is None and self._fallback_encoder is None:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            try:
                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception:
                try:
                    self._fallback_encoder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                except Exception:
                    try:
                        self._fallback_encoder = SentenceTransformer("all-MiniLM-L6-v2")
                    except Exception:
                        self._fallback_encoder = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        if model is not None:
            pairs = [(query, doc.get("text", "")) for doc in documents]
            try:
                scores = model.predict(pairs)
                if isinstance(scores, (int, float)):
                    scores = [scores]
            except Exception:
                scores = [doc.get("score", 0.0) for doc in documents]
        elif self._fallback_encoder is not None:
            import numpy as np
            q_emb = self._fallback_encoder.encode(query, show_progress_bar=False)
            d_embs = self._fallback_encoder.encode([d.get("text", "") for d in documents], show_progress_bar=False)
            scores = []
            for d_emb in d_embs:
                cos_sim = float(np.dot(q_emb, d_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-9))
                scores.append(cos_sim)
        else:
            scores = [doc.get("score", 0.0) for doc in documents]

        scored = sorted(
            zip(scores, documents),
            key=lambda x: float(x[0]),
            reverse=True
        )

        results = []
        for i, (score, doc) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from flashrank import Ranker
                self._model = Ranker()
            except Exception:
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        model = self._load_model()
        if model is None:
            return [
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i + 1
                )
                for i, doc in enumerate(documents[:top_k])
            ]
        try:
            from flashrank import RerankRequest
            passages = [{"id": i, "text": d.get("text", ""), "meta": d.get("metadata", {})} for i, d in enumerate(documents)]
            rerank_req = RerankRequest(query=query, passages=passages)
            results = model.rerank(rerank_req)
            return [
                RerankResult(
                    text=r["text"],
                    original_score=float(documents[r["id"]].get("score", 0.0)),
                    rerank_score=float(r["score"]),
                    metadata=r.get("meta", {}),
                    rank=i + 1
                )
                for i, r in enumerate(results[:top_k])
            ]
        except Exception:
            return [
                RerankResult(
                    text=doc.get("text", ""),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i + 1
                )
                for i, doc in enumerate(documents[:top_k])
            ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
