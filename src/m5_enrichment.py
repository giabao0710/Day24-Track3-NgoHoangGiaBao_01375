from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, re, json
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️  OpenAI summarize failed: {e}")

    # Extractive fallback (không cần API)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    if sentences:
        return ". ".join(sentences[:2]) + ("." if not sentences[0].endswith(".") else "")
    return text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            questions = resp.choices[0].message.content.strip().split("\n")
            cleaned = [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
            return [q if q.endswith("?") else f"{q}?" for q in cleaned][:n_questions]
        except Exception as e:
            print(f"  ⚠️  OpenAI HyQA failed: {e}")

    # Extractive rule-based fallback
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    questions = []
    for s in sentences:
        s_clean = s.rstrip(".")
        if any(w in s_clean.lower() for w in ["nghỉ", "phép", "lương", "ngày", "quy định", "mật khẩu"]):
            questions.append(f"Quy định chi tiết về {s_clean.lower()} là gì?")
        else:
            questions.append(f"{s_clean} như thế nào?")
        if len(questions) >= n_questions:
            break

    if not questions:
        questions = [f"Nội dung chính của tài liệu là gì?"]
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            context = resp.choices[0].message.content.strip()
            return f"{context}\n\n{text}"
        except Exception as e:
            print(f"  ⚠️  OpenAI contextual failed: {e}")

    # Deterministic fallback: preserves text and prepends clear context
    clean_title = document_title.replace(".md", "").replace(".pdf", "").replace("_", " ").title() if document_title else "Quy định nội bộ"
    prefix = f"Tài liệu {clean_title}: Trích đoạn hướng dẫn và chính sách chi tiết."
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  OpenAI metadata failed: {e}")

    # Fallback keyword extraction
    text_lower = text.lower()
    category = "policy"
    if any(k in text_lower for k in ["lương", "thưởng", "nghỉ phép", "thử việc", "hợp đồng", "nhân sự", "hr"]):
        category = "hr"
    elif any(k in text_lower for k in ["mật khẩu", "vpn", "wifi", "email", "server", "it", "bảo mật"]):
        category = "it"
    elif any(k in text_lower for k in ["tài chính", "ngân sách", "báo cáo", "bctc", "thuế", "chi phí"]):
        category = "finance"
    elif any(k in text_lower for k in ["an toàn", "pccc", "sơ cứu", "bảo hộ"]):
        category = "safety"

    entities = []
    for match in re.finditer(r'\b[A-ZĐÂÊÔƯÁÀẢÃẠ][a-zđâêôưáàảãạ]+(?:\s+[A-ZĐÂÊÔƯÁÀẢÃẠ][a-zđâêôưáàảãạ]+)*\b', text):
        ent = match.group()
        if len(ent) > 3 and ent not in entities:
            entities.append(ent)

    return {
        "topic": category,
        "entities": entities[:5],
        "category": category,
        "language": "vi"
    }


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("sk-..."):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=400,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"  ⚠️  Enrichment API failed: {e}")

    # Combined local deterministic fallback (0 API latency)
    clean_title = source.replace(".md", "").replace(".pdf", "").replace("_", " ").title() if source else "Quy định nội bộ"
    summary = summarize_chunk(text)
    questions = generate_hypothesis_questions(text, n_questions=3)
    context_line = f"Tài liệu {clean_title}: Trích đoạn hướng dẫn và chính sách."
    metadata = extract_metadata(text)

    return {
        "summary": summary,
        "questions": questions,
        "context": context_line,
        "metadata": metadata
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
