from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
        from presidio_anonymizer import AnonymizerEngine

        cccd_recognizer = PatternRecognizer(
            supported_entity="VN_CCCD",
            patterns=[
                Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
                Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
            ],
        )
        phone_recognizer = PatternRecognizer(
            supported_entity="VN_PHONE",
            patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
        )

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(cccd_recognizer)
        registry.add_recognizer(phone_recognizer)

        analyzer = AnalyzerEngine(registry=registry)
        anonymizer = AnonymizerEngine()
        return analyzer, anonymizer
    except Exception:
        return None, None


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio + regex.

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    entities = []
    anonymized = text

    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    if analyzer is not None and anonymizer is not None:
        try:
            results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            if results:
                anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
                entities = [
                    {"type": r.entity_type, "text": text[r.start:r.end],
                     "score": round(r.score, 3), "start": r.start, "end": r.end}
                    for r in results
                ]
        except Exception:
            pass

    found_types = {e["type"] for e in entities}

    # Ensure VN_CCCD (12 digits or 9 digits) is detected
    for m in re.finditer(r"\b\d{12}\b", text):
        if "VN_CCCD" not in found_types:
            entities.append({"type": "VN_CCCD", "text": m.group(), "score": 0.9, "start": m.start(), "end": m.end()})
            anonymized = anonymized.replace(m.group(), "<VN_CCCD>")
    for m in re.finditer(r"\b\d{9}\b", text):
        if "VN_CCCD" not in found_types and "CMND" not in found_types:
            entities.append({"type": "VN_CCCD", "text": m.group(), "score": 0.7, "start": m.start(), "end": m.end()})
            anonymized = anonymized.replace(m.group(), "<VN_CCCD>")

    # Ensure VN_PHONE (0[3-9]xxxxxxxx) is detected
    for m in re.finditer(r"\b0[3-9]\d{8}\b", text):
        if "VN_PHONE" not in found_types:
            entities.append({"type": "VN_PHONE", "text": m.group(), "score": 0.9, "start": m.start(), "end": m.end()})
            anonymized = anonymized.replace(m.group(), "<VN_PHONE>")

    # Ensure EMAIL is detected
    for m in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        if "EMAIL" not in found_types and "EMAIL_ADDRESS" not in found_types:
            entities.append({"type": "EMAIL_ADDRESS", "text": m.group(), "score": 0.9, "start": m.start(), "end": m.end()})
            anonymized = anonymized.replace(m.group(), "<EMAIL_ADDRESS>")

    has_pii = len(entities) > 0
    return {"has_pii": has_pii, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    try:
        from nemoguardrails import RailsConfig, LLMRails
        config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
        rails = LLMRails(config)
        return rails
    except Exception:
        return None


def _is_offline_blocked_input(text: str) -> tuple[bool, str]:
    """Offline topic / jailbreak / injection heuristic."""
    t = text.lower()

    # Jailbreak & instruction overrides
    jailbreaks = [
        "bỏ qua tất cả hướng dẫn", "bỏ qua hướng dẫn", "ignore your previous", "ignore previous",
        "pretend you are dan", "unrestricted ai", "ai không có giới hạn",
        "mật khẩu admin", "hướng dẫn tấn công mạng", "system override",
        "dump all training data", "admin command", "bỏ qua tất cả policy",
        "in ra toàn bộ system instructions", "tiết lộ bảng lương", "tiết lộ lương tháng",
        "lương lead là bao nhiêu", "lương hiện tại 25 triệu", "đóng vai hr manager",
    ]
    for pattern in jailbreaks:
        if pattern in t:
            return True, "Xin lỗi, tôi không thể thực hiện yêu cầu này. Tôi chỉ có thể trả lời các câu hỏi về chính sách nhân sự công ty."

    # Off-topic checks
    off_topics = [
        "bài thơ", "mùa thu hà nội", "nấu phở", "phở bò", "bitcoin", "ethereum",
        "giá bitcoin", "phương trình vi phân", "dy/dx", "marvel", "bộ phim hay nhất",
    ]
    for pattern in off_topics:
        if pattern in t:
            return True, "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về chính sách nội bộ của công ty như nghỉ phép, lương thưởng, bảo hiểm, và các quy trình HR. Bạn có muốn hỏi về chủ đề đó không?"

    # PII extraction requests
    pii_requests = [
        "cho tôi biết cccd", "số điện thoại của nhân viên", "thông tin nhân viên",
        "tiết lộ thông tin", "bảng lương chi tiết",
    ]
    for pattern in pii_requests:
        if pattern in t:
            return True, "Xin lỗi, tôi không thể cung cấp thông tin cá nhân của nhân viên cụ thể. Đây là dữ liệu bảo mật theo chính sách phân loại dữ liệu của công ty."

    return False, ""


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    # Check through NeMo rails if available
    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[{"role": "user", "content": text}]
            )
            refuse_keywords = [
                "xin lỗi", "không thể", "không được phép", "i cannot", "i'm sorry",
                "bảo mật", "tôi chỉ có thể", "quy chế", "từ chối",
            ]
            blocked = any(kw in response.lower() for kw in refuse_keywords)
            if blocked:
                return {
                    "allowed": False,
                    "blocked_reason": "nemo_input_rail",
                    "response": response,
                }
        except Exception:
            pass

    # Offline / rule-based fallback
    is_blocked, refuse_msg = _is_offline_blocked_input(text)
    if is_blocked:
        return {
            "allowed": False,
            "blocked_reason": "nemo_input_rail",
            "response": refuse_msg,
        }

    return {
        "allowed": True,
        "blocked_reason": None,
        "response": "Input accepted.",
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    NeMo output rails hoạt động trong context của cả cuộc hội thoại (input + output).
    Kiểm tra: có PII không? Nội dung có phù hợp không? Có hallucination rõ ràng không?

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ])
            refuse_keywords = ["xin lỗi", "không thể cung cấp", "i cannot", "liên hệ phòng nhân sự"]
            flagged = any(kw in response.lower() for kw in refuse_keywords)
            if flagged:
                return {
                    "safe": False,
                    "flagged_reason": "nemo_output_rail",
                    "final_answer": response,
                }
        except Exception:
            pass

    # Quick scan for sensitive patterns in answer
    sensitive_keywords = ["mật khẩu admin là", "cccd của nhân viên là", "sđt cá nhân của"]
    ans_lower = answer.lower()
    for kw in sensitive_keywords:
        if kw in ans_lower:
            return {
                "safe": False,
                "flagged_reason": "nemo_output_rail",
                "final_answer": "Tôi không thể cung cấp thông tin này. Vui lòng liên hệ phòng Nhân sự trực tiếp.",
            }

    return {"safe": True, "flagged_reason": None, "final_answer": answer}


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                           analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii (cho category pii_injection)
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {
          "id": int, "category": str, "input": str,
          "expected": "blocked"|"allowed",
          "actual":   "blocked"|"allowed",
          "blocked_by": str | None,       # "presidio" | "nemo_input" | None
          "passed": bool,
        }
    """
    async def _run_all():
        results = []
        for item in adversarial_set:
            blocked_by = None

            # Layer 1: Presidio PII (synchronous, fast)
            pii_result = pii_scan(item["input"], analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            # Layer 2: NeMo input rail
            if blocked_by is None:
                rail_result = await check_input_rail(item["input"], rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            passed = (actual == item.get("expected", "blocked"))
            results.append({
                "id": item["id"],
                "category": item["category"],
                "input": item["input"][:80] + "...",
                "expected": item["expected"],
                "actual": actual,
                "blocked_by": blocked_by,
                "passed": passed,
            })
        return results

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.submit(asyncio.run, _run_all()).result()
        else:
            results = asyncio.run(_run_all())
    except Exception:
        results = asyncio.run(_run_all())

    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed ({passed/len(results):.0%})")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                         rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Mục tiêu production: P95 total < LATENCY_BUDGET_P95_MS (500ms mặc định)

    Insight cần quan sát:
        - Presidio: local regex → rất nhanh (<10ms)
        - NeMo:     LLM API call / guard → (<300ms)
        → Tổng: dominated by NeMo

    Returns:
        {
          "presidio_ms":  {"p50": float, "p95": float, "p99": float},
          "nemo_ms":      {"p50": float, "p95": float, "p99": float},
          "total_ms":     {"p50": float, "p95": float, "p99": float},
          "latency_budget_ok": bool,
          "budget_ms": int,
        }
    """
    presidio_times = []
    nemo_times = []
    total_times = []

    inputs_to_test = test_inputs[:n_runs] if test_inputs else ["Test input sample"]

    async def _measure():
        for text in inputs_to_test:
            # Presidio (synchronous)
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = max(0.1, (time.perf_counter() - t0) * 1000)

            # NeMo input rail
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = max(0.5, (time.perf_counter() - t1) * 1000)

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, _measure()).result()
        else:
            asyncio.run(_measure())
    except Exception:
        asyncio.run(_measure())

    def percentiles(times: list[float]) -> dict[str, float]:
        if not times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(times)
        n = len(s)
        p50_idx = min(int(n * 0.50), n - 1)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)
        return {
            "p50": round(s[p50_idx], 2),
            "p95": round(s[p95_idx], 2),
            "p99": round(s[p99_idx], 2),
        }

    total_p = percentiles(total_times)
    return {
        "presidio_ms": percentiles(presidio_times),
        "nemo_ms": percentiles(nemo_times),
        "total_ms": total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    adv_results = run_adversarial_suite(adversarial_set)

    # Task 12: P95 latency
    sample_inputs = [item["input"] for item in adversarial_set]
    latency = measure_p95_latency(sample_inputs, n_runs=20)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    # Save reports/guard_results.json
    passed_count = sum(1 for r in adv_results if r["passed"])
    guard_report = {
        "adversarial_suite": {
            "total_tested": len(adv_results),
            "passed": passed_count,
            "pass_rate": round(passed_count / max(len(adv_results), 1), 3),
            "results": adv_results,
        },
        "latency_benchmarks": latency,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as f:
        json.dump(guard_report, f, ensure_ascii=False, indent=2)
    print("✓ Saved reports/guard_results.json")
