"""LLM Classifier Rejector provider: decides whether to use a tool or internal knowledge.

Uses Decoupled Single-Token Logit Gating (DSTL-Gate) to measure confidence
by decoupling native capability evaluation and tool utility evaluation.
"""

import json
import math
import re
import sys
from pathlib import Path

from sragents.corpus import load_corpus_dict
from sragents.infer.base import register_provider
from sragents.llm import chat, create_llm_client, get_extra_body, strip_think_tags
from sragents.prompts import build_prompt

# ── DSTL-GATE DECOUPLED PROMPTS ──
_PROMPT_NATIVE = """Question: {query}
Proposed answer: I can solve this entirely using my internal knowledge without any external tools.
Response (yes/no):"""

_PROMPT_TOOL = """Question: {query}
Tool Info: {candidates}
Proposed answer: This tool is necessary and provides the correct formulas/procedures to solve the question.
Response (yes/no):"""

# Ngưỡng quyết định mặc định dựa trên xác suất P(Use Tool) từ Softmax
_DEFAULT_CONFIDENCE_THRESHOLD = 0.50


def _format_candidates(candidates: list[dict]) -> str:
    from sragents.corpus import display_name
    lines = []
    for i, s in enumerate(candidates, 1):
        desc = s.get("description", "")
        lines.append(f"[{i}] {display_name(s, i)}: {desc}")
    return "\n".join(lines)


def _extract_yes_logit(logprobs_data: list) -> float:
    """Trích xuất logit (hoặc logprob) của token 'yes'/'Yes' từ token đầu tiên.
    
    Vì OpenAI API trả về logprob (log của xác suất), chúng ta có thể sử dụng trực tiếp
    giá trị logprob này làm logit để tính toán tỉ lệ Softmax tương đối.
    Nếu không tìm thấy, trả về một giá trị rất thấp (mặc định -100.0).
    """
    if not logprobs_data:
        return -100.0

    first_token = logprobs_data[0]
    
    # Kiểm tra token được chọn chính thức
    token_str = first_token.token.strip().lower()
    if token_str.startswith("yes"):
        return float(first_token.logprob)

    # Kiểm tra các token thay thế trong top_logprobs
    if hasattr(first_token, "top_logprobs") and first_token.top_logprobs:
        for alt in first_token.top_logprobs:
            alt_str = alt.token.strip().lower()
            if alt_str.startswith("yes"):
                return float(alt.logprob)
                
    return -100.0


def _compute_dstl_confidence(logit_native: float, logit_tool: float) -> float:
    """Tính toán xác suất P(Use Tool) sử dụng hàm Softmax chuẩn hóa.
    
    Áp dụng công thức: P(Use Tool) = exp(l_tool) / (exp(l_native) + exp(l_tool))
    Trừ đi max_logit để đảm bảo an toàn số học (chống tràn số exp).
    """
    max_logit = max(logit_native, logit_tool)
    
    # Trường hợp cả hai đều không kích hoạt token 'yes' (đều là -100.0)
    if max_logit == -100.0:
        return 0.0

    exp_native = math.exp(logit_native - max_logit)
    exp_tool = math.exp(logit_tool - max_logit)
    
    return exp_tool / (exp_native + exp_tool)


@register_provider("llm_rejector")
class LLMRejectorProvider:
    """DSTL-Gate Rejector: Tách biệt đánh giá Câu hỏi và Công cụ bằng Single-Token Logit.
    
    Chạy 2 prompt song song để đo lường độ đồng thuận độc lập, loại bỏ hiện tượng
    Multiple-Choice Symbol Binding (MCSB).
    """

    def __init__(
        self,
        source: str,
        model: str,
        api_base: str | None = None,
        pool: int = 50,
        corpus_path: str | None = None,
        max_retries: int = 3,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        self._pool = int(pool)
        self._model = model
        self._client = create_llm_client(api_base=api_base)
        self._extra_body = get_extra_body(model, thinking=False)
        self._max_retries = max_retries
        self._confidence_threshold = float(confidence_threshold)
        self._corpus = (
            load_corpus_dict(corpus_path) if corpus_path else load_corpus_dict()
        )
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(
                f"Retrieval source file not found: {src}. "
                "Run `sragents retrieve` to produce it first."
            )
        data = json.loads(src.read_text())
        self._lookup = {r["instance_id"]: r["retrieved"] for r in data["results"]}

    def provide(self, instance: dict) -> list[dict]:
        retrieved = self._lookup.get(instance["instance_id"], [])[: self._pool]
        candidates = [
            self._corpus[r["skill_id"]]
            for r in retrieved
            if r["skill_id"] in self._corpus
        ]
        if not candidates:
            return []
        
        # Chỉ lấy Top-1 Skill theo thiết kế Single-Token của bài báo để tối ưu hóa đánh giá tiện ích
        top_skill = [candidates[0]]
        
        _, query = build_prompt(instance)
        
        # Định dạng 2 prompt độc lập
        prompt_native = _PROMPT_NATIVE.format(query=query)
        prompt_tool = _PROMPT_TOOL.format(
            query=query, 
            candidates=_format_candidates(top_skill)
        )

        # Gọi API cho Prompt 1: Đánh giá khả năng nội tại (Need-Aware)
        result_native = chat(
            self._client, self._model, prompt_native,
            temperature=0.0, max_tokens=8,  # Giảm temperature để tập trung logprob vào token đầu
            extra_body=self._extra_body,
            logprobs=True,
            top_logprobs=5,
        )
        logprobs_native = result_native[1] if isinstance(result_native, tuple) else None
        logit_native = _extract_yes_logit(logprobs_native)

        # Gọi API cho Prompt 2: Đánh giá độ hữu ích của Tool (Utility-Aware)
        result_tool = chat(
            self._client, self._model, prompt_tool,
            temperature=0.0, max_tokens=8,
            extra_body=self._extra_body,
            logprobs=True,
            top_logprobs=5,
        )
        logprobs_tool = result_tool[1] if isinstance(result_tool, tuple) else None
        logit_tool = _extract_yes_logit(logprobs_tool)

        # Tính toán xác suất đồng thuận dùng Tool qua Softmax
        p_use_tool = _compute_dstl_confidence(logit_native, logit_tool)
        
        inst_id = instance.get("instance_id", "?")
        print(
            f"  [dstl_gate] {inst_id}: "
            f"Logit(Native_Yes)={logit_native:.4f} Logit(Tool_Yes)={logit_tool:.4f} "
            f"P(Use Tool)={p_use_tool:.2%} "
            f"(threshold={self._confidence_threshold:.0%})",
            file=sys.stderr,
        )

        # Cổng quyết định logic gộp (Decision Gate)
        if p_use_tool >= self._confidence_threshold:
            print(f"  [dstl_gate] {inst_id}: Chấp nhận sử dụng công cụ.", file=sys.stderr)
            return top_skill
        else:
            print(f"  [dstl_gate] {inst_id}: Từ chối công cụ (Sử dụng kiến thức nội tại).", file=sys.stderr)
            return []