import json
import os
import urllib.error
import urllib.request


DEFAULT_SYSTEM_PROMPT = (
    "你是一个谣言检测系统的解释生成模块。"
    "你只能根据给定的模型输出和证据解释判断依据，不能修改模型给出的最终结论。"
    "请用中文回答，语气客观、简洁，避免编造外部事实。"
)


class SchoolLLMExplainer:
    """OpenAI-compatible client for school-hosted LLM explanation APIs."""

    def __init__(
        self,
        api_key=None,
        api_url=None,
        base_url=None,
        model=None,
        timeout=20,
        temperature=0.2,
    ):
        self.api_key = api_key
        self.api_url = api_url or build_chat_completions_url(base_url)
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("SCHOOL_LLM_API_KEY"),
            api_url=os.getenv("SCHOOL_LLM_API_URL"),
            base_url=os.getenv("SCHOOL_LLM_BASE_URL"),
            model=os.getenv("SCHOOL_LLM_MODEL"),
            timeout=parse_float_env("SCHOOL_LLM_TIMEOUT", 20),
            temperature=parse_float_env("SCHOOL_LLM_TEMPERATURE", 0.2),
        )

    def is_configured(self):
        return bool(self.api_key and self.api_url and self.model)

    def status(self):
        return {
            "configured": self.is_configured(),
            "api_url": self.api_url,
            "model": self.model,
            "has_api_key": bool(self.api_key),
        }

    def explain(self, prediction):
        if not self.is_configured():
            raise RuntimeError(
                "LLM API is not configured. Set SCHOOL_LLM_API_KEY, "
                "SCHOOL_LLM_BASE_URL or SCHOOL_LLM_API_URL, and SCHOOL_LLM_MODEL."
            )

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": build_explanation_prompt(prediction)},
            ],
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("LLM API returned invalid JSON") from exc

        return extract_chat_content(payload)


def parse_float_env(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def build_chat_completions_url(base_url):
    if not base_url:
        return None
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def extract_chat_content(payload):
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM API response does not contain message content") from exc
    content = str(content).strip()
    if not content:
        raise RuntimeError("LLM API returned empty explanation")
    return content


def build_explanation_prompt(prediction):
    evidence = prediction.get("evidence", {})
    retrieval = evidence.get("retrieval_statistics", {})
    guard = evidence.get("probability_guard", {})
    baseline = evidence.get("baseline", {})

    keyword_terms = [
        item.get("term")
        for item in evidence.get("keyword_contributions", [])[:8]
        if item.get("term")
    ]
    retrieved_cases = [
        {
            "label": item.get("label"),
            "score": round_float(item.get("score")),
            "text": item.get("text"),
        }
        for item in evidence.get("retrieved_cases", [])[:3]
    ]

    compact_payload = {
        "original_tweet": prediction.get("input_text") or evidence.get("normalized_text"),
        "normalized_text": evidence.get("normalized_text"),
        "prediction_label": prediction.get("label_name"),
        "confidence": round_float(prediction.get("confidence")),
        "model_evidence_terms": keyword_terms,
        "rag_similar_samples": retrieved_cases,
        "final_decision": {
            "label_name": prediction.get("label_name"),
            "prob_rumor": round_float(prediction.get("prob_rumor")),
            "confidence": round_float(prediction.get("confidence")),
            "threshold": round_float(prediction.get("threshold")),
        },
        "baseline_decision": {
            "label_name": baseline.get("label_name"),
            "prob_rumor": round_float(baseline.get("prob_rumor")),
            "threshold": round_float(baseline.get("threshold")),
        },
        "retrieval_statistics": {
            "retrieved_count": retrieval.get("retrieved_count"),
            "retrieved_rumor_ratio": retrieval.get("retrieved_rumor_ratio"),
            "retrieved_nonrumor_ratio": retrieval.get("retrieved_nonrumor_ratio"),
            "max_similarity": retrieval.get("max_similarity"),
            "weighted_rumor_score": retrieval.get("weighted_rumor_score"),
            "retrieval_confidence": retrieval.get("retrieval_confidence"),
        },
        "decision_factors": evidence.get("decision_factors", []),
        "keyword_terms": keyword_terms,
        "claim_structure": evidence.get("claim_structure_features", {}),
        "probability_guard": {
            "applied": guard.get("applied", False),
            "type": guard.get("type"),
            "applied_rules": guard.get("applied_rules", []),
            "prob_before": round_float(guard.get("prob_before")),
            "prob_after": round_float(guard.get("prob_after")),
        },
        "retrieved_cases": retrieved_cases,
        "original_explanation": prediction.get("explanation"),
    }

    return (
        "请根据下面 JSON 中的模型输出，为用户生成一段中文解释。\n"
        "学校大语言模型接口只作为解释润色模块；分类标签由本地可复现模型输出，"
        "LLM 不参与标签决策。\n"
        "要求：\n"
        "1. 第一句明确说明最终判断是“谣言”还是“非谣言”。\n"
        "2. 不能修改模型给出的最终结论，只解释该结论的依据。\n"
        "3. 解释要覆盖原始推文、预测标签、置信度、模型证据词和 RAG 相似样本。\n"
        "4. 如果概率保护规则被触发，说明它如何影响概率。\n"
        "5. 不要声称你查询了互联网，也不要引入 JSON 之外的新事实。\n"
        "6. 控制在 120 到 180 个中文字符左右。\n\n"
        f"{json.dumps(compact_payload, ensure_ascii=False, indent=2)}"
    )


def round_float(value):
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return value
