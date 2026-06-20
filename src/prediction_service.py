from pathlib import Path
from threading import Lock

from src.evidence_pipeline import RumorDetectionPipeline
from src.llm_explainer import SchoolLLMExplainer


class RumorPredictionService:
    """Shared prediction service for CLI and web entry points."""

    def __init__(
        self,
        model_path="models/main_fusion.pkl",
        train_path="train.csv",
        llm_explainer=None,
        use_llm_by_default=True,
    ):
        self.model_path = Path(model_path)
        self.train_path = Path(train_path)
        self.llm_explainer = llm_explainer or SchoolLLMExplainer.from_env()
        self.use_llm_by_default = use_llm_by_default
        self._pipeline = None
        self._lock = Lock()

    def status(self):
        return {
            "model_path": str(self.model_path),
            "train_path": str(self.train_path),
            "model_exists": self.model_path.exists(),
            "train_exists": self.train_path.exists(),
            "loaded": self._pipeline is not None,
            "llm_evidence": self.llm_explainer.status(),
            "llm_enabled_by_default": self.use_llm_by_default,
        }

    def load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._lock:
            if self._pipeline is None:
                self._pipeline = RumorDetectionPipeline.load(
                    self.model_path,
                    train_path=self.train_path,
                )
        return self._pipeline

    def predict(self, text, top_k=None, use_llm=None):
        pipeline = self.load_pipeline()
        result = pipeline.predict(text, top_k=top_k, include_explanation=True)
        result["input_text"] = str(text)

        if use_llm is None:
            use_llm = self.use_llm_by_default
        if use_llm:
            result["llm_evidence"] = self.build_llm_evidence(result)
        else:
            result["llm_evidence"] = {
                "available": False,
                "enabled": False,
                "source": "school_llm",
                "reason": "LLM evidence was disabled for this request.",
            }
        return result

    def build_llm_evidence(self, result):
        try:
            explanation = self.llm_explainer.explain(result)
        except Exception as exc:
            return {
                "available": False,
                "enabled": True,
                "source": "school_llm",
                "error": str(exc),
                "fallback_explanation": result.get("explanation", ""),
            }
        return {
            "available": True,
            "enabled": True,
            "source": "school_llm",
            "model": self.llm_explainer.model,
            "explanation": explanation,
        }


def parse_top_k(value):
    if value in (None, ""):
        return None
    try:
        top_k = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be an integer") from exc
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    return top_k


def parse_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def validate_prediction_text(text):
    text = str(text or "").strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > 10_000:
        raise ValueError("text is too long; keep it under 10000 characters")
    return text
