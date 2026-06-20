import pickle
from pathlib import Path

from src.text_model import TfidfLinearRumorModel, dedupe_terms, metrics


DEFAULT_ENSEMBLE_CONFIGS = [
    {
        "name": "char_no_meta",
        "epochs": 50,
        "kwargs": {"max_features": 45000, "seed": 13, "use_meta": False},
    },
    {
        "name": "unigram_bigram",
        "epochs": 55,
        "kwargs": {
            "max_features": 30000,
            "seed": 29,
            "use_char": False,
            "use_meta": False,
        },
    },
]

ENSEMBLE_SEARCH_CONFIGS = [
    DEFAULT_ENSEMBLE_CONFIGS[0],
    DEFAULT_ENSEMBLE_CONFIGS[1],
    {
        "name": "char_only",
        "epochs": 50,
        "kwargs": {
            "max_features": 45000,
            "seed": 41,
            "use_unigram": False,
            "use_bigram": False,
            "use_char": True,
            "use_meta": False,
        },
    },
    {
        "name": "word_with_meta",
        "epochs": 55,
        "kwargs": {
            "max_features": 30000,
            "seed": 53,
            "use_char": False,
            "use_meta": True,
        },
    },
    {
        "name": "char_word_with_meta",
        "epochs": 50,
        "kwargs": {"max_features": 45000, "seed": 67, "use_meta": True},
    },
]


class EnsembleRumorModel:
    def __init__(self, models=None, model_names=None, threshold=0.5, weights=None):
        self.models = models or []
        self.model_names = model_names or []
        self.threshold = threshold
        self.weights = weights

    def fit(self, texts, labels, configs=None):
        configs = configs or DEFAULT_ENSEMBLE_CONFIGS
        self.models = []
        self.model_names = []
        for config in configs:
            model = TfidfLinearRumorModel(**config["kwargs"])
            model.fit(texts, labels, epochs=config["epochs"])
            self.models.append(model)
            self.model_names.append(config["name"])
        self.weights = [1.0 / len(self.models)] * len(self.models)
        return self

    def predict_proba_one(self, text):
        if not self.models:
            raise ValueError("ensemble has no sub-models")
        probs = [model.predict_proba_one(text) for model in self.models]
        weights = getattr(self, "weights", None) or [1.0 / len(probs)] * len(probs)
        return sum(weight * prob for weight, prob in zip(weights, probs))

    def predict_one(self, text, threshold=None):
        if threshold is None:
            threshold = self.threshold
        prob = self.predict_proba_one(text)
        return int(prob >= threshold), prob

    def explain_one(self, text, top_k=6):
        label, prob = self.predict_one(text)
        label_name = "rumor" if label == 1 else "non-rumor"
        confidence = prob if label == 1 else 1.0 - prob

        terms = []
        sub_predictions = []
        weights = getattr(self, "weights", None) or [
            1.0 / len(self.models)
        ] * len(self.models)
        for idx, (name, model) in enumerate(zip(self.model_names, self.models)):
            sub = model.explain_one(text, top_k=top_k)
            terms.extend(sub["evidence_terms"])
            sub_predictions.append(
                {
                    "name": name,
                    "weight": weights[idx],
                    "label": sub["label"],
                    "prob_rumor": sub["prob_rumor"],
                    "confidence": sub["confidence"],
                }
            )
        terms = dedupe_terms(terms)[:top_k]

        if terms:
            evidence = ", ".join(terms)
            reason = (
                f"Ensemble model predicts {label_name} with confidence "
                f"{confidence:.2f}. "
                f"The result is averaged from {len(self.models)} TF-IDF "
                f"sub-models; main evidence terms include: {evidence}."
            )
        else:
            reason = (
                f"Ensemble model predicts {label_name} with confidence "
                f"{confidence:.2f}. "
                f"The result is averaged from {len(self.models)} TF-IDF "
                "sub-models."
            )

        return {
            "label": label,
            "label_name": label_name,
            "prob_rumor": prob,
            "confidence": confidence,
            "evidence_terms": terms,
            "sub_predictions": sub_predictions,
            "explanation": reason,
        }

    def evaluate(self, texts, labels):
        probs = [self.predict_proba_one(text) for text in texts]
        return metrics(labels, probs, self.threshold)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
