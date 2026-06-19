"""Final evidence-aware rumor detection pipeline."""

from pathlib import Path

from src.claim_structure import extract_claim_structure_features
from src.defense import sanitize_text
from src.ensemble_model import EnsembleRumorModel
from src.evidence import build_retrieval_evidence_features, compact_keyword_evidence


DEFAULT_EVIDENCE_WEIGHTS = {
    "base": 0.94,
    "retrieval": 0.04,
    "structure": 0.02,
}


class RumorDetectionPipeline:
    """Evidence-aware inference used by training, evaluation, and prediction."""

    def __init__(
        self,
        ensemble,
        retriever=None,
        evidence_weights=None,
        retrieval_min_similarity=None,
        retrieval_min_confidence=None,
        use_retrieval=True,
        use_structure=True,
    ):
        self.ensemble = ensemble
        self.retriever = retriever
        self.evidence_weights = dict(DEFAULT_EVIDENCE_WEIGHTS)
        saved_weights = getattr(ensemble, "evidence_weights", None)
        if saved_weights:
            self.evidence_weights.update(saved_weights)
        if evidence_weights:
            self.evidence_weights.update(evidence_weights)
        self.retrieval_min_similarity = (
            retrieval_min_similarity
            if retrieval_min_similarity is not None
            else getattr(ensemble, "retrieval_min_similarity", 0.0)
        )
        self.retrieval_min_confidence = (
            retrieval_min_confidence
            if retrieval_min_confidence is not None
            else getattr(ensemble, "retrieval_min_confidence", 0.0)
        )
        self.use_retrieval = use_retrieval
        self.use_structure = use_structure

    @classmethod
    def load(cls, model_path, train_path=None, **kwargs):
        ensemble = EnsembleRumorModel.load(model_path)
        retriever = None
        if train_path:
            train_path = Path(train_path)
            if not train_path.exists():
                raise FileNotFoundError(
                    f"training evidence file not found: {train_path}. "
                    "Pass --train with the CSV used to train the model."
                )
            from src.retriever import TfidfEvidenceRetriever

            retriever = TfidfEvidenceRetriever.from_csv(ensemble.models[0], train_path)
        return cls(ensemble, retriever=retriever, **kwargs)

    def _base_prediction(self, text):
        prob = self.ensemble.predict_proba_one(text)
        return prob, self.ensemble.threshold, {"tfidf_ensemble": prob}

    def predict(
        self,
        text,
        top_k=None,
        include_explanation=True,
        exclude_exact_evidence=True,
    ):
        if top_k is None:
            top_k = getattr(self.ensemble, "evidence_top_k", 5)
        normalized_text = sanitize_text(text)
        base_prob, base_threshold, components = self._base_prediction(normalized_text)
        threshold = getattr(self.ensemble, "evidence_threshold", base_threshold)
        base_result = self.ensemble.explain_one(normalized_text)

        retrieval_evidence = build_retrieval_evidence_features(
            self.retriever if self.use_retrieval else None,
            normalized_text,
            top_k=top_k,
            exclude_exact=exclude_exact_evidence,
        )
        structure_features = (
            extract_claim_structure_features(normalized_text)
            if self.use_structure
            else {"structure_score": 0.5, "signals": []}
        )
        fusion = self._evidence_aware_fusion(
            base_prob,
            retrieval_evidence["retrieval_statistics"],
            structure_features,
        )

        final_prob = fusion["final_prob"]
        label = int(final_prob >= threshold)
        confidence = final_prob if label == 1 else 1.0 - final_prob
        label_name = label_to_name(label)

        evidence = {
            "normalized_text": normalized_text,
            "keyword_contributions": compact_keyword_evidence(base_result),
            "retrieved_cases": retrieval_evidence["retrieved_cases"],
            "retrieval_statistics": retrieval_evidence["retrieval_statistics"],
            "claim_structure_features": structure_features,
            "decision_factors": fusion["decision_factors"],
            "baseline": {
                "label": int(base_prob >= base_threshold),
                "label_name": label_to_name(int(base_prob >= base_threshold)),
                "prob_rumor": base_prob,
                "threshold": base_threshold,
                "components": components,
            },
        }

        result = {
            "label": label,
            "label_name": label_name,
            "prob_rumor": final_prob,
            "confidence": confidence,
            "threshold": threshold,
            "baseline_label": evidence["baseline"]["label"],
            "baseline_prob_rumor": base_prob,
            "evidence_aware": True,
            "evidence": evidence,
        }
        if include_explanation:
            result["explanation"] = build_evidence_first_explanation(result)
        return result

    def predict_proba_one(self, text, top_k=None):
        return self.predict(text, top_k=top_k, include_explanation=False)["prob_rumor"]

    def predict_one(self, text, threshold=None, top_k=None):
        result = self.predict(text, top_k=top_k, include_explanation=False)
        if threshold is None:
            return result["label"], result["prob_rumor"]
        return int(result["prob_rumor"] >= threshold), result["prob_rumor"]

    def _evidence_aware_fusion(self, base_prob, retrieval_stats, structure_features):
        base_w = self.evidence_weights["base"]
        retrieval_w = self.evidence_weights["retrieval"]
        structure_w = self.evidence_weights["structure"]

        retrieval_conf = retrieval_stats.get("retrieval_confidence", 0.0)
        if retrieval_stats.get("max_similarity", 0.0) < self.retrieval_min_similarity:
            retrieval_conf = 0.0
        if retrieval_conf < self.retrieval_min_confidence:
            retrieval_conf = 0.0

        effective_retrieval_w = retrieval_w * retrieval_conf
        effective_structure_w = structure_w
        effective_base_w = base_w + (retrieval_w - effective_retrieval_w)
        total = effective_base_w + effective_retrieval_w + effective_structure_w

        retrieval_score = retrieval_stats.get("weighted_rumor_score", 0.5)
        structure_score = structure_features.get("structure_score", 0.5)
        final_prob = (
            effective_base_w * base_prob
            + effective_retrieval_w * retrieval_score
            + effective_structure_w * structure_score
        ) / max(total, 1e-12)

        return {
            "final_prob": min(max(final_prob, 0.0), 1.0),
            "decision_factors": [
                {
                    "name": "base_model",
                    "score": round(base_prob, 4),
                    "weight": round(effective_base_w / total, 4),
                    "description": "TF-IDF ensemble probability",
                },
                {
                    "name": "retrieval_evidence",
                    "score": round(retrieval_score, 4),
                    "weight": round(effective_retrieval_w / total, 4),
                    "description": "top-k retrieved label support",
                },
                {
                    "name": "claim_structure",
                    "score": round(structure_score, 4),
                    "weight": round(effective_structure_w / total, 4),
                    "description": "claim-structure risk score",
                },
            ],
        }


def label_to_name(label):
    return "rumor" if int(label) == 1 else "non-rumor"


def build_evidence_first_explanation(result):
    evidence = result["evidence"]
    pieces = [
        f"Final decision: {result['label_name']} with confidence "
        f"{result['confidence']:.2f}. "
    ]

    baseline = evidence["baseline"]
    pieces.append(
        "The base TF-IDF ensemble gives rumor probability "
        f"{baseline['prob_rumor']:.2f} and baseline label "
        f"{baseline['label_name']}. "
    )

    retrieval = evidence["retrieval_statistics"]
    if retrieval["retrieved_count"]:
        pieces.append(
            f"Retrieved {retrieval['retrieved_count']} similar training cases: "
            f"rumor ratio {retrieval['retrieved_rumor_ratio']:.2f}, "
            f"non-rumor ratio {retrieval['retrieved_nonrumor_ratio']:.2f}. "
        )
    else:
        pieces.append(
            "No similar training cases were retrieved, so retrieval evidence "
            "does not materially affect the decision. "
        )

    keywords = [item["term"] for item in evidence["keyword_contributions"][:5]]
    if keywords:
        pieces.append(f"Keyword evidence: {', '.join(keywords)}. ")

    structure = evidence["claim_structure_features"]
    if structure.get("signals"):
        pieces.append(
            f"Claim-structure signals: {', '.join(structure['signals'][:4])}. "
        )

    return "".join(pieces)