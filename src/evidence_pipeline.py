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

        final_prob, probability_guard = self._apply_probability_guard(
            fusion["final_prob"],
            retrieval_evidence["retrieval_statistics"],
        )
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
            "probability_guard": probability_guard,
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

    def _apply_probability_guard(self, prob, retrieval_stats):
        guard = getattr(self.ensemble, "probability_guard", None)
        if not guard:
            return prob, {"applied": False}
        if guard.get("type") == "retrieval_accuracy_guard":
            return self._apply_retrieval_accuracy_guard(prob, retrieval_stats, guard)
        if guard.get("type") != "nonrumor_retrieval_guard":
            return prob, {"applied": False, "reason": "unknown_guard_type"}

        return self._apply_nonrumor_retrieval_guard(prob, retrieval_stats, guard)

    def _apply_nonrumor_retrieval_guard(self, prob, retrieval_stats, guard):
        min_similarity = float(guard.get("min_similarity", 1.0))
        max_rumor_score = float(guard.get("max_weighted_rumor_score", 0.0))
        min_confidence = float(guard.get("min_confidence", 0.0))
        delta = float(guard.get("delta", 0.0))

        max_similarity = retrieval_stats.get("max_similarity", 0.0)
        weighted_rumor_score = retrieval_stats.get("weighted_rumor_score", 0.5)
        retrieval_confidence = retrieval_stats.get("retrieval_confidence", 0.0)
        applied = (
            max_similarity >= min_similarity
            and weighted_rumor_score <= max_rumor_score
            and retrieval_confidence >= min_confidence
            and delta > 0.0
        )
        if not applied:
            return prob, {
                "applied": False,
                "type": guard.get("type"),
                "thresholds": {
                    "min_similarity": min_similarity,
                    "max_weighted_rumor_score": max_rumor_score,
                    "min_confidence": min_confidence,
                    "delta": delta,
                },
            }

        adjusted_prob = max(0.0, prob - delta)
        return adjusted_prob, {
            "applied": True,
            "type": guard.get("type"),
            "prob_before": prob,
            "prob_after": adjusted_prob,
            "delta": delta,
            "reason": "high-confidence retrieval support for non-rumor",
            "retrieval_snapshot": {
                "max_similarity": max_similarity,
                "weighted_rumor_score": weighted_rumor_score,
                "retrieval_confidence": retrieval_confidence,
            },
        }

    def _apply_retrieval_accuracy_guard(self, prob, retrieval_stats, guard):
        original_prob = prob
        applied_rules = []

        nonrumor_guard = guard.get("nonrumor_guard")
        if nonrumor_guard:
            adjusted_prob, nonrumor_result = self._apply_nonrumor_retrieval_guard(
                prob,
                retrieval_stats,
                nonrumor_guard,
            )
            if nonrumor_result.get("applied"):
                applied_rules.append(
                    {
                        "name": "nonrumor_retrieval_guard",
                        "delta": -nonrumor_result["delta"],
                        "prob_before": prob,
                        "prob_after": adjusted_prob,
                    }
                )
                prob = adjusted_prob

        rescue_guard = guard.get("rumor_rescue_guard")
        if rescue_guard and self._should_apply_rumor_rescue(
            prob,
            retrieval_stats,
            rescue_guard,
        ):
            delta = float(rescue_guard.get("delta", 0.0))
            adjusted_prob = min(1.0, prob + delta)
            applied_rules.append(
                {
                    "name": "rumor_retrieval_rescue",
                    "delta": delta,
                    "prob_before": prob,
                    "prob_after": adjusted_prob,
                }
            )
            prob = adjusted_prob

        if not applied_rules:
            return prob, {
                "applied": False,
                "type": guard.get("type"),
                "rules": {
                    "nonrumor_guard": nonrumor_guard,
                    "rumor_rescue_guard": rescue_guard,
                },
            }

        return prob, {
            "applied": True,
            "type": guard.get("type"),
            "prob_before": original_prob,
            "prob_after": prob,
            "applied_rules": applied_rules,
            "reason": "retrieval evidence adjusted the final probability",
            "retrieval_snapshot": {
                "max_similarity": retrieval_stats.get("max_similarity", 0.0),
                "weighted_rumor_score": retrieval_stats.get(
                    "weighted_rumor_score",
                    0.5,
                ),
                "retrieval_confidence": retrieval_stats.get(
                    "retrieval_confidence",
                    0.0,
                ),
            },
        }

    def _should_apply_rumor_rescue(self, prob, retrieval_stats, guard):
        return (
            prob >= float(guard.get("min_prob", 0.0))
            and prob < float(guard.get("max_prob", 1.0))
            and retrieval_stats.get("weighted_rumor_score", 0.5)
            >= float(guard.get("min_weighted_rumor_score", 1.0))
            and retrieval_stats.get("max_similarity", 0.0)
            >= float(guard.get("min_similarity", 1.0))
            and retrieval_stats.get("retrieval_confidence", 0.0)
            >= float(guard.get("min_confidence", 0.0))
            and float(guard.get("delta", 0.0)) > 0.0
        )


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

    guard = evidence.get("probability_guard", {})
    if guard.get("applied"):
        rules = [item["name"] for item in guard.get("applied_rules", [])]
        if "rumor_retrieval_rescue" in rules:
            pieces.append(
                "Retrieval evidence rescued a borderline rumor case, moving "
                f"the rumor probability from {guard['prob_before']:.2f} to "
                f"{guard['prob_after']:.2f}. "
            )
        else:
            pieces.append(
                "A high-confidence non-rumor retrieval guard lowered the final "
                f"rumor probability from {guard['prob_before']:.2f} to "
                f"{guard['prob_after']:.2f}. "
            )

    return "".join(pieces)
