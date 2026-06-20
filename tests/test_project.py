import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import evaluate_rows
from src.defense import sanitize_text
from src.ensemble_model import EnsembleRumorModel
from src.evidence import build_retrieval_evidence_features
from src.evidence_pipeline import RumorDetectionPipeline
from src.llm_explainer import (
    SchoolLLMExplainer,
    build_chat_completions_url,
    build_explanation_prompt,
    extract_chat_content,
)
from src.prediction_service import RumorPredictionService
from src.retriever import TfidfEvidenceRetriever
from src.text_model import find_best_threshold, read_csv, stratified_train_dev_split


class ConstantModel:
    threshold = 0.5

    def __init__(self, prob):
        self.prob = prob

    def predict_proba_one(self, text):
        return self.prob

    def explain_one(self, text, top_k=6):
        label = int(self.prob >= self.threshold)
        return {
            "label": label,
            "label_name": "rumor" if label else "non-rumor",
            "prob_rumor": self.prob,
            "confidence": self.prob if label else 1.0 - self.prob,
            "evidence_terms": ["breaking", "report"],
            "explanation": "base explanation",
        }


class RetrieverModel:
    def vectorize_one(self, text):
        tokens = str(text).lower().split()
        return {idx: 1.0 for idx, _ in enumerate(tokens)}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label", "event"])
        writer.writeheader()
        writer.writerows(rows)


def tiny_rows():
    rows = []
    for event in range(2):
        for idx in range(8):
            rows.append(
                {
                    "id": f"r-{event}-{idx}",
                    "text": f"breaking rumor report event {event} claim {idx}",
                    "label": 1,
                    "event": event,
                }
            )
            rows.append(
                {
                    "id": f"n-{event}-{idx}",
                    "text": f"official source confirms event {event} update {idx}",
                    "label": 0,
                    "event": event,
                }
            )
    return rows


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeLLMExplainer:
    model = "fake-school-model"

    def status(self):
        return {
            "configured": True,
            "api_url": "https://api.school.edu/v1/chat/completions",
            "model": self.model,
            "has_api_key": True,
        }

    def explain(self, prediction):
        return f"LLM evidence for {prediction['label_name']}"


def without_llm_env():
    env = os.environ.copy()
    for key in [
        "SCHOOL_LLM_API_KEY",
        "SCHOOL_LLM_API_URL",
        "SCHOOL_LLM_BASE_URL",
        "SCHOOL_LLM_MODEL",
    ]:
        env.pop(key, None)
    return env


class FinalPipelineTests(unittest.TestCase):
    def test_read_csv_accepts_utf8_bom_and_quoted_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text(
                '\ufeff"id","text","label","event"\n'
                '"1","Breaking report",1,2\n',
                encoding="utf-8",
            )

            rows = read_csv(path)

        self.assertEqual(rows[0]["id"], "1")
        self.assertEqual(rows[0]["text"], "Breaking report")
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["event"], 2)

    def test_internal_dev_split_is_reproducible_and_disjoint(self):
        rows = tiny_rows()

        fit_a, dev_a = stratified_train_dev_split(rows, dev_ratio=0.25, seed=7)
        fit_b, dev_b = stratified_train_dev_split(rows, dev_ratio=0.25, seed=7)

        self.assertEqual([row["id"] for row in fit_a], [row["id"] for row in fit_b])
        self.assertEqual([row["id"] for row in dev_a], [row["id"] for row in dev_b])
        self.assertFalse({row["id"] for row in fit_a} & {row["id"] for row in dev_a})

    def test_defense_sanitization_is_shared_by_pipeline(self):
        self.assertEqual(sanitize_text("g\u043e\u200bv\u0435rnment"), "government")

    def test_retrieval_features_summarize_top_k_evidence(self):
        rows = [
            {"id": "1", "event": 0, "label": 1, "text": "breaking report confirmed"},
            {"id": "2", "event": 0, "label": 0, "text": "official report confirmed"},
            {"id": "3", "event": 0, "label": 1, "text": "breaking rumor report"},
        ]
        model = RetrieverModel()
        vectors = [model.vectorize_one(row["text"]) for row in rows]
        retriever = TfidfEvidenceRetriever(model, rows, vectors)

        evidence = build_retrieval_evidence_features(
            retriever,
            "breaking report",
            top_k=3,
            exclude_exact=True,
        )

        stats = evidence["retrieval_statistics"]
        self.assertEqual(stats["retrieved_count"], 3)
        self.assertGreater(stats["retrieved_rumor_ratio"], 0.5)
        self.assertIn("weighted_rumor_score", stats)

    def test_evidence_pipeline_uses_saved_evidence_threshold(self):
        ensemble = EnsembleRumorModel(
            models=[ConstantModel(0.55), ConstantModel(0.55)],
            model_names=["left", "right"],
            threshold=0.5,
        )
        ensemble.evidence_threshold = 0.6
        ensemble.evidence_weights = {"base": 1.0, "retrieval": 0.0, "structure": 0.0}

        result = RumorDetectionPipeline(ensemble).predict("sample text")

        self.assertEqual(result["baseline_label"], 1)
        self.assertEqual(result["label"], 0)
        self.assertEqual(result["threshold"], 0.6)
        self.assertIn("Final decision", result["explanation"])

    def test_probability_guard_adjusts_high_confidence_nonrumor_retrieval(self):
        ensemble = EnsembleRumorModel(
            models=[ConstantModel(0.55), ConstantModel(0.55)],
            model_names=["left", "right"],
            threshold=0.5,
        )
        ensemble.probability_guard = {
            "type": "nonrumor_retrieval_guard",
            "min_similarity": 0.3,
            "max_weighted_rumor_score": 0.4,
            "min_confidence": 0.25,
            "delta": 0.1,
        }
        pipeline = RumorDetectionPipeline(ensemble)

        adjusted, guard = pipeline._apply_probability_guard(
            0.55,
            {
                "max_similarity": 0.35,
                "weighted_rumor_score": 0.2,
                "retrieval_confidence": 0.5,
            },
        )

        self.assertAlmostEqual(adjusted, 0.45)
        self.assertTrue(guard["applied"])

    def test_accuracy_guard_can_rescue_retrieval_supported_rumors(self):
        ensemble = EnsembleRumorModel(
            models=[ConstantModel(0.4), ConstantModel(0.4)],
            model_names=["left", "right"],
            threshold=0.5,
        )
        ensemble.probability_guard = {
            "type": "retrieval_accuracy_guard",
            "nonrumor_guard": {
                "min_similarity": 0.3,
                "max_weighted_rumor_score": 0.4,
                "min_confidence": 0.25,
                "delta": 0.1,
            },
            "rumor_rescue_guard": {
                "min_weighted_rumor_score": 0.55,
                "min_similarity": 0.18,
                "min_confidence": 0.0,
                "min_prob": 0.0,
                "max_prob": 0.45,
                "delta": 0.09,
            },
        }
        pipeline = RumorDetectionPipeline(ensemble)

        adjusted, guard = pipeline._apply_probability_guard(
            0.4,
            {
                "max_similarity": 0.2,
                "weighted_rumor_score": 0.7,
                "retrieval_confidence": 0.3,
            },
        )

        self.assertAlmostEqual(adjusted, 0.49)
        self.assertTrue(guard["applied"])
        self.assertEqual(guard["applied_rules"][0]["name"], "rumor_retrieval_rescue")

    def test_llm_url_builder_uses_chat_completions_endpoint(self):
        self.assertEqual(
            build_chat_completions_url("https://api.school.edu/v1"),
            "https://api.school.edu/v1/chat/completions",
        )
        self.assertEqual(
            build_chat_completions_url(
                "https://api.school.edu/v1/chat/completions"
            ),
            "https://api.school.edu/v1/chat/completions",
        )

    def test_llm_response_parser_extracts_message_content(self):
        content = extract_chat_content(
            {"choices": [{"message": {"content": "这是大模型解释"}}]}
        )

        self.assertEqual(content, "这是大模型解释")

    def test_school_llm_explainer_calls_chat_api(self):
        explainer = SchoolLLMExplainer(
            api_key="test-key",
            api_url="https://api.school.edu/v1/chat/completions",
            model="school-model",
        )
        prediction = {
            "label_name": "non-rumor",
            "prob_rumor": 0.2,
            "confidence": 0.8,
            "threshold": 0.5,
            "evidence": {},
        }

        with patch(
            "urllib.request.urlopen",
            return_value=FakeHttpResponse(
                {"choices": [{"message": {"content": "这是学校大模型解释"}}]}
            ),
        ) as mocked_urlopen:
            explanation = explainer.explain(prediction)

        request = mocked_urlopen.call_args[0][0]
        self.assertEqual(explanation, "这是学校大模型解释")
        self.assertEqual(request.full_url, "https://api.school.edu/v1/chat/completions")

    def test_prediction_service_adds_llm_evidence_by_default(self):
        ensemble = EnsembleRumorModel(
            models=[ConstantModel(0.7), ConstantModel(0.7)],
            model_names=["left", "right"],
            threshold=0.5,
        )
        service = RumorPredictionService(llm_explainer=FakeLLMExplainer())
        service._pipeline = RumorDetectionPipeline(ensemble)

        result = service.predict("sample text")

        self.assertIn("evidence", result)
        self.assertIn("llm_evidence", result)
        self.assertTrue(result["llm_evidence"]["available"])
        self.assertEqual(
            result["llm_evidence"]["explanation"],
            "LLM evidence for rumor",
        )

    def test_llm_prompt_keeps_final_model_decision(self):
        prediction = {
            "input_text": "sample claim raw tweet",
            "label_name": "rumor",
            "prob_rumor": 0.72,
            "confidence": 0.72,
            "threshold": 0.5,
            "explanation": "base explanation",
            "evidence": {
                "normalized_text": "sample claim",
                "baseline": {
                    "label_name": "rumor",
                    "prob_rumor": 0.7,
                    "threshold": 0.5,
                },
                "retrieval_statistics": {
                    "retrieved_count": 2,
                    "retrieved_rumor_ratio": 1.0,
                    "retrieved_nonrumor_ratio": 0.0,
                    "max_similarity": 0.3,
                    "weighted_rumor_score": 1.0,
                    "retrieval_confidence": 0.8,
                },
                "decision_factors": [],
                "keyword_contributions": [{"term": "breaking"}],
                "claim_structure_features": {"signals": ["hashtag_count=1"]},
                "probability_guard": {"applied": False},
                "retrieved_cases": [
                    {"label": 1, "score": 0.3, "text": "similar rumor"}
                ],
            },
        }

        prompt = build_explanation_prompt(prediction)

        self.assertIn('"label_name": "rumor"', prompt)
        self.assertIn('"original_tweet": "sample claim raw tweet"', prompt)
        self.assertIn('"model_evidence_terms"', prompt)
        self.assertIn('"rag_similar_samples"', prompt)
        self.assertIn("不能修改模型给出的最终结论", prompt)
        self.assertIn("breaking", prompt)

    def test_f1_recall_threshold_objective_prefers_recall_near_ties(self):
        labels = [1, 1, 1, 0, 0]
        probs = [0.49, 0.51, 0.70, 0.50, 0.10]

        result = find_best_threshold(
            labels,
            probs,
            start=0.49,
            end=0.51,
            step=0.01,
            objective="f1_recall",
        )

        self.assertEqual(result["threshold"], 0.49)
        self.assertEqual(result["recall"], 1.0)

    def test_save_load_preserves_final_pipeline_parameters(self):
        ensemble = EnsembleRumorModel(
            models=[ConstantModel(0.7), ConstantModel(0.6)],
            model_names=["left", "right"],
            threshold=0.45,
            weights=[0.25, 0.75],
        )
        ensemble.evidence_threshold = 0.58
        ensemble.evidence_top_k = 3
        ensemble.retrieval_min_similarity = 0.2
        ensemble.evidence_weights = {"base": 0.9, "retrieval": 0.1, "structure": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pkl"
            ensemble.save(model_path)
            loaded = RumorDetectionPipeline.load(model_path)

        self.assertEqual(loaded.ensemble.evidence_threshold, 0.58)
        self.assertEqual(loaded.ensemble.evidence_top_k, 3)
        self.assertEqual(loaded.retrieval_min_similarity, 0.2)
        self.assertEqual(loaded.evidence_weights, ensemble.evidence_weights)

    def test_load_with_missing_train_path_fails_clearly(self):
        ensemble = EnsembleRumorModel(
            models=[ConstantModel(0.7), ConstantModel(0.6)],
            model_names=["left", "right"],
            threshold=0.45,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pkl"
            missing_train = Path(tmp) / "missing.csv"
            ensemble.save(model_path)

            with self.assertRaisesRegex(FileNotFoundError, "training evidence file"):
                RumorDetectionPipeline.load(model_path, train_path=missing_train)

    def test_evaluate_rows_uses_final_pipeline_outputs(self):
        ensemble = EnsembleRumorModel(
            models=[ConstantModel(0.7), ConstantModel(0.7)],
            model_names=["left", "right"],
            threshold=0.5,
        )
        ensemble.evidence_threshold = 0.6
        pipeline = RumorDetectionPipeline(ensemble)
        rows = [
            {"id": "1", "event": 0, "text": "breaking report", "label": 1},
            {"id": "2", "event": 1, "text": "official source", "label": 1},
        ]

        overall, by_event, cases = evaluate_rows(pipeline, rows)

        self.assertEqual(overall["threshold"], 0.6)
        self.assertEqual(overall["accuracy"], 1.0)
        self.assertEqual(set(by_event), {"0", "1"})
        self.assertEqual(cases[0]["pred_label"], 1)
        self.assertIn("raw_prediction", cases[0])

    def test_train_evaluate_predict_cli_on_tiny_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train_path = tmp / "train.csv"
            val_path = tmp / "val.csv"
            model_path = tmp / "model.pkl"
            metrics_path = tmp / "metrics.json"
            out_dir = tmp / "outputs"
            write_csv(train_path, tiny_rows())
            write_csv(val_path, tiny_rows()[:8])

            subprocess.run(
                [
                    sys.executable,
                    "train.py",
                    "--train",
                    str(train_path),
                    "--val",
                    str(val_path),
                    "--model",
                    str(model_path),
                    "--metrics",
                    str(metrics_path),
                    "--dev-ratio",
                    "0.25",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    sys.executable,
                    "evaluate.py",
                    "--model",
                    str(model_path),
                    "--data",
                    str(val_path),
                    "--train",
                    str(train_path),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            prediction = subprocess.run(
                [
                    sys.executable,
                    "predict.py",
                    "--model",
                    str(model_path),
                    "--train",
                    str(train_path),
                    "--text",
                    "breaking rumor report",
                ],
                check=True,
                capture_output=True,
                env=without_llm_env(),
                text=True,
            )

            payload = json.loads(prediction.stdout)

            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertTrue((out_dir / "evaluation.json").exists())
            self.assertIn(payload["label"], (0, 1))
            self.assertIn("evidence", payload)
            self.assertIn("llm_evidence", payload)
            self.assertTrue(payload["llm_evidence"]["enabled"])
            self.assertFalse(payload["llm_evidence"]["available"])
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(
                metrics_payload["threshold_objective"],
                "precision_calibrated_accuracy",
            )
            self.assertEqual(metrics_payload["pipeline"], "accuracy_rescue_fusion")
            self.assertIn("threshold_calibration", metrics_payload)
            self.assertIn("probability_guard", metrics_payload)
            self.assertIn("validation_size", metrics_payload)
            self.assertNotIn("test_size", metrics_payload)


if __name__ == "__main__":
    unittest.main()
