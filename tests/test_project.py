import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate_rows
from src.defense import sanitize_text
from src.ensemble_model import EnsembleRumorModel
from src.evidence import build_retrieval_evidence_features
from src.evidence_pipeline import RumorDetectionPipeline
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
                text=True,
            )

            payload = json.loads(prediction.stdout)

            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertTrue((out_dir / "evaluation.json").exists())
            self.assertIn(payload["label"], (0, 1))
            self.assertIn("evidence", payload)
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["threshold_objective"], "f1_recall")
            self.assertIn("validation_size", metrics_payload)
            self.assertNotIn("test_size", metrics_payload)


if __name__ == "__main__":
    unittest.main()