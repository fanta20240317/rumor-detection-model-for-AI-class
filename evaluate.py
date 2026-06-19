import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.evidence_pipeline import RumorDetectionPipeline
from src.text_model import metrics, read_csv, save_json


CASE_FIELDS = [
    "id",
    "event",
    "text",
    "true_label",
    "pred_label",
    "prob_rumor",
    "confidence",
    "threshold",
    "baseline_prob_rumor",
    "retrieved_count",
    "retrieved_rumor_ratio",
    "retrieved_nonrumor_ratio",
    "explanation",
]


def write_cases(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value):
    return f"{value * 100:.2f}%"


def evaluate_rows(pipeline, rows, top_k=None):
    predictions = [
        pipeline.predict(row["text"], top_k=top_k, include_explanation=True)
        for row in rows
    ]
    labels = [row["label"] for row in rows]
    probs = [item["prob_rumor"] for item in predictions]
    threshold = (
        predictions[0]["threshold"]
        if predictions
        else getattr(pipeline.ensemble, "evidence_threshold", pipeline.ensemble.threshold)
    )
    overall = metrics(labels, probs, threshold)

    by_event = {}
    event_indices = defaultdict(list)
    for idx, row in enumerate(rows):
        event_indices[row["event"]].append(idx)
    for event, indices in sorted(event_indices.items()):
        event_labels = [labels[i] for i in indices]
        event_probs = [probs[i] for i in indices]
        result = metrics(event_labels, event_probs, threshold)
        result["count"] = len(indices)
        by_event[str(event)] = result

    cases = []
    for row, prediction in zip(rows, predictions):
        stats = prediction["evidence"]["retrieval_statistics"]
        cases.append(
            {
                "id": row["id"],
                "event": row["event"],
                "text": row["text"],
                "true_label": row["label"],
                "pred_label": prediction["label"],
                "prob_rumor": f"{prediction['prob_rumor']:.6f}",
                "confidence": f"{prediction['confidence']:.6f}",
                "threshold": f"{prediction['threshold']:.6f}",
                "baseline_prob_rumor": f"{prediction['baseline_prob_rumor']:.6f}",
                "retrieved_count": stats["retrieved_count"],
                "retrieved_rumor_ratio": f"{stats['retrieved_rumor_ratio']:.4f}",
                "retrieved_nonrumor_ratio": f"{stats['retrieved_nonrumor_ratio']:.4f}",
                "explanation": prediction["explanation"],
                "raw_prediction": prediction,
            }
        )

    return overall, by_event, cases


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the final evidence-aware rumor detection pipeline."
    )
    parser.add_argument("--model", default="models/ensemble.pkl")
    parser.add_argument("--data", default="val.csv")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--case-limit", type=int, default=12)
    args = parser.parse_args()

    pipeline = RumorDetectionPipeline.load(args.model, train_path=args.train)
    rows = read_csv(args.data)
    overall, by_event, cases = evaluate_rows(pipeline, rows, top_k=args.top_k)

    out_dir = Path(args.out_dir)
    save_json(
        {
            "overall": overall,
            "by_event": by_event,
            "model_threshold": overall["threshold"],
            "pipeline": "evidence-aware",
            "model_path": args.model,
            "train_path": args.train,
            "data_path": args.data,
            "data_size": len(rows),
            "top_k": args.top_k
            if args.top_k is not None
            else getattr(pipeline.ensemble, "evidence_top_k", 5),
        },
        out_dir / "evaluation.json",
    )

    public_cases = [{k: v for k, v in item.items() if k != "raw_prediction"} for item in cases]
    write_cases(out_dir / "predictions.csv", public_cases)
    write_cases(
        out_dir / "correct_cases.csv",
        [item for item in public_cases if item["true_label"] == item["pred_label"]][
            : args.case_limit
        ],
    )
    write_cases(
        out_dir / "wrong_cases.csv",
        [item for item in public_cases if item["true_label"] != item["pred_label"]][
            : args.case_limit
        ],
    )
    save_json(
        {
            "correct_examples": [
                item["raw_prediction"]
                for item in cases
                if item["true_label"] == item["pred_label"]
            ][: args.case_limit],
            "wrong_examples": [
                item["raw_prediction"]
                for item in cases
                if item["true_label"] != item["pred_label"]
            ][: args.case_limit],
        },
        out_dir / "explain_cases.json",
    )

    print("Overall")
    print(f"  accuracy:  {format_percent(overall['accuracy'])}")
    print(f"  precision: {format_percent(overall['precision'])}")
    print(f"  recall:    {format_percent(overall['recall'])}")
    print(f"  f1:        {format_percent(overall['f1'])}")
    print(f"  threshold: {overall['threshold']:.6f}")
    print(
        "  confusion: "
        f"TN={overall['tn']} FP={overall['fp']} FN={overall['fn']} TP={overall['tp']}"
    )
    print(f"Saved reports to {out_dir.resolve()}")


if __name__ == "__main__":
    main()