import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.evidence_pipeline import RumorDetectionPipeline
from src.text_model import metrics, read_csv, save_json


ZERO_WIDTH_SPACE = "\u200b"
HOMOGLYPH_ATTACK_MAP = str.maketrans(
    {
        "a": "\u0430",
        "e": "\u0435",
        "o": "\u043e",
        "p": "\u0440",
        "c": "\u0441",
        "y": "\u0443",
        "x": "\u0445",
        "i": "\u0456",
        "A": "\u0410",
        "B": "\u0412",
        "E": "\u0415",
        "K": "\u041a",
        "M": "\u041c",
        "H": "\u041d",
        "O": "\u041e",
        "P": "\u0420",
        "C": "\u0421",
        "T": "\u0422",
        "X": "\u0425",
    }
)


def insert_zero_width(text):
    text = str(text)
    for idx, char in enumerate(text):
        if char.isalpha():
            return text[: idx + 1] + ZERO_WIDTH_SPACE + text[idx + 1 :]
    return text + ZERO_WIDTH_SPACE


def substitute_homoglyphs(text):
    return str(text).translate(HOMOGLYPH_ATTACK_MAP)


def stretch_repeated_characters(text):
    text = str(text)
    for idx, char in enumerate(text):
        if char.isalpha():
            return text[:idx] + char * 4 + text[idx + 1 :]
    return text


def perturb_whitespace(text):
    return "   ".join(str(text).split())


def perturb_casing(text):
    chars = []
    alpha_index = 0
    for char in str(text):
        if char.isalpha():
            chars.append(char.upper() if alpha_index % 2 == 0 else char.lower())
            alpha_index += 1
        else:
            chars.append(char)
    return "".join(chars)


def perturb_punctuation(text):
    text = str(text).strip()
    if not text:
        return "!!!"
    return f"{text} !!!"


ATTACKS = {
    "zero_width": insert_zero_width,
    "homoglyph": substitute_homoglyphs,
    "repeat_chars": stretch_repeated_characters,
    "whitespace": perturb_whitespace,
    "casing": perturb_casing,
    "punctuation": perturb_punctuation,
}


PREDICTION_FIELDS = [
    "id",
    "event",
    "attack",
    "true_label",
    "clean_pred",
    "attacked_pred",
    "consistent",
    "clean_prob_rumor",
    "attacked_prob_rumor",
    "clean_confidence",
    "attacked_confidence",
    "text",
    "attacked_text",
]


def parse_attack_names(value):
    if not value or value == "all":
        return list(ATTACKS)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in ATTACKS]
    if unknown:
        raise ValueError(
            "unknown attack(s): "
            + ", ".join(unknown)
            + f". Available attacks: {', '.join(ATTACKS)}"
        )
    return names


def predict_rows(pipeline, rows, top_k=None):
    return [
        pipeline.predict(row["text"], top_k=top_k, include_explanation=False)
        for row in rows
    ]


def evaluate_robustness(pipeline, rows, attack_names=None, top_k=None):
    attack_names = attack_names or list(ATTACKS)
    clean_predictions = predict_rows(pipeline, rows, top_k=top_k)
    labels = [row["label"] for row in rows]
    clean_probs = [prediction["prob_rumor"] for prediction in clean_predictions]
    threshold = (
        clean_predictions[0]["threshold"]
        if clean_predictions
        else getattr(pipeline.ensemble, "evidence_threshold", pipeline.ensemble.threshold)
    )
    clean = metrics(labels, clean_probs, threshold)

    attack_reports = {}
    prediction_rows = []
    all_attacked_labels = []
    all_attacked_probs = []
    total_consistent = 0
    total_attacked = 0

    for attack_name in attack_names:
        attack_fn = ATTACKS[attack_name]
        attacked_rows = [
            {
                **row,
                "text": attack_fn(row["text"]),
                "clean_text": row["text"],
            }
            for row in rows
        ]
        attacked_predictions = predict_rows(pipeline, attacked_rows, top_k=top_k)
        attacked_probs = [
            prediction["prob_rumor"] for prediction in attacked_predictions
        ]
        attacked = metrics(labels, attacked_probs, threshold)
        consistent_count = 0

        for row, clean_prediction, attacked_prediction, attacked_row in zip(
            rows, clean_predictions, attacked_predictions, attacked_rows
        ):
            consistent = clean_prediction["label"] == attacked_prediction["label"]
            consistent_count += int(consistent)
            prediction_rows.append(
                {
                    "id": row["id"],
                    "event": row["event"],
                    "attack": attack_name,
                    "true_label": row["label"],
                    "clean_pred": clean_prediction["label"],
                    "attacked_pred": attacked_prediction["label"],
                    "consistent": int(consistent),
                    "clean_prob_rumor": f"{clean_prediction['prob_rumor']:.6f}",
                    "attacked_prob_rumor": f"{attacked_prediction['prob_rumor']:.6f}",
                    "clean_confidence": f"{clean_prediction['confidence']:.6f}",
                    "attacked_confidence": f"{attacked_prediction['confidence']:.6f}",
                    "text": row["text"],
                    "attacked_text": attacked_row["text"],
                }
            )

        consistency = consistent_count / max(len(rows), 1)
        attack_reports[attack_name] = {
            **attacked,
            "sample_count": len(rows),
            "accuracy_drop": clean["accuracy"] - attacked["accuracy"],
            "prediction_consistency": consistency,
        }
        total_consistent += consistent_count
        total_attacked += len(rows)
        all_attacked_labels.extend(labels)
        all_attacked_probs.extend(attacked_probs)

    aggregate_attacked = metrics(all_attacked_labels, all_attacked_probs, threshold)
    aggregate = {
        **aggregate_attacked,
        "sample_count": total_attacked,
        "accuracy_drop": clean["accuracy"] - aggregate_attacked["accuracy"],
        "prediction_consistency": total_consistent / max(total_attacked, 1),
    }

    by_event = defaultdict(lambda: {"total": 0, "consistent": 0})
    for item in prediction_rows:
        bucket = by_event[str(item["event"])]
        bucket["total"] += 1
        bucket["consistent"] += int(item["consistent"])
    by_event = {
        event: {
            "attacked_count": data["total"],
            "prediction_consistency": data["consistent"] / max(data["total"], 1),
        }
        for event, data in sorted(by_event.items())
    }

    return {
        "clean": clean,
        "attacked": aggregate,
        "attacks": attack_reports,
        "by_event": by_event,
        "prediction_rows": prediction_rows,
        "threshold": threshold,
    }


def write_prediction_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_percent(value):
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate robustness with synthetic character-level perturbations."
        )
    )
    parser.add_argument("--model", default="models/main_fusion.pkl")
    parser.add_argument("--data", default="val.csv")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--attacks",
        default="all",
        help=f"comma-separated attacks or 'all'. Available: {', '.join(ATTACKS)}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional number of clean rows to evaluate for quick checks",
    )
    args = parser.parse_args()

    attack_names = parse_attack_names(args.attacks)
    rows = read_csv(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    pipeline = RumorDetectionPipeline.load(args.model, train_path=args.train)
    result = evaluate_robustness(
        pipeline,
        rows,
        attack_names=attack_names,
        top_k=args.top_k,
    )

    out_dir = Path(args.out_dir)
    report = {
        "model_path": args.model,
        "train_path": args.train,
        "data_path": args.data,
        "clean_sample_count": len(rows),
        "attack_names": attack_names,
        "top_k": args.top_k
        if args.top_k is not None
        else getattr(pipeline.ensemble, "evidence_top_k", 5),
        "threshold": result["threshold"],
        "clean": result["clean"],
        "attacked": result["attacked"],
        "attacks": result["attacks"],
        "by_event": result["by_event"],
    }
    save_json(report, out_dir / "robustness_report.json")
    write_prediction_rows(
        out_dir / "robustness_predictions.csv",
        result["prediction_rows"],
    )

    print("Robustness")
    print(f"  clean accuracy:       {format_percent(report['clean']['accuracy'])}")
    print(
        "  attacked accuracy:    "
        f"{format_percent(report['attacked']['accuracy'])}"
    )
    print(
        "  accuracy drop:        "
        f"{format_percent(report['attacked']['accuracy_drop'])}"
    )
    print(
        "  consistency:          "
        f"{format_percent(report['attacked']['prediction_consistency'])}"
    )
    print(f"  attacks:              {', '.join(attack_names)}")
    print(f"Saved reports to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
