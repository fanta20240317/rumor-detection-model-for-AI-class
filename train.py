import argparse
import itertools
from pathlib import Path

from src.claim_structure import extract_claim_structure_features
from src.ensemble_model import ENSEMBLE_SEARCH_CONFIGS, EnsembleRumorModel
from src.evidence import build_retrieval_evidence_features
from src.evidence_pipeline import RumorDetectionPipeline
from src.retriever import TfidfEvidenceRetriever
from src.text_model import (
    find_best_threshold,
    metrics,
    read_csv,
    save_json,
    stratified_train_dev_split,
)


def _better(candidate, best):
    if best is None:
        return True
    candidate_f1_bucket = round(candidate["f1"] / 0.002)
    best_f1_bucket = round(best["f1"] / 0.002)
    candidate_key = (
        candidate_f1_bucket,
        candidate["recall"],
        candidate["precision"],
        candidate["accuracy"],
        -candidate["threshold"],
    )
    best_key = (
        best_f1_bucket,
        best["recall"],
        best["precision"],
        best["accuracy"],
        -best["threshold"],
    )
    return candidate_key > best_key


def select_ensemble(candidates, dev_texts, dev_labels):
    sub_probs = [
        [candidate.predict_proba_one(text) for text in dev_texts]
        for candidate in candidates.models
    ]
    best = None
    for left_idx, right_idx in itertools.combinations(range(len(candidates.models)), 2):
        for step in range(21):
            left_weight = step / 20
            probs = [
                left_weight * left + (1.0 - left_weight) * right
                for left, right in zip(sub_probs[left_idx], sub_probs[right_idx])
            ]
            result = find_best_threshold(
                dev_labels,
                probs,
                start=0.20,
                end=0.80,
                step=0.001,
                objective="f1_recall",
            )
            result.update(
                {
                    "left_idx": left_idx,
                    "right_idx": right_idx,
                    "weights": [left_weight, 1.0 - left_weight],
                }
            )
            if _better(result, best):
                best = result

    return EnsembleRumorModel(
        models=[
            candidates.models[best["left_idx"]],
            candidates.models[best["right_idx"]],
        ],
        model_names=[
            candidates.model_names[best["left_idx"]],
            candidates.model_names[best["right_idx"]],
        ],
        threshold=best["threshold"],
        weights=best["weights"],
    ), best


def tune_evidence(model, fit_rows, dev_rows, top_k_values=(3, 5, 7)):
    retriever_model = model.models[0]
    vectors = [retriever_model.vectorize_one(row["text"]) for row in fit_rows]
    retriever = TfidfEvidenceRetriever(retriever_model, fit_rows, vectors)
    labels = [row["label"] for row in dev_rows]

    cache = {}
    for top_k in top_k_values:
        items = []
        for row in dev_rows:
            text = row["text"]
            retrieval = build_retrieval_evidence_features(
                retriever, text, top_k=top_k, exclude_exact=True
            )["retrieval_statistics"]
            items.append(
                {
                    "base_prob": model.predict_proba_one(text),
                    "retrieval": retrieval,
                    "structure": extract_claim_structure_features(text),
                }
            )
        cache[top_k] = items

    best = None
    for top_k in top_k_values:
        for retrieval_weight in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
            for structure_weight in (0.0, 0.01, 0.02, 0.03):
                if retrieval_weight + structure_weight >= 0.25:
                    continue
                for min_similarity in (0.0, 0.12, 0.18, 0.24, 0.30):
                    pipeline = RumorDetectionPipeline(
                        model,
                        evidence_weights={
                            "base": 1.0 - retrieval_weight - structure_weight,
                            "retrieval": retrieval_weight,
                            "structure": structure_weight,
                        },
                        retrieval_min_similarity=min_similarity,
                    )
                    probs = [
                        pipeline._evidence_aware_fusion(
                            item["base_prob"],
                            item["retrieval"],
                            item["structure"],
                        )["final_prob"]
                        for item in cache[top_k]
                    ]
                    result = find_best_threshold(
                        labels,
                        probs,
                        start=0.20,
                        end=0.80,
                        step=0.001,
                        objective="f1_recall",
                    )
                    result.update(
                        {
                            "top_k": top_k,
                            "evidence_weights": pipeline.evidence_weights,
                            "retrieval_min_similarity": min_similarity,
                        }
                    )
                    if _better(result, best):
                        best = result
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--val", default="val.csv")
    parser.add_argument("--model", default="models/ensemble.pkl")
    parser.add_argument("--metrics", default="outputs/metrics.json")
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    train_rows = read_csv(args.train)
    val_rows = read_csv(args.val)
    fit_rows, dev_rows = stratified_train_dev_split(
        train_rows, dev_ratio=args.dev_ratio, seed=args.seed
    )

    candidates = EnsembleRumorModel()
    candidates.fit(
        [row["text"] for row in fit_rows],
        [row["label"] for row in fit_rows],
        configs=ENSEMBLE_SEARCH_CONFIGS,
    )
    dev_texts = [row["text"] for row in dev_rows]
    dev_labels = [row["label"] for row in dev_rows]
    model, selection = select_ensemble(candidates, dev_texts, dev_labels)

    evidence_selection = tune_evidence(model, fit_rows, dev_rows)
    model.evidence_weights = evidence_selection["evidence_weights"]
    model.retrieval_min_similarity = evidence_selection["retrieval_min_similarity"]
    model.evidence_top_k = evidence_selection["top_k"]
    model.evidence_threshold = evidence_selection["threshold"]

    fit_vectors = [model.models[0].vectorize_one(row["text"]) for row in fit_rows]
    fit_retriever = TfidfEvidenceRetriever(model.models[0], fit_rows, fit_vectors)
    dev_pipeline = RumorDetectionPipeline(model, retriever=fit_retriever)

    retriever = TfidfEvidenceRetriever.from_csv(model.models[0], args.train)
    pipeline = RumorDetectionPipeline(model, retriever=retriever)

    dev_probs = [
        dev_pipeline.predict_proba_one(text, top_k=model.evidence_top_k)
        for text in dev_texts
    ]
    val_texts = [row["text"] for row in val_rows]
    val_labels = [row["label"] for row in val_rows]
    val_probs = [
        pipeline.predict_proba_one(text, top_k=model.evidence_top_k)
        for text in val_texts
    ]

    dev_result = metrics(dev_labels, dev_probs, model.evidence_threshold)
    validation_result = metrics(val_labels, val_probs, model.evidence_threshold)
    report = {
        **validation_result,
        "selected_models": model.model_names,
        "weights": model.weights,
        "model_selection": selection,
        "evidence_selection": evidence_selection,
        "train_size": len(train_rows),
        "fit_size": len(fit_rows),
        "dev_size": len(dev_rows),
        "validation_size": len(val_rows),
        "dev_ratio": args.dev_ratio,
        "seed": args.seed,
        "threshold_objective": "f1_recall",
        "dev_metrics": dev_result,
        "validation_metrics": validation_result,
        "validation_data": args.val,
    }

    model.save(args.model)
    save_json(report, args.metrics)
    print(report)
    print(f"saved model to {Path(args.model).resolve()}")


if __name__ == "__main__":
    main()