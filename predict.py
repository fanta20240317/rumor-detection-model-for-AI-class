import argparse
import json

from src.evidence_pipeline import RumorDetectionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Predict one text with the final evidence-aware pipeline."
    )
    parser.add_argument("--model", default="models/ensemble.pkl")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    pipeline = RumorDetectionPipeline.load(args.model, train_path=args.train)
    result = pipeline.predict(args.text, top_k=args.top_k, include_explanation=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()