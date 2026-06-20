import argparse
import json

from src.prediction_service import RumorPredictionService


def main():
    parser = argparse.ArgumentParser(
        description="Predict one text with the final evidence-aware pipeline."
    )
    parser.add_argument("--model", default="models/main_fusion.pkl")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--no-llm",
        dest="llm",
        action="store_false",
        help="disable school-LLM evidence generation",
    )
    parser.set_defaults(llm=True)
    args = parser.parse_args()

    service = RumorPredictionService(args.model, args.train)
    result = service.predict(args.text, top_k=args.top_k, use_llm=args.llm)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
