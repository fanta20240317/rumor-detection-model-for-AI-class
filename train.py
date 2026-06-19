import argparse

from src.baseline import KeywordBaseline
from src.text_model import read_csv

# the simplest edition

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--val", default="val.csv")
    parser.add_argument("--model", default="models/ensemble.pkl")
    parser.add_argument("--metrics", default="outputs/metrics.json")
    args = parser.parse_args()
    records = read_csv(args.train)
    model = KeywordBaseline().fit(records)
    print(f"trained baseline on {len(records)} records")


if __name__ == "__main__":
    main()