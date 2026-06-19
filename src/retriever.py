from pathlib import Path

from src.text_model import read_csv


def normalize_match_text(text):
    return " ".join(str(text).lower().split())


def sparse_dot(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(idx, 0.0) for idx, value in left.items())


class TfidfEvidenceRetriever:
    def __init__(self, model, rows, vectors):
        self.model = model
        self.rows = rows
        self.vectors = vectors

    @classmethod
    def from_csv(cls, model, path):
        rows = read_csv(path)
        vectors = [model.vectorize_one(row["text"]) for row in rows]
        return cls(model, rows, vectors)

    def search(self, text, label=None, top_k=3, same_event=None, exclude_exact=False):
        query = self.model.vectorize_one(text)
        normalized_query = normalize_match_text(text)
        scored = []
        for row, vector in zip(self.rows, self.vectors):
            if label is not None and row["label"] != label:
                continue
            if same_event is not None and row["event"] != same_event:
                continue
            if exclude_exact and normalize_match_text(row["text"]) == normalized_query:
                continue
            score = sparse_dot(query, vector)
            if score > 0:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "score": score,
                "id": row["id"],
                "event": row["event"],
                "label": row["label"],
                "text": row["text"],
            }
            for score, row in scored[:top_k]
        ]


def build_retriever(model_path, train_path):
    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    retriever_model = model.models[0] if hasattr(model, "models") else model
    return model, TfidfEvidenceRetriever.from_csv(retriever_model, train_path)


def ensure_model_exists(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model not found: {model_path}. Please run train.py first.")