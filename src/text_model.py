import csv
import json
import math
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.defense import sanitize_text


TOKEN_RE = re.compile(r"[a-z0-9_#@']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}
WEAK_EVIDENCE_WORDS = STOPWORDS | {
    "about",
    "after",
    "again",
    "all",
    "also",
    "any",
    "been",
    "being",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "had",
    "her",
    "him",
    "his",
    "how",
    "into",
    "just",
    "more",
    "now",
    "only",
    "our",
    "out",
    "over",
    "she",
    "so",
    "than",
    "their",
    "them",
    "then",
    "there",
    "they",
    "too",
    "what",
    "when",
    "where",
    "who",
    "why",
    "you",
    "your",
}
BROAD_EVENT_HASHTAGS = {
    "#ferguson",
    "#mikebrown",
    "#ottawashooting",
    "#sydneysiege",
    "#charliehebdo",
    "#germanwings",
    "#prince",
}


def normalize_text(text):
    text = sanitize_text(text)
    text = text.lower()
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" USER ", text)
    return text


def tokenize(text):
    return TOKEN_RE.findall(normalize_text(text))


def feature_terms(text, use_unigram=True, use_bigram=True, use_char=True, use_meta=True):
    tokens = tokenize(text)
    terms = []

    if use_unigram:
        terms.extend("w=" + tok for tok in tokens)
    if use_bigram:
        terms.extend(
            "wb=" + tokens[i] + "_" + tokens[i + 1]
            for i in range(len(tokens) - 1)
        )

    if use_char:
        compact = " ".join(tokens)
        for n in (3, 4, 5):
            if len(compact) >= n:
                terms.extend(
                    "c=" + compact[i : i + n]
                    for i in range(len(compact) - n + 1)
                )

    if use_meta:
        raw_tokens = TOKEN_RE.findall(text)
        lower_text = text.lower()
        url_count = len(URL_RE.findall(text))
        mention_count = len(MENTION_RE.findall(text))
        hashtag_count = len(re.findall(r"#\w+", text))
        question_count = text.count("?")
        exclaim_count = text.count("!")
        quote_count = text.count('"') + text.count("'")
        upper_count = sum(1 for tok in raw_tokens if len(tok) >= 3 and tok.isupper())

        if "URL" in text or "http" in lower_text:
            terms.append("meta=has_url")
        if "@" in text:
            terms.append("meta=has_mention")
        if "#" in text:
            terms.append("meta=has_hashtag")
        if "?" in text:
            terms.append("meta=has_question")
        if "!" in text:
            terms.append("meta=has_exclaim")
        if re.search(r"\brt\b", lower_text):
            terms.append("meta=has_rt")
        if "breaking" in lower_text:
            terms.append("meta=has_breaking")
        if quote_count >= 2:
            terms.append("meta=has_quote")
        if any(ch.isdigit() for ch in text):
            terms.append("meta=has_digit")
        if upper_count >= 1:
            terms.append("meta=has_upper_word")
        if upper_count >= 2:
            terms.append("meta=has_many_upper_words")

        terms.append(f"meta=len_bin={bucket(len(tokens), [6, 12, 20, 35])}")
        terms.append(f"meta=url_count={bucket(url_count, [1, 2])}")
        terms.append(f"meta=mention_count={bucket(mention_count, [1, 2, 4])}")
        terms.append(f"meta=hashtag_count={bucket(hashtag_count, [1, 2, 4])}")
        terms.append(f"meta=question_count={bucket(question_count, [1, 2])}")
        terms.append(f"meta=exclaim_count={bucket(exclaim_count, [1, 2])}")

    return terms


def bucket(value, cutoffs):
    for cutoff in cutoffs:
        if value <= cutoff:
            return f"le{cutoff}"
    return f"gt{cutoffs[-1]}"


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is not None:
            reader.fieldnames = [
                name.strip().lstrip("\ufeff").strip('"')
                for name in reader.fieldnames
            ]
        for row in reader:
            rows.append(
                {
                    "id": row["id"].strip(),
                    "text": row["text"],
                    "label": int(str(row["label"]).strip()),
                    "event": int(str(row["event"]).strip()),
                }
            )
    return rows


def stratified_train_dev_split(rows, dev_ratio=0.2, seed=2026):
    """Split rows reproducibly while preserving each event/label group."""
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["event"], row["label"])].append(row)

    fit_rows = []
    dev_rows = []
    rng = random.Random(seed)
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        if len(group) < 2:
            dev_count = 0
        else:
            dev_count = max(1, int(round(len(group) * dev_ratio)))
            dev_count = min(dev_count, len(group) - 1)
        dev_rows.extend(group[:dev_count])
        fit_rows.extend(group[dev_count:])

    rng.shuffle(fit_rows)
    rng.shuffle(dev_rows)
    return fit_rows, dev_rows


class TfidfLinearRumorModel:
    def __init__(
        self,
        max_features=45000,
        min_df=2,
        seed=2026,
        use_unigram=True,
        use_bigram=True,
        use_char=True,
        use_meta=True,
    ):
        self.max_features = max_features
        self.min_df = min_df
        self.seed = seed
        self.use_unigram = use_unigram
        self.use_bigram = use_bigram
        self.use_char = use_char
        self.use_meta = use_meta
        self.threshold = 0.5
        self.vocab = {}
        self.idf = None
        self.weights = None
        self.bias = 0.0

    def build_vocab(self, texts):
        df = Counter()
        for text in texts:
            df.update(set(self.extract_terms(text)))

        candidates = [
            (term, count)
            for term, count in df.items()
            if count >= self.min_df
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        selected = candidates[: self.max_features]

        self.vocab = {term: i for i, (term, _) in enumerate(selected)}
        n_docs = len(texts)
        self.idf = np.ones(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

    def vectorize_one(self, text):
        counts = defaultdict(float)
        for term in self.extract_terms(text):
            idx = self.vocab.get(term)
            if idx is not None:
                counts[idx] += 1.0

        if not counts:
            return {}

        norm = 0.0
        vec = {}
        for idx, count in counts.items():
            value = math.log1p(count) * float(self.idf[idx])
            vec[idx] = value
            norm += value * value
        norm = math.sqrt(norm) or 1.0
        return {idx: value / norm for idx, value in vec.items()}

    def vectorize(self, texts):
        return [self.vectorize_one(text) for text in texts]

    def extract_terms(self, text):
        return feature_terms(
            text,
            use_unigram=getattr(self, "use_unigram", True),
            use_bigram=getattr(self, "use_bigram", True),
            use_char=getattr(self, "use_char", True),
            use_meta=getattr(self, "use_meta", True),
        )

    def fit(self, texts, labels, epochs=55, lr=0.22, l2=1e-5):
        if not self.vocab:
            self.build_vocab(texts)

        x = self.vectorize(texts)
        y = np.asarray(labels, dtype=np.float32)
        self.weights = np.zeros(len(self.vocab), dtype=np.float32)
        self.bias = 0.0

        pos = float(y.sum())
        neg = float(len(y) - pos)
        class_weight = {
            0: len(y) / (2.0 * max(neg, 1.0)),
            1: len(y) / (2.0 * max(pos, 1.0)),
        }

        rng = random.Random(self.seed)
        indices = list(range(len(y)))
        for epoch in range(epochs):
            rng.shuffle(indices)
            step_lr = lr / math.sqrt(1.0 + epoch * 0.15)
            for i in indices:
                score = self.bias + sum(
                    self.weights[idx] * value
                    for idx, value in x[i].items()
                )
                prob = 1.0 / (1.0 + math.exp(-max(min(score, 35.0), -35.0)))
                grad = (prob - y[i]) * class_weight[int(y[i])]
                for idx, value in x[i].items():
                    self.weights[idx] -= step_lr * (
                        grad * value + l2 * self.weights[idx]
                    )
                self.bias -= step_lr * grad

        return self

    def predict_proba_one(self, text):
        vec = self.vectorize_one(text)
        score = self.bias + sum(
            self.weights[idx] * value
            for idx, value in vec.items()
        )
        prob = 1.0 / (1.0 + math.exp(-max(min(score, 35.0), -35.0)))
        return prob

    def predict_one(self, text, threshold=None):
        if threshold is None:
            threshold = self.threshold
        prob = self.predict_proba_one(text)
        return int(prob >= threshold), prob

    def explain_one(self, text, top_k=6):
        vec = self.vectorize_one(text)
        contributions = []
        inverse_vocab = {idx: term for term, idx in self.vocab.items()}
        for idx, value in vec.items():
            contrib = float(self.weights[idx] * value)
            if abs(contrib) > 0:
                contributions.append((contrib, inverse_vocab[idx]))

        label, prob = self.predict_one(text)
        if label == 1:
            ranked = sorted(contributions, reverse=True)[: max(40, top_k)]
            direction = "rumor"
            confidence = prob
        else:
            ranked = sorted(contributions)[: max(40, top_k)]
            direction = "non-rumor"
            confidence = 1.0 - prob

        readable_terms = [
            clean_term(term)
            for _, term in ranked
            if is_readable_evidence(term)
        ]
        readable_terms = dedupe_terms([term for term in readable_terms if term])
        if len(readable_terms) < top_k:
            readable_terms = dedupe_terms(readable_terms + salient_terms(text))
        if readable_terms:
            evidence = ", ".join(readable_terms[:top_k])
            reason = (
                f"Base model predicts {direction} with confidence {confidence:.2f}. "
                "The strongest text evidence terms are: "
                f"{evidence}."
            )
        else:
            reason = (
                f"Base model predicts {direction} with confidence {confidence:.2f}. "
                "No strong readable term-level evidence was found."
            )
        return {
            "label": label,
            "label_name": direction,
            "prob_rumor": prob,
            "confidence": confidence,
            "evidence_terms": readable_terms[:top_k],
            "explanation": reason,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def clean_term(term):
    if term.startswith("w="):
        return term[2:]
    if term.startswith("wb="):
        words = [
            word
            for word in term[3:].split("_")
            if is_informative_word(word)
        ]
        return " ".join(words)
    if term.startswith("c="):
        return term[2:].strip()
    if term.startswith("meta="):
        known = {
            "meta=has_url": "contains URL",
            "meta=has_mention": "contains user mention",
            "meta=has_hashtag": "contains hashtag",
            "meta=has_question": "contains question mark",
            "meta=has_exclaim": "contains exclamation mark",
            "meta=has_rt": "contains retweet marker",
            "meta=has_breaking": "contains breaking-news wording",
            "meta=has_quote": "contains quotation marks",
            "meta=has_digit": "contains digits",
            "meta=has_upper_word": "contains uppercase word",
            "meta=has_many_upper_words": "contains multiple uppercase words",
        }
        if term in known:
            return known[term]
        if term.startswith("meta=len_bin="):
            return "text length feature"
        if term.startswith("meta=url_count="):
            return "URL count feature"
        if term.startswith("meta=mention_count="):
            return "mention count feature"
        if term.startswith("meta=hashtag_count="):
            return "hashtag count feature"
        if term.startswith("meta=question_count="):
            return "question mark count feature"
        if term.startswith("meta=exclaim_count="):
            return "exclamation mark count feature"
        return term
    return term


def is_readable_evidence(term):
    if term.startswith("meta="):
        return True
    if term.startswith("wb="):
        words = term[3:].split("_")
        return any(is_informative_word(word) for word in words)
    if term.startswith("w="):
        return is_informative_word(term[2:])
    return False


def is_informative_word(word):
    word = word.strip("'").lower()
    if len(word) <= 2:
        return False
    if word in WEAK_EVIDENCE_WORDS:
        return False
    if word in {"url", "user"}:
        return False
    if word in BROAD_EVENT_HASHTAGS:
        return False
    return any(ch.isalpha() for ch in word) or word.startswith("#")


def dedupe_terms(terms):
    seen = set()
    deduped = []
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped


def salient_terms(text):
    terms = []
    for raw in re.findall(r"#\w+|[A-Za-z][A-Za-z0-9_'-]{3,}", text):
        word = raw.strip()
        normalized = word.lower().strip("'")
        if normalized in WEAK_EVIDENCE_WORDS:
            continue
        if normalized in {"http", "https", "user", "url"}:
            continue
        if normalized in BROAD_EVENT_HASHTAGS:
            continue
        terms.append(word)
    return terms


def metrics(labels, probs, threshold=0.5):
    preds = [int(p >= threshold) for p in probs]
    labels = list(labels)
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    tn = sum(int(p == 0 and y == 0) for p, y in zip(preds, labels))
    tp = sum(int(p == 1 and y == 1) for p, y in zip(preds, labels))
    fp = sum(int(p == 1 and y == 0) for p, y in zip(preds, labels))
    fn = sum(int(p == 0 and y == 1) for p, y in zip(preds, labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": correct / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def threshold_key(result, objective):
    if objective == "accuracy":
        return (
            result["accuracy"],
            result["f1"],
            result["recall"],
            -abs(result["threshold"] - 0.5),
        )
    if objective == "f1_recall":
        f1_bucket = round(result["f1"] / 0.002)
        return (
            f1_bucket,
            result["recall"],
            result["precision"],
            result["accuracy"],
            -result["threshold"],
        )
    raise ValueError(f"unknown threshold objective: {objective}")


def find_best_threshold(
    labels,
    probs,
    start=0.30,
    end=0.70,
    step=0.01,
    objective="accuracy",
):
    best = None
    n_steps = int(round((end - start) / step)) + 1
    for i in range(n_steps):
        threshold = round(start + i * step, 4)
        result = metrics(labels, probs, threshold)
        if best is None:
            best = result
        elif threshold_key(result, objective) > threshold_key(best, objective):
            best = result
    return best


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
