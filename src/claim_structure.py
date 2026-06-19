"""Lightweight claim/entity structure features for tweet-level rumor detection."""

import re

from src.text_model import MENTION_RE, URL_RE, TOKEN_RE


RUMOR_TRIGGER_WORDS = {
    "alleged",
    "apparently",
    "breaking",
    "claim",
    "claims",
    "confirmed",
    "exclusive",
    "hoax",
    "report",
    "reported",
    "reports",
    "rumor",
    "rumored",
    "rumour",
    "rumours",
    "secret",
    "unconfirmed",
}

UNCERTAINTY_WORDS = {
    "allegedly",
    "apparently",
    "maybe",
    "might",
    "possibly",
    "reportedly",
    "rumored",
    "unconfirmed",
    "unknown",
}

SOURCE_WORDS = {
    "according",
    "announced",
    "cnn",
    "confirmed",
    "official",
    "police",
    "reuters",
    "said",
    "says",
    "source",
    "sources",
    "statement",
    "update",
    "via",
}


def _matches(tokens, lexicon):
    return sorted({token for token in tokens if token in lexicon})


def extract_claim_structure_features(text):
    """Return graph-lite claim features and a conservative rumor-risk score.

    The layer is intentionally dependency-free. It captures structural signals
    that often matter for breaking-news rumors: links, mentions, hashtags,
    capitalized emphasis, numbers, uncertainty, trigger words, and citation
    words. The score is not a classifier by itself; it is a weak evidence stream
    used by the evidence-aware fusion layer.
    """
    raw = str(text or "")
    tokens = [token.lower().strip("'") for token in TOKEN_RE.findall(raw)]
    word_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    upper_tokens = [token for token in TOKEN_RE.findall(raw) if len(token) >= 3 and token.isupper()]

    url_count = len(URL_RE.findall(raw))
    mention_count = len(MENTION_RE.findall(raw))
    hashtag_count = len(re.findall(r"#\w+", raw))
    digit_count = sum(ch.isdigit() for ch in raw)
    question_count = raw.count("?")
    exclaim_count = raw.count("!")
    upper_word_ratio = len(upper_tokens) / max(len(word_tokens), 1)

    rumor_triggers = _matches(tokens, RUMOR_TRIGGER_WORDS)
    uncertainty_words = _matches(tokens, UNCERTAINTY_WORDS)
    source_words = _matches(tokens, SOURCE_WORDS)

    risk = 0.5
    risk += min(len(rumor_triggers), 4) * 0.04
    risk += min(len(uncertainty_words), 3) * 0.05
    risk += min(exclaim_count, 3) * 0.025
    risk += min(question_count, 2) * 0.02
    risk += min(upper_word_ratio, 0.4) * 0.15
    risk += min(digit_count, 5) * 0.006
    risk -= min(len(source_words), 3) * 0.025
    risk -= min(url_count, 2) * 0.015
    risk = min(max(risk, 0.05), 0.95)

    signals = []
    if rumor_triggers:
        signals.append(f"rumor_trigger_words={','.join(rumor_triggers[:4])}")
    if uncertainty_words:
        signals.append(f"uncertainty_words={','.join(uncertainty_words[:4])}")
    if source_words:
        signals.append(f"source_words={','.join(source_words[:4])}")
    if url_count:
        signals.append(f"url_count={url_count}")
    if mention_count:
        signals.append(f"mention_count={mention_count}")
    if hashtag_count:
        signals.append(f"hashtag_count={hashtag_count}")
    if upper_word_ratio >= 0.12:
        signals.append(f"upper_word_ratio={upper_word_ratio:.2f}")
    if question_count or exclaim_count:
        signals.append(f"question_exclaim={question_count}/{exclaim_count}")

    return {
        "url_count": url_count,
        "mention_count": mention_count,
        "hashtag_count": hashtag_count,
        "upper_word_ratio": round(upper_word_ratio, 4),
        "digit_count": digit_count,
        "question_count": question_count,
        "exclaim_count": exclaim_count,
        "rumor_trigger_words": rumor_triggers,
        "uncertainty_words": uncertainty_words,
        "source_words": source_words,
        "structure_score": round(risk, 4),
        "signals": signals,
    }