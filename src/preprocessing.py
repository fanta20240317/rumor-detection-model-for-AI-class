import html
import re

#text preprocessing

def normalize_text(text: str) -> str:
    return html.unescape(text or "").strip()


def sanitize_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
