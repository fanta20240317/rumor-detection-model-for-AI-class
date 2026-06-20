import re
import unicodedata


ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # zero width no-break space / BOM
    "\u00ad",  # soft hyphen
    "\u180e",  # mongolian vowel separator
}
ZERO_WIDTH_RE = re.compile("[" + "".join(ZERO_WIDTH_CHARS) + "]")

# Confusable letters that NFKC does NOT fold: Cyrillic / Greek look-alikes that
# attackers swap in for visually identical Latin letters.
HOMOGLYPH_MAP = {
    # Cyrillic lowercase -> Latin
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c",
    "\u0443": "y", "\u0445": "x", "\u0456": "i", "\u0458": "j", "\u04bb": "h",
    "\u0455": "s", "\u0501": "d", "\u051b": "q", "\u0261": "g", "\u043c": "m",
    "\u043d": "h", "\u0442": "t", "\u0432": "b", "\u043a": "k",
    # Cyrillic uppercase -> Latin
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M",
    "\u041d": "H", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0422": "T",
    "\u0425": "X", "\u0406": "I", "\u0408": "J",
    # Greek -> Latin
    "\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "\u03c1": "p", "\u03c5": "u",
    "\u03b9": "i", "\u03ba": "k", "\u03bd": "v", "\u03c4": "t", "\u03c7": "x",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0396": "Z", "\u0397": "H",
    "\u0399": "I", "\u039a": "K", "\u039c": "M", "\u039d": "N", "\u039f": "O",
    "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X",
    # Common math/full-width style letters fold via NFKC, but a few stray ones:
    "\u0131": "i",  # dotless i
}
HOMOGLYPH_TABLE = {ord(k): v for k, v in HOMOGLYPH_MAP.items()}

REPEAT_RE = re.compile(r"(.)\1{2,}")
MULTISPACE_RE = re.compile(r"\s+")


def remove_zero_width(text):
    return ZERO_WIDTH_RE.sub("", text)


def map_homoglyphs(text):
    return text.translate(HOMOGLYPH_TABLE)


def collapse_repeats(text):
    """Collapse runs of 3+ identical characters down to 2 (coooool -> cool)."""
    return REPEAT_RE.sub(r"\1\1", text)


def sanitize_text(text):
    """Canonicalize text to blunt character-level perturbations.

    Order matters: NFKC first folds compatibility/full-width forms, then we
    strip invisible characters, map residual confusable letters, and collapse
    stretched repeats. Digits and casing are preserved so downstream meta
    features and tokenization behave as before on clean input.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFKC", str(text))
    text = remove_zero_width(text)
    text = map_homoglyphs(text)
    text = collapse_repeats(text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def count_suspicious_chars(text):
    """Count suspicious obfuscation characters in raw input.

    Operates on the original pre-sanitize text so the obfuscation signal is
    still present for diagnostics.
    """
    if not text:
        return {"zero_width": 0, "homoglyph": 0, "stretch": 0}
    zero_width = sum(1 for ch in text if ch in ZERO_WIDTH_CHARS)
    homoglyph = sum(1 for ch in text if ord(ch) in HOMOGLYPH_TABLE)
    stretch = len(REPEAT_RE.findall(text))
    return {"zero_width": zero_width, "homoglyph": homoglyph, "stretch": stretch}