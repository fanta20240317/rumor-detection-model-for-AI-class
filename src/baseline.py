from collections import Counter

from .preprocessing import sanitize_text

# define whether a text is a rumor based on words
class KeywordBaseline:
    def __init__(self):
        self.rumor_words = Counter()
        self.non_rumor_words = Counter()

    def fit(self, records):
        for record in records:
            words = sanitize_text(record.text).lower().split()
            if record.label == 1:
                self.rumor_words.update(words)
            else:
                self.non_rumor_words.update(words)
        return self

    def predict_proba(self, text):
        words = sanitize_text(text).lower().split()
        score = sum(self.rumor_words[w] - self.non_rumor_words[w] for w in words)
        return 1.0 / (1.0 + pow(2.718281828, -score / 20.0))
