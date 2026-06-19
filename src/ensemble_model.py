class DualTfidfEnsemble:
    def __init__(self, models=None, weights=None, threshold=0.5):
        self.models = models or []
        self.weights = weights or []
        self.threshold = threshold

    def predict_probability(self, text):
        if not self.models:
            return 0.5
        return sum(w * m.predict_proba(text) for m, w in zip(self.models, self.weights)) / sum(self.weights)

    def predict_label(self, text):
        return 1 if self.predict_probability(text) >= self.threshold else 0


def default_tfidf_branch_names():
    return ["char_no_meta", "unigram_bigram"]