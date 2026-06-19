def apply_threshold(probabilities, threshold):
    return [1 if p >= threshold else 0 for p in probabilities]


def threshold_candidates():
    return [0.45, 0.48, 0.50, 0.52, 0.55]