import random

#cut the data into training and validation sets

def fixed_validation_split(items, validation_fraction=0.2, seed=42):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * validation_fraction))
    return shuffled[cut:], shuffled[:cut]
