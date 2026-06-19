import unittest

from src.ensemble_model import EvidenceAwareEnsemble


class FusionContractTests(unittest.TestCase):
    def test_label_threshold_direction(self):
        model = EvidenceAwareEnsemble(models=[], weights=[], threshold=0.5)
        self.assertEqual(model.predict_label_from_probability(0.9), 1)
        self.assertEqual(model.predict_label_from_probability(0.1), 0)


if __name__ == "__main__":
    unittest.main()