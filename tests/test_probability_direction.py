import unittest


from src.ensemble_model import EvidenceAwareEnsemble




class ProbabilityDirectionTests(unittest.TestCase):
    def test_default_threshold_label_direction(self):
        model = EvidenceAwareEnsemble(models=[], weights=[], threshold=0.5)
        self.assertEqual(model.predict_label_from_probability(0.7), 1)
        self.assertEqual(model.predict_label_from_probability(0.3), 0)




if __name__ == "__main__":
    unittest.main()