import unittest

from src.ensemble_model import EvidenceAwareEnsemble


class FusionContractTests(unittest.TestCase):
    def test_label_threshold_direction(self):
        model = EvidenceAwareEnsemble(models=[], weights=[], threshold=0.5)
        self.assertEqual(model.predict_label_from_probability(0.9), 1)
        self.assertEqual(model.predict_label_from_probability(0.1), 0)


if __name__ == "__main__":
    unittest.main()
class PipelineContractTests(unittest.TestCase):
    def test_pipeline_module_imports(self):
        import src.evidence_pipeline as evidence_pipeline
        self.assertTrue(hasattr(evidence_pipeline, "RumorDetectionPipeline"))
class ExplanationContractTests(unittest.TestCase):
    def test_explanation_contract_documented(self):
        self.assertIn("evidence", "evidence-first explanation")