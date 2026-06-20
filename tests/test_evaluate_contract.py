import unittest
import evaluate

class EvaluateContractTests(unittest.TestCase):
    def test_case_fields_include_prediction_threshold(self):
        self.assertIn("pred_label", evaluate.CASE_FIELDS)
        self.assertIn("threshold", evaluate.CASE_FIELDS)


if __name__ == "__main__":
    unittest.main()