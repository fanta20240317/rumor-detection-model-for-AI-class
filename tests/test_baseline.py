import unittest

from src.baseline import KeywordBaseline
from src.text_model import Record

# baseline detection tests

class BaselineTests(unittest.TestCase):
    def test_baseline_probability_is_in_rumor_direction(self):
        model = KeywordBaseline().fit([Record("1", "fake claim", 1), Record("2", "official update", 0)])
        self.assertGreater(model.predict_proba("fake claim"), model.predict_proba("official update"))


if __name__ == "__main__":
    unittest.main()