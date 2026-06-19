import unittest


from src.retriever import RetrievedEvidence




class RetrieverContractTests(unittest.TestCase):
    def test_retrieved_evidence_stores_similarity(self):
        item = RetrievedEvidence(item_id="1", text="claim", label=1, event="0", similarity=0.8)
        self.assertEqual(item.label, 1)
        self.assertGreater(item.similarity, 0.0)




if __name__ == "__main__":
    unittest.main()