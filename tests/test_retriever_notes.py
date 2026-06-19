import unittest

from src.retriever import RetrievedEvidence


class RetrieverContractTests(unittest.TestCase):
    def test_retrieved_evidence_stores_similarity(self):
        item = RetrievedEvidence(item_id="1", text="claim", label=1, event="0", similarity=0.8)
        self.assertEqual(item.label, 1)
        self.assertGreater(item.similarity, 0.0)


if __name__ == "__main__":
 unittest.main()
class EvidenceFeatureContractTests(unittest.TestCase):
    def test_evidence_module_imports(self):
        import src.evidence as evidence
        self.assertTrue(hasattr(evidence, "extract_evidence_features"))
class ClaimStructureContractTests(unittest.TestCase):
    def test_claim_structure_module_imports(self):
        import src.claim_structure as claim_structure
        self.assertTrue(hasattr(claim_structure, "extract_claim_structure_features"))