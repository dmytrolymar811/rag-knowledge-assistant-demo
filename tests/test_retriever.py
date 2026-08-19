import tempfile
import unittest
from pathlib import Path

from rag_assistant import KnowledgeBase


class KnowledgeBaseTests(unittest.TestCase):
    def make_documents(self, root: Path) -> None:
        (root / "booking.txt").write_text(
            "Customers can reschedule appointments using the confirmation email link.",
            encoding="utf-8",
        )
        (root / "support.md").write_text(
            "Support is available Monday through Friday during business hours.",
            encoding="utf-8",
        )

    def test_returns_relevant_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_documents(root)
            results = KnowledgeBase.from_directory(root).search("How do I reschedule my appointment?")
            self.assertTrue(results)
            self.assertEqual(results[0].source, "booking.txt")

    def test_unknown_question_returns_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_documents(root)
            results = KnowledgeBase.from_directory(root).search("What is the lunar gravity constant?")
            self.assertEqual(results, [])

    def test_empty_query_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_documents(root)
            knowledge_base = KnowledgeBase.from_directory(root)
            with self.assertRaises(ValueError):
                knowledge_base.search("   ")


if __name__ == "__main__":
    unittest.main()
