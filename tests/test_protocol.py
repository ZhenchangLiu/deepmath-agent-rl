import unittest

from deepmath_lite.protocol import (
    extract_boxed_answer,
    find_first_code_block,
    has_malformed_boxed_answer,
    has_malformed_code_block,
    has_markdown_code_block,
    make_observation,
)


class ProtocolTests(unittest.TestCase):
    def test_finds_code_block(self):
        block = find_first_code_block("hi\n<python>\nprint(2 + 3)\n</python>")
        self.assertIsNotNone(block)
        self.assertEqual(block.code, "print(2 + 3)")

    def test_extracts_last_boxed_answer(self):
        self.assertEqual(extract_boxed_answer("a \\boxed{1} b \\boxed{5}"), "5")

    def test_extracts_nested_boxed_answer(self):
        self.assertEqual(
            extract_boxed_answer("\\[ \\boxed{\\frac{3\\sqrt{3}}{4}} \\]"),
            "\\frac{3\\sqrt{3}}{4}",
        )

    def test_detects_malformed_boxed_answer(self):
        self.assertTrue(has_malformed_boxed_answer("\\boxed{\\frac{1}{2}"))

    def test_detects_malformed_code_block(self):
        self.assertTrue(has_malformed_code_block("<python>\nprint(2 + 3)"))

    def test_detects_markdown_code_block(self):
        self.assertTrue(has_markdown_code_block("```python\nprint(2 + 3)\n```"))

    def test_formats_observation(self):
        self.assertEqual(make_observation("5\n"), "<observation>\n5\n</observation>")


if __name__ == "__main__":
    unittest.main()
