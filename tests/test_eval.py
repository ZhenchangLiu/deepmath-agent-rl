import unittest

from deepmath_lite.eval import verify_answer


class EvalTests(unittest.TestCase):
    def test_verifies_identical_answer(self):
        self.assertTrue(verify_answer("5", "5"))

    def test_rejects_missing_answer(self):
        self.assertFalse(verify_answer(None, "5"))

    def test_verifies_identical_fraction(self):
        self.assertTrue(verify_answer("\\frac{1}{2}", "\\dfrac{1}{2}"))

    # def test_verifies_equivalent_polynomial_forms(self):
    #     # self.assertTrue(verify_answer("x^{2} - 2x - 15", "x^2-2x-15"))
    #     self.assertTrue(verify_answer("x**2 - 2*x - 15", "x^2-2x-15"))
    
    # def test_verifies_equivalent_sympy_forms(self):
        
    #     str1 = "(3, \\frac{\\pi}{2})"
    #     str2 = "\\left(3, \\frac{\\pi}{2}\\right)"

    #     from deepmath_lite.eval import normalize_for_sympy
    #     self.assertEqual(normalize_for_sympy(str1), normalize_for_sympy(str2))

    #     from deepmath_lite.eval import verify_sympy_expression
    #     self.assertTrue(verify_sympy_expression(str1, str2))

    #     # self.assertTrue(verify_answer("(3, \\frac{\\pi}{2})", "\\left(3, \\frac{\\pi}{2}\\right)"))


if __name__ == "__main__":
    unittest.main()
