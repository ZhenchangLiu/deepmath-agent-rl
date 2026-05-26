import unittest

from deepmath_lite.executor import run_python


class ExecutorTests(unittest.TestCase):
    def test_runs_basic_math(self):
        result = run_python("print(2 + 3)")
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "5")

    def test_allows_math_import(self):
        result = run_python("import math\nprint(math.factorial(5))")
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "120")

    def test_allows_cmath_import(self):
        result = run_python("import cmath\nprint(abs((1 - 1j) ** 8))")
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "16.0")

    def test_allows_common_exception_classes_and_reversed(self):
        result = run_python(
            "try:\n"
            "    raise ValueError('x')\n"
            "except Exception:\n"
            "    print(''.join(reversed(['0', '4'])))"
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "40")

    def test_allows_sympy_import(self):
        result = run_python(
            "import sympy as sp\n"
            "x = sp.symbols('x')\n"
            "print(sp.solve(sp.Eq(x + 7, 19), x)[0])"
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "12")

    def test_captures_final_expression(self):
        result = run_python(
            "from sympy import symbols, Eq, solve\n"
            "y = symbols('y')\n"
            "equation = Eq(4*y - 9, 31)\n"
            "solution = solve(equation, y)\n"
            "solution"
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "[10]")

    def test_blocks_file_access(self):
        result = run_python("print(open('/etc/passwd').read())")
        self.assertFalse(result.ok)
        self.assertIn("call not allowed: open", result.error or "")

    def test_blocks_bad_import(self):
        result = run_python("import os\nprint(os.getcwd())")
        self.assertFalse(result.ok)
        self.assertIn("import not allowed: os", result.error or "")

    def test_times_out(self):
        result = run_python("while True:\n    pass", timeout_s=0.2)
        self.assertTrue(result.timed_out)


if __name__ == "__main__":
    unittest.main()
