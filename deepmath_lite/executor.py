"""A small restricted Python executor for model-generated calculations."""

from __future__ import annotations

import ast
import importlib
import multiprocessing as mp
import os
import queue
import threading
import textwrap
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from types import MappingProxyType
from typing import Any


ALLOWED_IMPORTS = frozenset({"math", "cmath", "fractions", "decimal", "statistics", "itertools", "sympy"})
BLOCKED_CALL_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input"})
MAX_CODE_PROCESSES = int(os.environ.get("DEEPMATH_LITE_MAX_CODE_PROCESSES", "16"))
PROCESS_SEMAPHORE = threading.BoundedSemaphore(MAX_CODE_PROCESSES)


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    error: str | None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out


class SafetyError(ValueError):
    """Raised when code uses a construct outside the allowed subset."""


class SafetyVisitor(ast.NodeVisitor):
    """Reject filesystem, process, and dynamic-code features before execution."""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in ALLOWED_IMPORTS:
                raise SafetyError(f"import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if root not in ALLOWED_IMPORTS:
            raise SafetyError(f"import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALL_NAMES:
            raise SafetyError(f"call not allowed: {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            raise SafetyError("dunder attribute access is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise SafetyError("dunder name access is not allowed")
        self.generic_visit(node)


# 静态检查（语法树）
def validate_code(code: str) -> ast.Module:
    """Parse and statically validate a candidate code snippet."""

    normalized = textwrap.dedent(code).strip()
    if not normalized:
        raise SafetyError("empty code block")
    tree = ast.parse(normalized, mode="exec")
    SafetyVisitor().visit(tree)
    return tree


# 动态检查（安全字典）
def _safe_builtins() -> MappingProxyType[str, Any]:
    def safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        root = name.split(".", 1)[0]
        if level != 0 or root not in ALLOWED_IMPORTS:
            raise ImportError(f"import not allowed: {name}")
        return importlib.import_module(name)

    allowed = {
        "abs": abs,
        "all": all,
        "any": any,
        "ArithmeticError": ArithmeticError,
        "bool": bool,
        "complex": complex,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "Exception": Exception,
        "filter": filter,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "print": print,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "ValueError": ValueError,
        "zip": zip,
        "ZeroDivisionError": ZeroDivisionError,
        "__import__": safe_import,
    }
    return MappingProxyType(allowed)


def _worker(code: str, result_queue: mp.Queue) -> None:
    stream = StringIO()
    try:
        tree = validate_code(code)
        globals_dict = {"__builtins__": _safe_builtins()}
        with redirect_stdout(stream):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                prefix = ast.Module(body=tree.body[:-1], type_ignores=tree.type_ignores)
                expr = ast.Expression(body=tree.body[-1].value)
                ast.fix_missing_locations(prefix)
                ast.fix_missing_locations(expr)
                exec(compile(prefix, filename="<model-python>", mode="exec"), globals_dict, globals_dict)
                value = eval(compile(expr, filename="<model-python>", mode="eval"), globals_dict, globals_dict)
                if value is not None:
                    print(repr(value))
            else:
                compiled = compile(tree, filename="<model-python>", mode="exec")
                exec(compiled, globals_dict, globals_dict)
        result_queue.put(ExecutionResult(stdout=stream.getvalue(), error=None))
    except Exception as exc:  # noqa: BLE001 - return errors to the agent trace.
        result_queue.put(ExecutionResult(stdout=stream.getvalue(), error=f"{type(exc).__name__}: {exc}"))


def run_python(code: str, timeout_s: float = 2.0) -> ExecutionResult:
    """Run validated Python in a subprocess with a hard timeout."""

    result_queue: mp.Queue | None = None
    process: mp.Process | None = None
    with PROCESS_SEMAPHORE:
        try:
            result_queue = mp.Queue(maxsize=1)
            process = mp.Process(target=_worker, args=(code, result_queue))
            process.start()
            process.join(timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(0.5)
                return ExecutionResult(stdout="", error=f"timeout after {timeout_s:.1f}s", timed_out=True)

            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return ExecutionResult(stdout="", error="executor process exited without result")
        except OSError as exc:
            return ExecutionResult(stdout="", error=f"OSError: {exc}")
        finally:
            if process is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(0.5)
                try:
                    process.close()
                except ValueError:
                    pass
            if result_queue is not None:
                result_queue.close()
                result_queue.join_thread()
