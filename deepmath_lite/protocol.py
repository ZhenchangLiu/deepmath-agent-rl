"""Text protocol helpers for code-tool reasoning."""

from __future__ import annotations

import re
from dataclasses import dataclass


PYTHON_BLOCK_RE = re.compile(r"<python>\s*(.*?)\s*</python>", re.DOTALL)
MARKDOWN_CODE_BLOCK_RE = re.compile(r"```(?:[^\n`]*)\n.*?(?:```|$)", re.DOTALL)


SYSTEM_PROMPT = """You are a careful math problem solver.
You must write down your natural language reasoning and thinking process before giving the final answer.

You must choose exactly one response mode each turn.

Mode A: Tool call.
Use this mode only when you need Python. Output exactly one Python block and then stop immediately.
When you need to do exact mathematical calculation, you must use Python.
<python>
# your calculation code
</python>

The system will execute the code and return:
<observation>
# system observation
</observation>

Mode B: Final answer.
Use this mode only when you know the final answer. Output the final answer in \\boxed{...} and do not include any Python block.

You can iteratively use Mode A based on the system's <observation> result until you can reach a definitive conclusion.

Strict Rules:
1. Never output <observation>; only the system may output observations.
2. Never output <python> and \\boxed{...} in the same response.
3. If you output <python>, stop immediately after </python>.
4. No ```python code blocks allowed; use only <python> </python> tags for all code.
"""


@dataclass(frozen=True)
class CodeBlock:
    """One Python tool call found in model text."""

    code: str
    start: int
    end: int


# 组装 prompt
def build_prompt(question: str, history: list[str] | None = None) -> str:
    """Build a simple single-string prompt.

    Keeping the prompt as plain text makes the first version backend-agnostic:
    a local mock model, vLLM completion endpoint, or chat model can all consume it.
    """

    parts = [SYSTEM_PROMPT.strip(), "\nProblem:\n", question.strip()]
    if history:
        parts.append("\nScratchpad:\n")
        parts.extend(history)
    return "\n".join(parts).strip() + "\n"


# 抽代码
def find_first_code_block(text: str) -> CodeBlock | None:
    """Return the first <python>...</python> block, if present."""

    match = PYTHON_BLOCK_RE.search(text)
    if not match:
        return None
    return CodeBlock(code=match.group(1).strip(), start=match.start(), end=match.end())


def has_malformed_code_block(text: str) -> bool:
    """Return whether Python tags appear without a complete tool block."""

    return ("<python>" in text or "</python>" in text) and find_first_code_block(text) is None


def has_markdown_code_block(text: str) -> bool:
    """Return whether the model used a Markdown fenced code block."""

    return MARKDOWN_CODE_BLOCK_RE.search(text) is not None


# 组装 observation
def make_observation(stdout: str, stderr: str | None = None) -> str:
    """Format executor output so it can be appended to the model context."""

    content = stdout.strip()
    if stderr:
        content = f"{content}\nERROR: {stderr.strip()}".strip()
    return f"<observation>\n{content}\n</observation>"


# 抽答案
def extract_boxed_answer(text: str) -> str | None:
    """Extract the last complete answer inside \\boxed{...}."""

    answers: list[str] = []
    search_from = 0
    marker = "\\boxed"
    while True:
        start = text.find(marker, search_from)
        if start == -1:
            break

        brace_start = start + len(marker)
        while brace_start < len(text) and text[brace_start].isspace():
            brace_start += 1

        if brace_start >= len(text) or text[brace_start] != "{":
            search_from = start + len(marker)
            continue

        depth = 0
        for pos in range(brace_start, len(text)):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(text[brace_start + 1 : pos].strip())
                    search_from = pos + 1
                    break
        else:
            break

    if not answers:
        return None
    return answers[-1]


def has_malformed_boxed_answer(text: str) -> bool:
    """Return whether a boxed marker appears without a complete answer."""

    return "\\boxed" in text and extract_boxed_answer(text) is None
