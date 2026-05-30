#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== Python =="
python - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY

echo "== Core imports =="
python - <<'PY'
from deepmath_lite.verl_agent_loop import build_output_payload, extract_question
from deepmath_lite.verl_agent_loop_core import AgentLoopCore
from deepmath_lite.verl_reward import compute_score

print("deepmath_lite imports ok")
print("reward", compute_score("smoke", "answer \\boxed{2}", "2"))
print("question", extract_question({"prompt": [{"role": "user", "content": "1+1?"}]}))
print("core", AgentLoopCore)
print("payload_builder", build_output_payload)
PY

echo "== VeRL AgentLoop API import =="
python - <<'PY'
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput

print("AgentLoopBase", AgentLoopBase)
print("AgentLoopMetrics fields", AgentLoopMetrics.model_fields.keys())
print("AgentLoopOutput fields", AgentLoopOutput.model_fields.keys())
PY

echo "== Project tests =="
python -m unittest discover -s tests

echo "== Tiny data preparation =="
TMP_DATA_DIR="${TMPDIR:-/tmp}/deepmath_verl_smoke_data"
python scripts/prepare_deepmath_verl.py \
  --limit 32 \
  --val-size 4 \
  --output-dir "$TMP_DATA_DIR"

echo "== Local payload smoke =="
python - <<'PY'
from deepmath_lite.verl_agent_loop import build_output_payload
from deepmath_lite.verl_agent_loop_core import AgentLoopCore


class ScriptedModel:
    def __init__(self):
        self.outputs = [
            "<python>\nprint(2 + 3)\n</python>",
            "\\boxed{5}",
        ]

    def generate(self, prompt: str) -> str:
        return self.outputs.pop(0)


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


question = "What is 2+3?"
rollout = AgentLoopCore(model=ScriptedModel()).run(question)
payload = build_output_payload(question, rollout, CharacterTokenizer())

assert len(payload.response_ids) == len(payload.response_mask)
assert 0 in payload.response_mask
assert 1 in payload.response_mask
assert payload.extra_fields["final_answer"] == "5"
print("payload ok", len(payload.prompt_ids), len(payload.response_ids), payload.extra_fields)
PY

echo "H800 smoke checks passed."
