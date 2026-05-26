# DeepMath Lite

这是一个面向学习和复现的轻量版 DeepMath 路线项目。目标不是逐行复刻
IntelLabs/DeepMath，而是把核心系统拆成容易理解、容易测试、容易迁移到多卡
H800 机器上的小组件。

## 当前版本做什么

当前版本实现推理、评测、trace 记录和协议诊断闭环：

```text
jsonl 数据 -> prompt -> model backend -> agent loop -> python executor
          -> trace jsonl -> answer extractor -> verifier -> report
```

暂时不做 SFT / GRPO 训练。训练会在推理闭环、协议稳定性和评测口径明确后再加入。

## 目录结构

```text
deepmath_lite/
  agent.py        # 工具调用推理循环
  data.py         # jsonl 数据加载
  eval.py         # 答案抽取与比较
  executor.py     # 受限 Python 执行器
  models.py       # Mock 和 OpenAI-compatible backend
  protocol.py     # <python> / <observation> / \boxed{} 协议
scripts/
  run_agent.py    # 跑单题或小 jsonl
  run_eval.py     # 对 jsonl 数据集做评测
  inspect_trace.py
  inspect_dataset.py
tests/
  test_agent.py
  test_eval.py
  test_executor.py
  test_protocol.py
```

## 环境

按你的约定，运行 Python 前使用：

```zsh
eval "$(conda shell.zsh hook)"
conda activate llm
```

本地开发只需要 Python 标准库即可跑通 Mock 流程。

连接 H800 机器上的 vLLM OpenAI-compatible server 时，需要额外安装：

```zsh
pip install openai
```

使用 DeepSeek API 时同样使用 `openai` SDK，并把 API key 放到环境变量：

```zsh
export DEEPSEEK_API_KEY="..."
```

如果想使用更强的数学答案等价验证，建议安装：

```zsh
pip install math-verify
```

当前 `verify_answer` 会先做简单归一化字符串比较，再调用 `math_verify`。
没有安装 `math-verify` 时，只能得到较弱的字符串级比较结果。

## 数据格式

输入 jsonl 每行一个题目：

```json
{"id": "demo-1", "question": "What is 2+3?", "answer": "5"}
```

## 快速运行

跑 Mock agent：

```zsh
eval "$(conda shell.zsh hook)"
conda activate llm
python scripts/run_agent.py --question "What is 2+3?"
```

用 DeepSeek API 跑 demo 评测：

```zsh
python scripts/run_eval.py \
  --input data/demo.jsonl \
  --trace-output traces_deepseek_demo.jsonl \
  --backend chat \
  --base-url https://api.deepseek.com \
  --model deepseek-v4-flash
```

跑测试：

```zsh
eval "$(conda shell.zsh hook)"
conda activate llm
python -m unittest discover -s tests
```

## Trace 与停止原因

`scripts/run_eval.py` 会把每题的 agent trace 和评测结果写入 jsonl。核心字段：

```text
problem     # 原始题目
trace       # 每轮 prompt、模型输出、代码、执行结果和最终答案
eval        # predicted / gold / correct / stopped_reason / steps
```

当前 agent 使用严格协议。`stopped_reason` 的主要取值：

```text
boxed_answer
  正常抽到 \boxed{...} 最终答案。

protocol_violation_fabricated_observation
  模型自己输出了 <observation>，这是协议违规。

protocol_violation_code_and_answer
  同一轮同时输出合法 <python>...</python> 和合法 \boxed{...}。

protocol_violation_markdown_code_block
  模型使用了 Markdown fenced code block，而不是 <python>...</python>。

truncated_generation
  模型输出因为长度限制被截断，且没有合法 code / answer。

malformed_code_block
  出现 <python> 或 </python>，但没有形成合法闭合代码块。

malformed_boxed_answer
  出现 \boxed，但答案括号没有合法闭合。

no_code_or_answer
  没有可执行代码、没有最终答案，也不属于以上更具体类别。

max_steps
  agent 达到最大轮数。

worker_error
  单题评测 worker 抛出未处理异常。
```

注意：旧 trace 文件里的 `eval.correct` 可能来自旧版 verifier 或旧依赖环境。
比较不同 verifier 时，应基于同一份 `predicted` / `gold` 重新判定，不要直接把旧
`correct` 字段当成当前口径。

## 下一步路线

1. 写一个 `scripts/rejudge_trace.py`，用当前 `verify_answer` 重算已有 trace 的准确率。
2. 用 `inspect_trace.py` / `rejudge_trace.py` 固定分析流程，避免临时脚本散落。
3. 在小样本上优化 prompt，降低 Markdown code、fabricated observation 和截断。
4. 接 H800 上的 vLLM server，跑稳定的 MATH-500 baseline。
5. 再讨论 SFT / GRPO 训练。
