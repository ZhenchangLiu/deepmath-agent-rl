# VeRL AgentLoop 路线规划

## 当前决策

本项目选择 VeRL AgentLoop 路线，而不是 Intel-style TRL vLLM server 路线。

核心原因：

- 需要显式区分模型生成 token 与环境插入的 observation token。
- 需要在 rollout 中建模多步 `code -> sandbox -> observation -> next generation`。
- 需要为后续 token-level mask、credit assignment、并发工具执行保留清晰接口。
- 不希望把 agent rollout 伪装成普通 vLLM `/generate` completion，避免语义边界混淆。

## 暂停事项

本地 `llm` 环境不继续安装或补齐 VeRL 依赖。

原因：

- 当前环境已有其他项目依赖，继续升级 `pydantic`、`fastapi`、`starlette`、`numpy` 等包有污染风险。
- VeRL/Ray/vLLM 真实训练栈更适合在 H800 上的独立 conda 环境或容器中运行。
- 本地 Mac 不作为端到端 VeRL 训练环境，只作为代码开发与轻量单元测试环境。

## 环境现状记录

在决定暂停前，本地 `llm` 环境曾尝试安装 VeRL 相关包：

```text
verl 0.8.0.dev0
ray 2.55.1
tensordict 0.10.0
hydra-core 1.3.2
pydantic 2.13.4
```

后续如果发现本地其他项目受影响，应优先考虑恢复或重建 `llm` 环境，而不是继续在该环境内修补 VeRL。

## 本地开发范围

本地只做以下工作：

1. 数据准备脚本。
2. reward 函数。
3. AgentLoop 核心逻辑的可测试实现。
4. token mask 逻辑单元测试。
5. H800 启动脚本和配置文件的静态准备。

本地不要求：

- 导入完整 VeRL 训练栈。
- 启动 Ray。
- 启动 vLLM rollout server。
- 跑 GRPO trainer。

## 目标架构

目标 rollout 语义：

```text
problem
-> LLM generates reasoning and <python>...</python>
-> sandbox executes code
-> environment inserts <observation>...</observation>
-> LLM continues generation
-> final answer appears in \boxed{...}
-> reward is computed from final answer
```

关键边界：

```text
model generated tokens: response_mask = 1
environment observation tokens: response_mask = 0
prompt/problem tokens: not part of response loss
```

## VeRL 接口目标

后续自定义 AgentLoop 应以 VeRL 标准 AgentLoop 输出为目标：

```text
AgentLoopOutput(
  prompt_ids=...,
  response_ids=...,
  response_mask=...,
  num_turns=...,
  metrics=...
)
```

其中：

- `response_ids` 包含模型输出和 observation 上下文。
- `response_mask` 与 `response_ids` 等长。
- 模型实际生成的 token mask 为 `1`。
- sandbox observation token mask 为 `0`。

本地测试可以先用 fake tokenizer / fake model runner 验证该契约。

## 已完成

- `scripts/prepare_deepmath_verl.py`
  - 将 DeepMath-103K 转为 VeRL parquet。
  - 默认跳过缺失答案样本。
  - 默认不保留 `r1_solution_*`，避免训练 parquet 过重。

- `deepmath_lite/verl_reward.py`
  - 提供 VeRL custom reward 入口。
  - 抽取 `\boxed{...}`。
  - 复用现有 `verify_answer`，不修改评测口径。

## 下一步实现顺序

1. 新增 agent trajectory 数据结构。
   - 表示 assistant span、observation span、final answer。
   - 不依赖 VeRL。

2. 新增异步 sandbox 执行接口。
   - 先复用现有 `run_python`。
   - 后续扩展为进程池或 Ray actor pool。

3. 新增本地可测的 AgentLoop core。
   - 输入 question。
   - 调用抽象 model runner。
   - 解析 `<python>...</python>`。
   - 执行 sandbox。
   - 插入 observation。
   - 停在 `\boxed{...}` 或 `max_steps`。

4. 新增 mask 构造测试。
   - assistant text token mask = 1。
   - observation text token mask = 0。
   - 多轮拼接顺序正确。

5. 新增 VeRL adapter。
   - 在 H800 环境中继承真实 `AgentLoopBase`。
   - 将本地 core 输出转换为 `AgentLoopOutput`。

6. 新增 H800 smoke 配置。
   - 只跑少量样本。
   - 只验证 rollout、reward、GRPO step、checkpoint。

## H800 环境建议

不要复用本机 `llm` 环境。建议新建独立环境：

```bash
conda create -n verl-deepmath python=3.12
conda activate verl-deepmath
```

安装策略：

```bash
git clone https://github.com/verl-project/verl.git
cd verl
pip install --no-deps -e .
```

然后按 H800 的 CUDA、torch、vLLM 版本补齐依赖。具体版本应以 VeRL 官方文档和 H800 机器已有驱动为准。

## 风险与检查点

主要风险：

- VeRL AgentLoop API 可能随版本变化。
- vLLM rollout 的 token ids 与本地 tokenizer 重建文本之间可能出现对齐问题。
- observation mask 必须在 tokenization 后严格校验长度。
- sandbox 并发过高可能拖慢 rollout 或造成进程泄漏。

必须通过的检查：

```text
len(response_ids) == len(response_mask)
observation span mask 全为 0
assistant span mask 全为 1
最终 reward 只依赖 final answer
MATH-500 仍只作为评测集
```

## 当前原则

- 不再动评测口径。
- 不再处理旧 trace 的协议失败。
- MATH-500 只用于评测。
- DeepMath-103K / OpenMathReasoning-TIR 等训练集用于 RL。
- 本地不追求跑通 VeRL。
- H800 上用独立环境做真实 VeRL 集成。
