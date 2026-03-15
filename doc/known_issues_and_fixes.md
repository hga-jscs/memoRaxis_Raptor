# 代码问题排查与修复清单

本文档记录了本次对仓库进行静态检查与最小可复现运行检查后，**可以确定存在问题**的点，以及对应修复方案。

## 1. 适配器与记忆接口签名不一致（会直接抛异常）

### 问题现象
- `SingleTurnAdaptor` / `IterativeAdaptor` / `PlanAndActAdaptor` 在调用 `memory.retrieve(...)` 时传入了 `max_tokens`、`stage` 参数。
- 但 `BaseMemorySystem.retrieve` 和 `MockMemory.retrieve` 原签名只有 `(query, top_k)`。
- 这会导致运行时抛出：`TypeError: ... got an unexpected keyword argument 'max_tokens'`。

### 最小复现
```bash
python - <<'PY'
from src.adaptors import SingleTurnAdaptor
from src.llm_interface import MockLLMClient
from src.memory_interface import MockMemory

adaptor = SingleTurnAdaptor(MockLLMClient(), MockMemory())
adaptor.run('test')
PY
```

### 修复
- 统一 `BaseMemorySystem.retrieve` 签名：增加 `max_tokens`、`stage` 可选参数。
- 更新 `MockMemory.retrieve` 与 `SimpleRAGMemory.retrieve` 签名保持一致。
- 增加参数说明与调试日志，提升可观测性。

---

## 2. `main.py` 调用 `PlanAndActAdaptor` 参数名错误（会直接抛异常）

### 问题现象
- `main.py` 使用 `PlanAndActAdaptor(..., max_replan=2)`。
- 但 `PlanAndActAdaptor.__init__` 并不存在 `max_replan`，实际参数为 `max_additions` 等。
- 这会导致启动时直接抛 `TypeError`。

### 最小复现
```bash
python - <<'PY'
from src.adaptors import PlanAndActAdaptor
from src.llm_interface import MockLLMClient
from src.memory_interface import MockMemory

PlanAndActAdaptor(MockLLMClient(), MockMemory(), max_replan=2)
PY
```

### 修复
- 将 `main.py` 中参数改为 `max_additions=2`。

---

## 3. `run_all_tasks.py` 的 `--adaptors all` 形同虚设（逻辑缺陷）

### 问题现象
- 参数定义允许 `--adaptors all`。
- 但代码没有将 `all` 展开为 `R1 R2 R3`，会把字符串 `all` 原样传给下游脚本，导致下游不识别或行为异常。

### 修复
- 在参数解析后，若检测到 `all`，自动展开为 `['R1', 'R2', 'R3']`。

---

## 4. `Evidence.metadata` 使用可变默认值 `{}`（隐患）

### 问题现象
- `Evidence` 模型字段原定义为 `metadata: Dict[str, Any] = {}`。
- 可变默认值会带来实例间状态污染风险（在 dataclass/pydantic 使用中均属于常见陷阱）。

### 修复
- 改为 `Field(default_factory=dict)`。

---

## 附：本次额外改进（可视化调试输出 + 注释增强）

- `run_all_tasks.py` 中 `run_cmd` 现输出命令耗时与返回状态，便于肉眼定位慢步骤和失败步骤。
- `main.py` 与关键函数增加更完整中文注释与流程说明，提升可维护性。
