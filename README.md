# memoRaxis_Raptor
这个项目致力于将Raptor的代码整合进memoRaxis中
# memoRaxis_Raptor

> 本仓库是从 `memoRaxis` 中拆分出的 RAPTOR 单后端实验仓库，目标是在保留统一 R1 / R2 / R3 推理适配器的前提下，以 RAPTOR 作为唯一记忆后端，在 MemoryAgentBench 上完成 ingest、infer、evaluate、analyze 的完整实验流程。

### 这个仓库是做什么的

`memoRaxis_Raptor` 的定位不是通用型 RAG 产品，而是一个**研究型、可复现实验仓库**。它聚焦两件事：

1. 把 RAPTOR 从原始多后端项目中独立出来，形成单后端、单路径、可单独验证的仓库；
2. 在统一的 R1 / R2 / R3 推理范式下，观察 RAPTOR 对四类记忆任务的支持能力与问答表现。

也就是说，这个仓库不是为了“封装所有功能”，而是为了让 **RAPTOR × R1/R2/R3** 这条实验链条本身更清楚、更可跑、更容易分析。

---

### 这个仓库怎么用

当前仓库支持两种使用方式：

- **一键运行**：适合第一次跑通、统一复现、快速演示
- **CLI 分步运行**：适合调参、排错、局部重跑、做科研实验

推荐顺序是：

1. 先准备好 `config/config.yaml`
2. 准备好 `MemoryAgentBench/data/` 下的四个 parquet 数据文件
3. 先做一次 **小规模 smoke test**
4. 再跑标准流程
5. 稳定后改用分步 CLI 做实验

---

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3
```

它的含义是：

- 四个任务都只跑 `instance 0`
- `Accurate_Retrieval / Conflict_Resolution / Test_Time_Learning` 各只回答前 3 个问题
- `Long_Range_Understanding` 只跑前 1 个问题
- 对首次验证更友好，能快速判断环境、路径、接口和输出链条是否正常

---

### 标准完整入口

若需要按默认参数执行四个任务的一整轮实验，可直接使用：

```bash
python run_all_tasks.py
```

若只想跑四个任务的 `instance 0`，可使用：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0
```

若要做四个任务 `instance 0` 前 10 个问题的小型测试，可使用：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 10 --conflict_limit 10 --long_limit 10 --ttl_limit 10
```

---

### 运行前必须确认的三件事

1. **Conda 环境已激活，Python 解释器正确**  
   推荐 Python 3.11，并统一在 Anaconda Prompt 中运行。

2. **`config/config.yaml` 可用**  
   LLM / embedding 接口必须正确配置，否则 ingest / infer / evaluate 的若干阶段都会失败。

3. **MemoryAgentBench 数据已就位**  
   至少需要以下文件存在：

```text
MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet
MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet
MemoryAgentBench/data/Long_Range_Understanding-00000-of-00001.parquet
MemoryAgentBench/data/Test_Time_Learning-00000-of-00001.parquet
```

---

### 这个仓库和 LightRAG / HippoRAG 子仓库最不一样的地方

RAPTOR 这条路线的核心产物是 **tree**，因此它的参数体系与另外两个后端不同：

- ingest 主要使用 `--save_dir`
- infer 主要使用 `--tree_dir`
- 树结构层数使用 `--tb_num_layers`

也就是说，本仓库操作的核心对象不是向量存储目录，也不是图索引目录，而是 **RAPTOR tree**。

---

### 当前仓库的成熟度判断

当前版本已经具备以下条件：

- 目录结构清晰
- 后端定位明确
- 四个任务脚本齐全
- 支持一键运行
- 支持分步 CLI
- 可以完成小规模到中等规模实验验证

但仍然属于**研究代码风格**，因此更适合：

- 实验复现
- 调参分析
- 结果比较
- 后续重构

而不是直接作为面向普通用户的产品系统。

---

## 1. 项目概述

`memoRaxis_Raptor` 是从原始 `memoRaxis` 中拆分出来的 RAPTOR 单后端实验仓库。拆分的意义不在于减少文件数量，而在于**收敛实验变量**，让以下关系更清晰：

- 记忆后端固定为 RAPTOR；
- 推理层保留统一的三种适配器；
- 数据集固定来自 MemoryAgentBench；
- 输出链条固定为 ingest → infer → evaluate → analyze。

从研究视角看，这样的拆分有三个直接好处：

1. 可以避免不同 M 轴后端之间的依赖相互干扰；
2. 可以更清楚地定位某个问题究竟来自 RAPTOR 逻辑、adaptor 逻辑还是评测逻辑；
3. 可以把 README、运行脚本、输出目录和调试方法都围绕 RAPTOR 本身组织起来。

本仓库的设计重点，不是“封装得像产品”，而是“让后端实验路径尽可能透明”。

---

## 2. 设计思路：R 轴与 M 轴的解耦

本仓库继承了 memoRaxis 最核心的设计理念之一：**推理与记忆分离**。

### 2.1 R-axis：推理适配器

当前仓库保留三种推理范式：

- `R1`：SingleTurnAdaptor
- `R2`：IterativeAdaptor
- `R3`：PlanAndActAdaptor

这三者的差别，主要体现在“如何使用检索到的证据”。

- `R1`：一次检索，直接回答
- `R2`：迭代检索，边查边判断是否足够
- `R3`：先规划再执行，允许多步扩展和补充检索

### 2.2 M-axis：RAPTOR 记忆后端

在本仓库里，M 轴被固定为 RAPTOR。  
其核心思想是通过 **树形摘要结构** 来组织文本内容，使长文本能够在不同粒度层级上被压缩与检索。

因此，RAPTOR 路线和普通向量索引不同，它的重点不只是“把 chunk 存进去”，而是“构建多层摘要树”。

### 2.3 这一设计的实验意义

这样的设计，让实验变成一件很清楚的事：

- 若固定 RAPTOR，可以比较 R1/R2/R3 的差异；
- 若固定某个 adaptor，可以横向和其他后端仓库做比较；
- 若修改树参数（如 `tb_num_layers`），可以观察同一记忆后端内部结构变化带来的影响。

也正因为这样，本仓库更接近一个**研究基线平台**，而不是一个对外服务的产品模块。

---

## 3. 目录结构说明

当前项目核心目录如下：

```text
memoRaxis_Raptor/
├─ config/
│  ├─ config.yaml
│  └─ prompts.yaml
├─ docs/
│  └─ bluePrint.md
├─ external/
├─ MemoryAgentBench/
├─ scripts/
│  ├─ Raptor_MAB/
│  │  ├─ data/
│  │  ├─ ingest/
│  │  ├─ infer/
│  │  ├─ evaluate/
│  │  ├─ analyze/
│  │  └─ debug/
├─ src/
│  ├─ __init__.py
│  ├─ adaptors.py
│  ├─ benchmark_utils.py
│  ├─ config.py
│  ├─ llm_interface.py
│  ├─ logger.py
│  ├─ memory_interface.py
│  └─ raptor_memory.py
├─ main.py
├─ run_all_tasks.py
└─ requirements.txt
```

### 3.1 `config/`

- `config.yaml`：模型接口、embedding 接口等运行配置
- `prompts.yaml`：三种 adaptor 的提示模板

### 3.2 `scripts/Raptor_MAB/`

这是当前最重要的实验入口集合。

- `data/`：数据预处理
- `ingest/`：构建 RAPTOR tree
- `infer/`：调用 R1/R2/R3 作答
- `evaluate/`：执行任务评测
- `analyze/`：做进一步结果分析
- `debug/`：接口检查、维度检查、故障定位

### 3.3 `src/`

这部分是一方核心实现，不是脚本堆叠层。  
尤其重要的文件包括：

- `adaptors.py`
- `memory_interface.py`
- `raptor_memory.py`
- `llm_interface.py`
- `config.py`
- `benchmark_utils.py`

### 3.4 `run_all_tasks.py`

这是当前仓库推荐的一键运行入口。  
它将四个任务按统一顺序串起来，降低首次运行时的路径和命令复杂度。

---

## 4. 运行环境准备

### 4.1 推荐环境

建议使用：

- Python 3.11
- Anaconda / Miniconda 环境
- Windows / Linux / macOS 均可
- 推荐统一使用 Anaconda Prompt 或同一类 shell

### 4.2 创建环境

```bash
conda create -n memoraxis_raptor python=3.11 -y
conda activate memoraxis_raptor
```

### 4.3 进入项目根目录

Windows 示例：

```bash
cd /d D:\memoRaxis_Raptor
```

### 4.4 安装基础依赖

```bash
pip install -r requirements.txt
```

### 4.5 安装 RAPTOR 及其依赖

若仓库中包含 `third_party/raptor`，推荐：

```bash
pip install -r third_party/raptor/requirements.txt
export PYTHONPATH=$PWD/third_party/raptor:$PYTHONPATH
```

Windows 下如果使用 Anaconda Prompt，则思路等价，只是环境变量设置方式不同。  
如果仓库对 RAPTOR 做了 vendored 引入并在 `src/raptor_memory.py` 中完成路径处理，则无需额外设置 `PYTHONPATH`，以实际仓库实现为准。

---

## 5. 配置说明

### 5.1 配置文件位置

```text
config/config.yaml
```
首次运行前，需要将 config/config.example.yaml 复制为 config/config.yaml，并手动填入可用的 API Key 与 Base URL。
### 5.2 配置的主要作用

配置文件主要承载：

- LLM 配置
- Embedding 配置
- 其他运行相关配置

RAPTOR 在构树阶段可能需要模型参与摘要、压缩或其他处理，因此模型接口是否可访问会直接影响 ingest 阶段。

### 5.3 配置示例

```yaml
llm:
  provider: openai_compat
  model: gpt-4o-mini
  base_url: "https://your-llm-base-url/v1"
  api_key: "YOUR_LLM_API_KEY"
  timeout: 120

embedding:
  provider: openai_compat
  model: text-embedding-3-small
  base_url: "https://your-embedding-base-url/v1"
  api_key: "YOUR_EMBEDDING_API_KEY"
  dim: 1536

database: {}
```

若该仓库需要公开，推荐将 `config.yaml` 保持为模板配置，不放真实 key。

---

## 6. 数据准备

四个任务依赖的原始数据文件如下：

```text
MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet
MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet
MemoryAgentBench/data/Long_Range_Understanding-00000-of-00001.parquet
MemoryAgentBench/data/Test_Time_Learning-00000-of-00001.parquet
```

在首次运行前，推荐先生成 preview 样本：

```bash
python scripts/Raptor_MAB/data/convert_all_data.py
```

或单独执行：

```bash
python scripts/Raptor_MAB/data/convert_parquet_to_json.py
```

生成后通常位于：

```text
MemoryAgentBench/preview_samples/
```

这一步对于后续评测尤为重要。

---

## 7. 使用方式总览

当前仓库保留两种使用方式：

### 7.1 一键运行

适合：

- 首次验证仓库是否可用
- 按 README 快速复现
- 小规模 smoke test
- 标准统一跑法

### 7.2 CLI 分步运行

适合：

- 单独调 ingest
- 单独调 infer
- 改参数后局部重跑
- 评测逻辑调试
- 实验分析

两种方式并不冲突。通常推荐先用一键运行确认大链路可用，再用 CLI 分步方式做深入实验。

### 固定 chunk 数成本实验

从本版本开始，四个 ingest 脚本统一支持 `--target_chunks`。<br>
该参数用于**固定 chunk 数的 ingest 成本实验**，适合研究：

- chunk 数变化对 build time 的影响
- chunk 数变化对 token 消耗的影响
- 不同任务在相同 chunk 数下的 ingest 开销差异

需要注意的是，`--target_chunks` 的设计目标是**实验可控性**，不是官方评测口径下的默认分块方式。因此：

- 当 `--target_chunks` 被设置时，会优先进入“固定 chunk 数实验模式”
- 这时原有的 `chunk_size / min_chars / auto chunking` 将被覆盖
- 该模式更适合成本分析，不建议直接拿来做正式准确率结论

---

## 8. 一键运行：`run_all_tasks.py`

### 8.1 它会做什么

`run_all_tasks.py` 会按顺序执行：

1. 数据预处理
2. ingest
3. infer
4. evaluate

覆盖四个任务：

- Accurate_Retrieval
- Conflict_Resolution
- Long_Range_Understanding
- Test_Time_Learning

### 8.2 最简命令

```bash
python run_all_tasks.py
```

### 8.3 常用命令

只跑四个任务的 `instance 0`：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0
```

四个任务的 `instance 0` 前 10 个问题的小型测试：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 10 --conflict_limit 10 --long_limit 10 --ttl_limit 10
```

只跑 `Accurate_Retrieval` 和 `Conflict_Resolution`：

```bash
python run_all_tasks.py --tasks acc conflict
```

只跑 R1 和 R2：

```bash
python run_all_tasks.py --adaptors R1 R2
```

复用已有 tree，跳过 ingest：

```bash
python run_all_tasks.py --skip_ingest
```

只做 ingest：

```bash
python run_all_tasks.py --skip_infer --skip_eval
```

---

## 9. 一键脚本参数说明

### 9.1 通用参数

#### `--tasks`

选择要执行的任务集合。默认等价于：

```bash
--tasks acc conflict long ttl
```

可选值：

- `acc`
- `conflict`
- `long`
- `ttl`

#### `--adaptors`

选择推理适配器。默认等价于：

```bash
--adaptors R1 R2 R3
```

可选值：

- `R1`
- `R2`
- `R3`
- `all`

#### `--tree_dir`

指定 RAPTOR tree 输出目录。默认值：

```bash
--tree_dir out/raptor_trees
```

#### `--output_suffix`

给结果文件附加后缀，用于区分实验。默认值为空字符串。

#### `--tb_num_layers`

RAPTOR tree 的层数参数。默认值：

```bash
--tb_num_layers 3
```

这是本仓库最重要的结构参数之一。

#### `--skip_preprocess`

跳过 preview 数据生成。

#### `--skip_ingest`

跳过 ingest，直接复用已有 tree。

#### `--skip_infer`

跳过推理阶段。

#### `--skip_eval`

跳过评测阶段。

---

### 9.2 Accurate Retrieval 专属参数

- `--acc_instance_idx`：默认 `0`
- `--acc_limit`：默认 `5`
- `--acc_chunk_size`：默认 `850`
- `--acc_target_chunks`：默认 `None`，用于强制 Accurate_Retrieval ingest 生成固定数量的 chunk

---

### 9.3 Conflict Resolution 专属参数

- `--conflict_instance_idx`：默认 `0-7`
- `--conflict_limit`：默认 `-1`
- `--conflict_min_chars`：默认 `800`
- `--conflict_target_chunks`：默认 `None`，用于强制 Conflict_Resolution ingest 生成固定数量的 chunk

---

### 9.4 Long Range Understanding 专属参数

- `--long_instance_idx`：默认 `0-39`
- `--long_limit`：默认 `-1`
- `--long_chunk_size`：默认 `1200`
- `--long_overlap`：默认 `100`
- `--long_target_chunks`：默认 `None`，用于强制 Long_Range_Understanding ingest 生成固定数量的 chunk

---

### 9.5 Test Time Learning 专属参数

- `--ttl_instance_idx`：默认 `0-5`
- `--ttl_limit`：默认 `-1`
- `--ttl_target_chunks`：默认 `None`，用于强制 Test_Time_Learning ingest 生成固定数量的 chunk

当 `*_target_chunks` 被设置时，对应任务会优先使用固定 chunk 数实验模式；<br>
原有 `chunk_size / min_chars / auto strategy` 仅作为未设置 `target_chunks` 时的默认逻辑。

---

## 10. CLI 分步运行

### 10.1 数据预处理

```bash
python scripts/Raptor_MAB/data/convert_all_data.py
```

或：

```bash
python scripts/Raptor_MAB/data/convert_parquet_to_json.py
```

### 10.2 Ingest

#### Accurate Retrieval

```bash
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 850 --save_dir out/raptor_trees --tb_num_layers 3
```

固定 chunk 数实验示例：

```bash
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 16 --save_dir out/chunk_sweep/acc_k16 --tb_num_layers 3
```

#### Conflict Resolution

```bash
python scripts/Raptor_MAB/ingest/ingest_conflict_resolution.py --instance_idx 0-7 --min_chars 800 --save_dir out/raptor_trees --tb_num_layers 3
```

固定 chunk 数实验示例：

```bash
python scripts/Raptor_MAB/ingest/ingest_conflict_resolution.py --instance_idx 0 --target_chunks 16 --save_dir out/chunk_sweep/conflict_k16 --tb_num_layers 3
```

#### Long Range Understanding

```bash
python scripts/Raptor_MAB/ingest/ingest_long_range.py --instance_idx 0-39 --chunk_size 1200 --overlap 100 --save_dir out/raptor_trees --tb_num_layers 3
```

固定 chunk 数实验示例：

```bash
python scripts/Raptor_MAB/ingest/ingest_long_range.py --instance_idx 0 --target_chunks 16 --save_dir out/chunk_sweep/long_k16 --tb_num_layers 3
```

#### Test Time Learning

```bash
python scripts/Raptor_MAB/ingest/ingest_test_time.py --instance_idx 0-5 --save_dir out/raptor_trees --tb_num_layers 3
```

固定 chunk 数实验示例：

```bash
python scripts/Raptor_MAB/ingest/ingest_test_time.py --instance_idx 0 --target_chunks 16 --save_dir out/chunk_sweep/ttl_k16 --tb_num_layers 3
```

## 固定 chunk 数 sweep 示例

若目标是只研究 ingest 开销，而不运行 infer / evaluate，可直接使用以下命令。

### 一键脚本方式

只跑 Accurate_Retrieval，固定 16 个 chunk，仅做 ingest：

```bash
python run_all_tasks.py --tasks acc --acc_instance_idx 0 --acc_target_chunks 16 --skip_infer --skip_eval
```

四个任务都跑 `instance 0`，并分别固定为 16 个 chunk，仅做 ingest：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_target_chunks 16 --conflict_target_chunks 16 --long_target_chunks 16 --ttl_target_chunks 16 --skip_infer --skip_eval
```

### 逐条命令方式

```bash
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 1 --save_dir out/chunk_sweep/acc_k1 --tb_num_layers 3
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 2 --save_dir out/chunk_sweep/acc_k2 --tb_num_layers 3
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 4 --save_dir out/chunk_sweep/acc_k4 --tb_num_layers 3
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 8 --save_dir out/chunk_sweep/acc_k8 --tb_num_layers 3
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 16 --save_dir out/chunk_sweep/acc_k16 --tb_num_layers 3
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 32 --save_dir out/chunk_sweep/acc_k32 --tb_num_layers 3
python scripts/Raptor_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --target_chunks 64 --save_dir out/chunk_sweep/acc_k64 --tb_num_layers 3
```

对另外三个任务，只需把脚本名与输出目录前缀替换为：

- `ingest_conflict_resolution.py` / `conflict_k*`
- `ingest_long_range.py` / `long_k*`
- `ingest_test_time.py` / `ttl_k*`

### 结果提取

每次 ingest 运行后，日志目录会生成 `*.events.jsonl`。<br>
可以直接执行：

```bash
python scripts/Raptor_MAB/analyze/report_ingest_sweep.py --pattern "log/*.events.jsonl"
```

输出文件为：

```text
out/chunk_sweep_reports/ingest_sweep_report.csv
```

可直接拿去画图。

### 10.3 Infer

#### Accurate Retrieval

```bash
python scripts/Raptor_MAB/infer/infer_accurate_retrieval.py --instance_idx 0 --adaptor all --limit 5 --tree_dir out/raptor_trees --output_suffix ""
```

#### Conflict Resolution

```bash
python scripts/Raptor_MAB/infer/infer_conflict_resolution.py --instance_idx 0-7 --adaptor all --limit -1 --tree_dir out/raptor_trees --output_suffix ""
```

#### Long Range Understanding

```bash
python scripts/Raptor_MAB/infer/infer_long_range.py --instance_idx 0-39 --adaptor all --limit -1 --tree_dir out/raptor_trees --output_suffix ""
```

#### Test Time Learning

```bash
python scripts/Raptor_MAB/infer/infer_test_time.py --instance_idx 0-5 --adaptor all --limit -1 --tree_dir out/raptor_trees --output_suffix ""
```

### 10.4 Evaluate

#### Accurate Retrieval

```bash
python scripts/Raptor_MAB/evaluate/evaluate_mechanical.py --results out/acc_ret_results_0.json --instance MemoryAgentBench/preview_samples/Accurate_Retrieval/instance_0.json
```

#### Conflict Resolution

```bash
python scripts/Raptor_MAB/evaluate/evaluate_conflict_official.py
```

#### Long Range Understanding

```bash
python scripts/Raptor_MAB/evaluate/evaluate_long_range_A.py --results out/long_range_results_0.json --instance_folder MemoryAgentBench/preview_samples/Long_Range_Understanding
```

#### Test Time Learning

```bash
python scripts/Raptor_MAB/evaluate/evaluate_ttl_mechanical.py --results_pattern out/ttl_results_*.json
```

---

## 11. 推荐工作流

### 第一步：做小规模 smoke test

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3
```

### 第二步：跑四任务 `instance 0`

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0
```

### 第三步：进入标准实验

```bash
python run_all_tasks.py
```

### 第四步：按需要切换到分步 CLI

例如：

- 调 `tb_num_layers` 时只重跑 ingest
- 改 adaptor 时只重跑 infer
- 改评测逻辑时只重跑 evaluate

---

## 12. 输出说明

### 12.1 tree 输出目录

默认目录：

```text
out/raptor_trees/
```

### 12.2 推理结果输出目录

默认目录：

```text
out/
```

典型文件名：

```text
out/acc_ret_results_0.json
out/conflict_res_results_0.json
out/long_range_results_0.json
out/ttl_results_0.json
```

### 12.3 评测输出

评测结果通常打印在终端中。  
不同任务的评测方式不同：

- Accurate Retrieval：mechanical evaluation
- Conflict Resolution：official evaluation
- Long Range Understanding：judge / LLM-based evaluation
- Test Time Learning：mechanical evaluation

---

## 13. 常见问题

### 13.1 一进脚本就报缺包

优先检查：

- Conda 环境是否激活
- `requirements.txt` 是否安装
- RAPTOR 本体是否安装
- Python 解释器是否来自目标环境

### 13.2 配置错误

优先检查：

- `config/config.yaml` 是否存在
- YAML 格式是否正确
- API 地址、模型名、key 是否可用

### 13.3 数据文件缺失

优先检查：

- `MemoryAgentBench/data/` 是否存在四个 parquet 文件
- 当前工作目录是否为项目根目录

### 13.4 结果文件找不到

优先检查：

- infer 是否真的完成
- `output_suffix` 是否改变了文件名
- 任务实例号是否与预期一致

---

## 14. 当前版本的边界

当前版本已经足以支持：

- 单后端实验
- 四任务复现
- 一键运行
- 分步调试
- 结果比较

但仍然属于研究代码，主要边界包括：

- 入口还不是统一工业级 CLI
- 某些评测脚本仍依赖固定文件命名
- 配置和外部接口强相关
- 运行稳定性取决于本地实验环境

因此，本仓库更适合作为**研究型、可演化的后端子项目**，而不是产品化交付形态。

---

## 15. 总结

`memoRaxis_Raptor` 的意义，不在于把 RAPTOR 封装成一个完全黑箱的系统，而在于：

- 将 RAPTOR 从多后端实验框架中独立出来；
- 保留统一推理适配器；
- 收敛实验变量；
- 提供清晰的 ingest / infer / evaluate 路径；
- 为后续重构、文档化和分析沉淀提供稳定基础。

从开发者视角看，这个仓库最重要的价值是：  
**RAPTOR 作为独立记忆后端，已经能够在统一推理框架下被单独运行、单独验证、单独改造。**
