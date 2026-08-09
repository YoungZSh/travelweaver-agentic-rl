# TravelWeaver 仓库协作指南

## 项目目标与当前阶段

TravelWeaver 是面向长程旅行规划 Agent 的确定性、可回放环境。环境、Function Calling
协议、通用任务规格、确定性 Reward、可审计任务合成和闭源模型批量 rollout 已经可用。
当前工作重心是：

1. 继续提高合成任务和 rollout 轨迹质量；
2. 将通过 Reward 的轨迹转换为版本化 SFT 数据；
3. 在独立训练环境中完成 SFT、在线 GRPO 及统一评测。

ChinaTravel benchmark 用于约束分布和表达风格参考，以及最终效果评估；合成训练数据不要求
逐字或逐题复制 benchmark，也不得直接复用 benchmark 原句。训练数据应覆盖相近能力分布，
同时保留更丰富的组合和表述。

## 两套隔离环境

### 根目录：环境、合成与 API rollout

- Python 固定为 3.10，使用 `uv`，不要向系统 Python 安装依赖。
- 初始化子模块：`git submodule update --init --recursive`
- 安装普通开发环境：`uv sync --dev`
- 安装 API rollout 依赖：`uv sync --extra api --dev`
- 准备 ChinaTravel 数据：`uv run travelweaver bootstrap chinatravel`
- 导入 benchmark：`uv run travelweaver import-tasks --split benchmark`
- 环境冒烟测试：`uv run travelweaver smoke-env`
- 根目录测试：`uv run pytest`
- 根目录静态检查：`uv run ruff check .`

根目录环境必须保持轻量，不得引入 PyTorch、CUDA、veRL、vLLM 或训练框架。

### `training/`：SFT 与 GRPO

训练栈是独立的 uv project，不得从根项目导入 GPU 依赖：

```bash
uv sync --project training --dev
uv run --project training python training/scripts/check_environment.py
```

当前训练环境：

- Python `3.10.19`
- veRL `main` commit `4a2cba76f7f605d2b9f56e640faaeaa71c2c7f71`（`0.9.0.dev`）
- PyTorch `2.10.0`，CUDA runtime `12.8`
- Transformers `5.5.3`，vLLM `0.19.1`
- FlashAttention `2.8.3`
- Flash Linear Attention `0.5.1`，FlashInfer `0.6.6`
- FSDP 训练后端，vLLM rollout 后端
- 6 张 NVIDIA A800 80GB PCIe，compute capability `8.0`

FlashAttention 在当前 glibc 2.31 主机上使用 CUDA 12.6 toolkit 本地编译；相关构建变量已写入
`training/pyproject.toml`。不要单独升级 PyTorch、vLLM、FlashAttention 或 CUDA 组合。当前不使用
SGLang、Megatron-LM、Apex 或 TransformerEngine。

修改训练依赖后应同步更新 `training/uv.lock`，重新运行环境检查。新增训练代码后使用：

```bash
uv run --project training pytest training/tests
uv run --project training ruff check training
```

## 当前代码结构

- `src/travelweaver/env/`：episode 状态机、13 个公开工具、稳定 ID、Scenario 和
  ChinaTravel backend。
- `src/travelweaver/data/`：数据库准备、校验和 benchmark 任务快照导入。
- `src/travelweaver/tasks/`：`TaskBlueprint`、`TaskSurface`、通用 `TravelTaskSpec`、编译与解析。
- `src/travelweaver/synthesis/`：配额目录、可行 witness、canonical 渲染、LLM polisher、
  preference audit 和版本化产物。
- `src/travelweaver/llm/`：provider-neutral OpenAI-compatible client 和 DeepSeek 配置适配。
- `src/travelweaver/rollout/`：Agent 循环、单任务 API rollout、可恢复的生成任务批量 rollout
  和完整轨迹。
- `src/travelweaver/reward/`：确定性约束验证、Reward 和严格 RFT 接纳过滤。
- `src/travelweaver/evaluation/`：与训练 Reward 分离的盲测 LLM Judge。
- `src/travelweaver/cli/`：数据、合成、重写、rollout 和 smoke test 命令入口。
- `training/`：隔离的 SFT/GRPO 依赖、配置、预处理、训练与测试代码。
- `tests/`：按根项目模块镜像组织的离线测试。
- `docs/`：架构、协议、Reward、合成计划和人工审计结论。
- `vendor/ChinaTravel/`：固定版本上游子模块；除非任务明确要求，不要直接修改。

## 协议与产物版本

修改序列化结构或行为时，检查并按需升级对应版本：

- Environment：`travelweaver-environment-v0.3`
- Observation：`travelweaver-observation-v3`
- Tools：`travelweaver-tools-v2-agent`
- TaskSpec：`travelweaver-task-spec-v2`
- Blueprint / Surface：`travelweaver-task-blueprint-v2` / `travelweaver-task-surface-v3`
- Reward：`travelweaver-reward-v1`
- Trajectory：`travelweaver-trajectory-v5`
- Model tool response：`travelweaver-model-tool-response-v1`（默认 `delta`，兼容 `snapshot`）
- Scenario：`travelweaver-scenario-v1`
- Synthesis / artifacts：`travelweaver-synthesis-v3` /
  `travelweaver-synthesis-artifacts-v5`
- Polisher prompt：`travelweaver-zh-polisher-v4`

不要在不升级版本和补兼容测试的情况下静默改变字段含义。读取旧快照时保持显式兼容或明确
拒绝，不要猜测缺失字段。

## 核心设计约束

1. 环境必须保持确定性和可回放。排序、分页、ID、路线、Scenario 和 Reward 不得依赖未固定
   的随机状态。
2. `tool_schemas.py` 是面向模型的统一 Function Calling 协议，不是 ChinaTravel 原始 API 的
   逐字段复制；上游字段转换集中在 backend。
3. Agent 只能引用本 episode 已展示的实体；计划只能引用已保存候选和已查询路线。不得绕过
   evidence contract 直接读取 oracle 数据。
4. ChinaTravel 只是首个任务来源。新约束进入通用 `TravelTaskSpec`，不得把 benchmark 特有
   逻辑硬编码进 Reward。
5. 训练 Reward 必须确定、可审计。LLM Judge 只做离线主观评估，不参与训练 Reward，也不与
   Reward 合并成单分数。
6. 默认使用进程内 Function Calling，不引入 MCP，除非项目范围明确改变。
7. Preference-like 的 `unscored_preferences` 当前不进入训练 Reward；只能通过独立 preference
   audit 或离线 Judge 分析，不能宣称 Reward 证明了偏好最优。
8. Scenario 是 episode 开始前冻结的替代世界快照，不是 rollout 中途随机注入的故障。

## 任务合成约定

- 正式混合数据使用 `--profile chinatravel_blended_v1_1`；配额按比例扩展，`--count` 不再限制
  为 200。分批使用不同 seed 合成，例如每批 500 条，以便及时人工抽查。
- 单一 seed 必须派生槽位、城市、人数、天数、约束、偏好、Scenario 和表层风格；不得混入
  未记录的随机状态。
- 先从固定世界构造可行 witness，再派生 Blueprint 和题面。LLM 只改写 `TaskSurface`，不能
  改动 Blueprint、witness、数字、实体或约束方向。
- 默认使用 `minimal_semantic` polisher validation。自然数字、同义表达和可修复 mention 可
  放宽，但城市、实体、归一化数值、交通方式、上下限方向、去返程作用域和硬约束弱化仍是
  hard error。
- 餐厅每餐预算必须明确至少安排一顿用餐；市内交通方式必须明确至少安排两个市内地点，防止
  约束因没有对应餐厅或路线而被架空。
- polisher 失败时允许使用相同 Blueprint 的自然 canonical fallback；fallback 必须记录完整
  audit，不能悄悄换题或绕过 Reward。
- 每批产物必须检查：数量和配额、query/Blueprint 唯一性、benchmark 原句复用、witness 与
  materialized TaskSpec 的硬 Reward、alignment、fallback 率和分类预览。

常用命令示例：

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1_1 \
  --count 500 \
  --seed 20260811 \
  --max-api-calls 1000 \
  --validation-policy minimal_semantic \
  --output-dir data/generated/<batch-name>
```

## DeepSeek 与批量 rollout 约定

- API 配置只从 `.env`/环境变量进入适配层，不得写入代码、日志或提交。
- 当前 rollout 基线为 `deepseek-v4-flash`、thinking enabled、`max_tokens=16384`、请求超时
  `600s`、每题一次 rollout。
- Surface polisher 始终使用 thinking disabled；供应商特有 thinking 参数只能留在 LLM 配置
  适配层。
- 所有支持并发的 DeepSeek 批处理默认使用 256 并发；当前 `repolish-tasks` 的
  `--llm-concurrency` 和 `rollout-generated` 的 `--concurrency` 都应设为 256。新增批量入口也
  使用相同默认值，除非用户明确修改。
- 批量 rollout 必须可恢复：按 `task_id` 跳过已有结果，将 API 错误单独写入 errors JSONL，
  不因单题错误丢失整批结果。
- 模型可见工具返回默认使用 `--tool-response-mode delta`，只发送本轮结果、错误和剩余步数；
  完整 StepResult 仍写入轨迹用于回放和审计。`snapshot` 只用于复现旧实验。
- 只有用户明确要求时才能调用付费外部 API。普通单元测试使用 fake client 或 mock。

```bash
uv run travelweaver rollout-generated \
  --input-dir data/generated/<batch-name> \
  --output data/trajectories/<rollout-name>.jsonl \
  --concurrency 256 \
  --max-api-turns 40
```

## SFT 数据转换约定

SFT 转换使用版本化、可审计的 `travelweaver-sft-v2`，不直接把原始 JSONL 临时拼成训练输入。

- 只接纳正常 `plan_submitted`、`reward_valid=true`、全部硬约束通过且 `rft_accepted=true`
  的轨迹。
- 最终 Reward 为 1.0 的恢复型轨迹可以用于 SFT。清洗时从 reset 开始跳过 invalid action，
  重放有效 action 并重生成 observation；不得直接从原 messages 中删行后继续使用旧 observation。
- 当前采用 action-only SFT：删除全部 `reasoning_content`。system、user 和 tool observation 只作
  上下文，不计算 loss；仅正确 assistant tool call 是监督目标。
- 首条 user content 使用真人题面的纯自然语言 `query`，不包装 JSON observation，不暴露
  episode ID、协议版本、合成类型或 Blueprint/Surface ID。工具 observation 仍使用版本化 JSON。
- SFT 中间 tool message 默认复用 `travelweaver-model-tool-response-v1` 的 `delta` 序列化，
  不重复 task、全部 candidates 或 visible ID 集合；旧 v3 轨迹可通过重放生成该格式。
- 删除最终 `submit_plan` 之后包含 Reward 明细的 tool response；Reward、隐藏 TaskSpec、oracle
  witness 和验证明细不得进入模型输入，只能写入 audit sidecar。
- 先产出 trainer-neutral JSONL，再通过目标模型官方 chat template 转换为 veRL Parquet；工具
  arguments 必须保持 JSON object，不能携带供应商原始字符串编码。
- Qwen3.5-4B 使用 `enable_thinking=false`。模板生成的空 thinking wrapper 是协议结构且须 mask，
  不属于 reasoning 训练内容；不要手工拼接 Qwen XML 工具调用。
- 清洗后校验 tool-call ID、消息配对、loss mask、协议版本和最大序列长度。超长轨迹隔离，
  不得截掉早期证据后只保留最终计划。
- 当前数据量较少时 Reward 1.0 样本全部进入 `all.parquet`，暂不切 train/dev。数据完成后再按
  Blueprint 约束组合或语义家族分组切分；benchmark 始终保持为独立测试集。

## 修改规范

- Python 使用 4 空格缩进、完整类型标注和 100 字符行宽。
- 优先复用已有 dataclass、模型和错误类型，不新增重复结构。
- 新增或修改公开工具时，同时更新 schema、环境执行与状态逻辑、协议版本、测试和文档。
- 修改 TaskSpec、Blueprint、Surface、Reward、轨迹、合成或训练数据格式时，检查序列化兼容、
  版本常量、manifest 和已有快照。
- 外部模型调用保持 provider-neutral；供应商参数只放配置适配层，不污染通用 Agent 循环。
- 一次性验证逻辑要么沉淀为可复用脚本并配测试，要么在验证结束后删除；不要长期保留用途不明
  的临时脚本、兼容 wrapper 或死代码。
- 不覆盖或回退与当前任务无关的用户改动。工作区可能非 clean，修改前后都要检查 status 和
  diff，只提交本任务文件。

## 数据与安全

- 不提交 `.env`、API Key、模型权重、数据库副本、下载缓存、生成任务或 rollout 轨迹。
- `data/generated/` 和 `data/trajectories/` 是本地生成产物；结论应来自 manifest/audit，不把
  大文件纳入 Git。
- `training/.venv/`、`training/checkpoints/`、`training/logs/` 和 `training/outputs/` 不提交。
- 删除或覆盖生成批次、轨迹、checkpoint 前先确认精确路径；优先保留可审计原始产物。

## 验证要求

- 小改动至少运行相关测试。
- 环境协议、Reward、TaskSpec、synthesis 或 rollout 改动运行完整 `uv run pytest` 和
  `uv run ruff check .`。
- 涉及真实 ChinaTravel backend 时再运行 `uv run travelweaver smoke-env`。
- 训练依赖或 GPU 接口改动运行 training 环境检查；训练预处理和 adapter 改动运行 training
  tests 与 Ruff。
- Bug 修复必须补充能复现问题的回归测试。

## 文档与提交

- 设计决策优先写入 `docs/`，README 保持为安装和入口说明，训练环境细节写入
  `training/README.md`。
- 注释解释约束和原因，不重复代码表面行为。
- 提交按逻辑拆分，使用简洁 Conventional Commit，如 `feat:`、`fix:`、`test:`、`docs:`、
  `chore:`。
- 不提交与当前任务无关的用户改动，不修改已发布提交历史，除非用户明确要求。
