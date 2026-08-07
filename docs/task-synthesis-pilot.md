# TravelWeaver 50 条任务合成试运行

本文档记录已确认的任务合成方案和实现边界。这一阶段只调通 TravelWeaver 环境
和任务生成，不做 DeepSeek agent rollout、RFT 轨迹筛选、SFT/GRPO 训练或 MCP。

## 目标与原则

产出 50 条中文旅行任务，每条都同时具有：

- 模型可见的自然语言 query；
- 模型不可见的类型化 `TravelTaskSpec`；
- 由现有 13 个工具真实执行、可重放的 witness plan 和 evidence；
- 对该 witness 为 `reward=1.0`、`all_hard_pass=true` 的确定性证明。

正确性来自「先 witness，后约束」，而不是让 LLM 猜测任务的 DSL。DeepSeek 只负责
中文润色，不产生或修改预算、时间、实体、交通和数量约束。

## 分层数据模型

```text
PilotSlot
   │  固定配额：城市/天数/约束数/去返程模式
   ▼
WitnessResult ── 环境搜索、保存、路线和 submit_plan 实际通过
   │
   ▼
TaskBlueprint ── 与文案无关的内部类型化 DSL
   │
   ├── 规则渲染 ──> canonical query
   │                       │
   │                       ▼
   │                 DeepSeek 仅润色
   │                       │
   ▼                       ▼
TravelTaskSpec <── TaskSurface + constraint mentions
   │
   ▼
TravelReward(witness) == 1.0
```

`TaskBlueprint` 是合成器的权威语义层。它使用 TravelWeaver 内部 typed DSL，不直接
复用 ChinaTravel 的 Python 字符串 DSL，也不执行任意代码。`TaskSurface` 保存润色后 query、
每条约束的精确文本 span、模型/提示词版本和 token usage。

## 50 条冻结配额

| 维度 | 配额 |
| --- | --- |
| 目的地 | 上海、北京、南京、广州、成都、杭州、武汉、深圳、苏州、重庆各 5 条 |
| 天数 | 1 天 20 条，2 天 20 条，3 天 10 条 |
| 硬约束数 | 1 条×15，2 条×20，3 条×10，4 条×5 |
| 去/返程 | 火车/火车 20，飞机/飞机 10，火车/飞机 10，飞机/火车 10 |

至少 10 条混合交通任务会在 DSL 中分别写出 `leg=outbound` 和 `leg=return`。
苏州在当前 ChinaTravel 快照中没有航班，因此预留 5 个火车/火车槽位。预算、景点/餐饮/
酒店类别与包含、数量、时间、房型/房间数、市内交通等约束家族均覆盖；
非城际模式家族在本试运行中至少出现 3 次。

「去返程使用同一模式」不是全局限制，只是 20 条火车/火车和 10 条飞机/飞机配额；
另外 20 条强制混合模式，因此不会牺牲该维度的多样性。

## Witness 和润色验收

Witness builder 只把当前 episode 已搜索显示的实体保存为候选，并为同日相邻地点引用
`get_route` 返回的 `route_id`。城际时刻、营业时间、房型/房间数和所有费用均由环境
重算。Blueprint 派生后先对 witness 评分一次，润色并物化为最终 TaskSpec 后再评分一次。

DeepSeek 调用固定为 `deepseek-v4-flash`、thinking disabled、单一 required function call。
每个 Blueprint 最多尝试 3 次，全局最多 200 次 API 调用。最终接纳的 query 不允许模板
回退；必须通过以下规则校验：

- 起终城市、天数、人数、实体名、预算/时间等 protected literals 逐字保留；
- 不新增城市或数字，不引入额外偏好或否定要求；
- 上限/下限、去程/返程、确定值/包含语义不变；
- 每个 `constraint_id` 恰好有一个不重叠、能在 query 中唯一定位的 mention。

## 运行和产物

```bash
uv sync --extra api --dev
uv run travelweaver synthesize-tasks \
  --count 50 \
  --seed 20260807 \
  --output-dir data/generated/pilot-50-v1
```

输出目录包含：

- `manifest.json`：配置、版本、分布、API/token 用量和完成状态；
- `blueprints.jsonl` 与 `surfaces.jsonl`：语义层和文案层；
- `tasks.public.jsonl` 与 `tasks.oracle.jsonl`：可直接装载的公开/隐藏任务；
- `witnesses.jsonl`：计划、快照、证据和最终 Reward 明细；
- `quarantine.jsonl`：候选失败阶段和脱敏错误；
- `preview.md`：50 条 query 及其配额摘要；
- `records/`：按 slot 原子写入的恢复点。

相同输出目录只能用完全一致的配置续跑，不同配置会拒绝覆盖。`.env`、API key、
原始模型响应和 reasoning 不写入产物。`data/generated/` 默认被 Git 忽略。

## 后续阶段

本试运行通过后，可以把同一 Blueprint 生成多个 Surface，或扩展到约 1k 任务。
之后再单独增加 DeepSeek 工具 rollout、严格 RFT 过滤、SFT 轨迹转换和 GRPO batch runner；
这些不与本次环境/任务合成改动耦合。
