# TravelWeaver 任务合成 v2

本文档记录 witness-first 任务合成方案。当前阶段只生成可验证的任务、Scenario 和
witness，不做 Agent rollout、RFT 轨迹筛选、SFT/GRPO 训练或 MCP。

## 目标与原则

每条合成任务同时包含：

- 模型可见的自然语言 query；
- 模型不可见的类型化 `TravelTaskSpec`；
- 明确物化、可审计的 `ScenarioSpec`；
- 由现有 13 个工具实际执行、可重放的 witness plan 和 evidence；
- 对该 witness 为 `reward=1.0`、`all_hard_pass=true` 的确定性证明。

正确性来自「先 Scenario、再 witness、后约束」。DeepSeek 只润色中文表述，不产生或
修改预算、时间、实体、交通和数量等评分语义。

## 单一 seed 与复现

CLI 只暴露一个 `--seed`。目的地、实际起点、天数、人数、约束组合、交通、行程密度、
tightness 和 Scenario 都通过带命名空间的哈希，从该 seed 派生彼此隔离的确定性随机流。
因此新增一个维度不会重排其他维度，也不需要用户维护 `catalog_seed`、`entity_seed` 等
多个参数。

Scenario 的随机选择只发生在合成时。最终产物保存具体 effect、`scenario_id` 和
`world_snapshot_version`；回放直接应用这些物化差异，不重新抽样。LLM 服务本身不承诺
逐 token 确定性，因此完整复现以已保存的 Blueprint、Surface、Scenario 和 witness 为
权威，而不是重新请求润色接口。

## 数据流

```text
PilotSlot（单 seed、配额规划）
   │
   ├── ScenarioSpec（闭园/酒店不可用/车次取消/价格变化）
   │        │
   │        ▼
   └── ScenarioBackend ──> WitnessResult（环境真实执行并 submit_plan 通过）
                                  │
                                  ▼
                           TaskBlueprint
                                  │
                   规则渲染 ──────┴────── DeepSeek 仅润色
                                  │
                                  ▼
                         TravelTaskSpec + TaskSurface
                                  │
                                  ▼
                         TravelReward(witness) == 1.0
```

Scenario 是环境状态，不是用户约束，因此不会在 query 或公开工具协议中暴露。Agent 只会
看到应用 Scenario 后的正常搜索结果。`TaskBlueprint` 仍是评分语义的权威层；
`TaskSurface` 保存 query、约束 span、模型/提示词版本和 token usage。

## 默认 100 条配额

配额使用 Hamilton 最大余数法，可扩展到任意正整数；下表是 `count=100` 的准确分布。

| 维度 | 配额 |
| --- | --- |
| 目的地 | 10 个城市各 10 条 |
| 天数 | 1/2/3/4/5 天分别为 10/30/35/15/10 |
| 人数 | 1–4 人各约 22.5%，5–6 人各 5% |
| 硬约束数 | 1/2/3/4/5/6 条分别为 10/20/30/25/10/5 |
| 去/返程 | 火车/火车 40，其他三种组合各 20 |
| 市内交通 | 出租车 40，地铁 35，步行 25 |
| tightness | easy/medium/hard 为 25/50/25 |
| Scenario | 正常 70，闭园 8，酒店不可用 6，车次取消 8，价格变化 8 |
| 文案风格 | 10 种开头、组织顺序和语气各 10 条 |

约束组合使用家族频次和两两共现次数共同做贪心均衡，避免“人数等于约束数”等取模相关性。
实际起点会先过滤没有可用同日往返的 OD，再按已使用的 OD 和起点次数均衡。苏州没有
航班的快照事实也在分配时保留。

行程有 1 或 2 个景点/天，部分任务加入邻近餐食。4–5 天任务固定为 1 个景点/天，避免
超过 Agent 的步骤预算。市内路线使用空间聚类和硬边界：步行单段不超过 2 km，出租车
不超过 30 km，地铁总路程不超过 40 km 且步行接驳不超过 2 km。预算和时间约束根据
easy/medium/hard tightness 使用不同余量，但具体阈值始终由已通过的 witness 反推。

## ChinaTravel 混合覆盖 200 题

完整的初始设计与验收配额见
[ChinaTravel 混合覆盖 200 题合成计划](chinatravel-blended-200-v1-plan.md)。
自然表面、Human 和 Preference 的定向修正版见
[V1.1 试验计划](chinatravel-blended-200-v1.1-plan.md)。

`--profile chinatravel_blended_v1` 是固定 200 题的合成配置：Easy-like 50、
Medium-like 70、Human-like 50、Preference-like 20、Generalization 10。场景配额固定为
正常 180、景点关闭 5、酒店无房 4、交通取消 5、价格上涨 6。天数、人数、城市和约束
家族使用 `0.65 × ChinaTravel 固定先验 + 0.35 × 均匀先验`，所有结构随机性仍只来自
一个 `--seed`。

Human-like 中 35 条保留方括号元数据，15 条为纯自然对话，并只使用 Blueprint 分配的
独自、情侣、朋友或亲子背景。它显式启用 `human_conservative` 校验：实体、数字、单位、
时间和交通方式仍逐字保留；上下限方向不可模糊；但确定句可写成“往返坐高铁”“想去
西湖”“酒店订2间房”等真人表达。偏好必须返回独立 mention，最终进入
`unscored_preferences`。其余四类继续使用 `strict` 校验。

Preference-like 每题只有一个主要偏好。14 条覆盖官方六类偏好且每类至少两条，另 6 条
来自可审计扩展池。合成器生成至少两个通过同一组硬约束的 witness 候选，按偏好对应的
确定性指标选优，并把候选指标、选择结果和逐候选硬 Reward 写入
`preference-audit.jsonl`；偏好本身不进入训练 Reward。

## Scenario

`ScenarioSpec` 保存 base snapshot、profile 和逐条 effect。当前支持：

- 景点不可用；
- 酒店不可用；
- 去返程部分车次/航班取消（始终保留可行候选）；
- POI 价格变动；
- 无变化的正常世界。

这些差异由 `ScenarioBackend` 统一应用到搜索、详情、附近查询、路线和城际交通上。公开
Function Calling schema 不增加 Scenario 参数，也没有可供 Agent 查询的“世界 seed”。

## Witness 与润色验收

Witness builder 只保存当前 episode 已搜索显示的实体，并为同日相邻地点引用
`get_route` 返回的 `route_id`。城际时刻、营业时间、房型/房间数和费用均由环境重算。
Blueprint 派生后评分一次，润色并物化为最终 TaskSpec 后再评分一次。

DeepSeek 使用 thinking disabled 和 required function call。每个 Blueprint 最多润色 2 次；
若两次均未通过严格校验，就使用该任务已分配 style 的确定性 canonical 文案，不更换
Blueprint 或 OD。默认全局上限 300 次 API 调用。最终 query 必须通过以下规则：

- 起终城市、天数、人数、实体名、预算和时间等 protected literals 逐字保留；
- 不新增城市、数字、额外偏好或否定要求；
- 上下限、去返程、确定值和包含语义不变；
- 每个 `constraint_id` 恰好有一个可唯一定位且不重叠的 mention。

## 运行与产物

```bash
uv sync --extra api --dev
uv run travelweaver synthesize-tasks \
  --count 100 \
  --seed 20260807 \
  --max-api-calls 300 \
  --output-dir data/generated/pilot-100-v2.1
```

输出目录包含：

- `manifest.json`：配置、版本、分布、API/token 用量和完成状态；
- `blueprints.jsonl`、`surfaces.jsonl`、`scenarios.jsonl`：三层权威产物；
- `tasks.public.jsonl` 与 `tasks.oracle.jsonl`：公开任务和隐藏 TaskSpec/Scenario；
- `witnesses.jsonl`：计划、快照、证据和最终 Reward 明细；
- `diversity.json`：OD、天数、人数、路线、Scenario 和约束共现统计；
- `alignment.json`：profile 配额、硬 Reward、去重、benchmark 原句复用和 Human 文案验收；
- `preference-audit.jsonl`：Preference-like 多 witness 选优审计；
- `polish-audit.jsonl`：每次 LM tool call 的请求、原始响应、解析 payload、验收结果和
  失败原因；
- `preview-*.md`：按任务类型输出的分类预览；
- `quarantine.jsonl`：候选失败阶段和脱敏错误；
- `preview.md`：所有 query 的可读预览；
- `records/`：按 slot 原子写入的恢复点。

相同输出目录只能用完全一致的配置续跑，不同配置会拒绝覆盖。`.env`、API key 和模型
reasoning 不写入产物；为分析 validator 误杀，surface polisher 的原始 tool response 会写入
本地 `polish-audit.jsonl`。`data/generated/` 默认被 Git 忽略。

如果只需要重跑自然语言 surface，而不改变 Blueprint、Scenario、witness 和硬 Reward，使用：

```bash
uv run travelweaver repolish-tasks \
  --input-dir data/generated/chinatravel-blended-200-v1.1 \
  --output-dir data/generated/chinatravel-blended-200-v1.1-repolished \
  --llm-concurrency 256 \
  --validation-policy minimal_semantic \
  --max-api-calls 400
```

`minimal_semantic` 是默认策略：会把不会造成 query 与隐藏 TaskSpec 矛盾的问题记录为
`validation_warnings`，只有关键事实改变或硬约束弱化才重试。需要复现实验早期的逐字校验时，
可显式传入 `--validation-policy strict`。

混合覆盖版本的标准命令为：

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1 \
  --count 200 \
  --seed 20260808 \
  --max-api-calls 400 \
  --output-dir data/generated/chinatravel-blended-200-v1
```
