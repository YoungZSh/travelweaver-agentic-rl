# TravelWeaver 任务合成 v2

本文档记录 witness-first 任务合成方案。当前阶段只生成可验证的任务、Scenario 和
witness，不做 Agent rollout、RFT 轨迹筛选、SFT/GRPO 训练或 MCP。

## 目标与原则

每条合成任务同时包含：

- 模型可见的自然语言 query；
- 模型不可见的类型化 `TravelTaskSpec`；
- 明确物化、可审计的 `ScenarioSpec`；
- 由当前公开工具实际执行、可重放的 witness plan 和 evidence；
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
实际起点由目录层先做均衡；witness 优先尝试目录起点，失败时再使用确定性的候选城市，
避免每个槽位先扫描所有 OD。苏州没有航班的快照事实也在分配时保留。

行程有 1 或 2 个景点/天，部分任务加入邻近餐食。4–5 天任务固定为 1 个景点/天，避免
超过 Agent 的步骤预算。市内路线使用空间聚类和硬边界：步行单段不超过 2 km，出租车
不超过 30 km，地铁总路程不超过 40 km 且步行接驳不超过 2 km。预算和时间约束根据
easy/medium/hard tightness 使用不同余量，但具体阈值始终由已通过的 witness 反推。

## ChinaTravel 混合覆盖 profile

完整的初始设计与验收配额见
[ChinaTravel 混合覆盖 200 题合成计划](chinatravel-blended-200-v1-plan.md)。
自然表面、Human 和 Preference 的定向修正版见
[V1.1 试验计划](chinatravel-blended-200-v1.1-plan.md)。

`--profile chinatravel_blended_v1` 和 `chinatravel_blended_v1_1` 以已验收的 200 题
配比为基准：Easy-like 25%、Medium-like 35%、Human-like 25%、Preference-like 10%、
Generalization 5%。场景基准配比为正常 90%、景点关闭 2.5%、酒店无房 2%、交通取消
2.5%、价格上涨 3%。任意 `--count` 都通过 Hamilton 最大余数法确定性地换算为整数
配额；余数并列时由 `--seed` 决定。原来的 `count=200, seed=20260808` 槽位保持逐条
不变。天数、人数、城市和约束家族使用
`0.65 × ChinaTravel 固定先验 + 0.35 × 均匀先验`，所有结构随机性仍只来自一个
`--seed`。

在 200 题基准中，Human-like 有 35 条保留方括号元数据、15 条为纯自然对话；更大批次
按相同比例缩放，并只使用 Blueprint 分配的
独自、情侣、朋友或亲子背景。它显式启用 `human_conservative` 校验：实体、数字、单位、
时间和交通方式仍逐字保留；上下限方向不可模糊；但确定句可写成“往返坐高铁”“想去
西湖”“酒店订2间房”等真人表达。偏好必须返回独立 mention，最终进入
`unscored_preferences`。其余四类继续使用 `strict` 校验。

Preference-like 每题只有一个主要偏好。官方偏好和可审计扩展偏好按 70%/30% 缩放，
组内尽量均衡；在 200 题基准中对应 14 条官方偏好和 6 条扩展偏好。合成器生成至少两个
通过同一组硬约束的 witness 候选，按偏好对应的
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
- 多值组内的“且/分别”、`any_of` 组间的“或”以及 `not_in/exclude` 的否定范围不变；
- 餐厅预算必须同时明确至少安排一顿用餐，市内交通方式必须同时明确至少安排两个市内地点，
  避免约束因没有对应餐厅或路线而被架空；
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

- `manifest.json`：配置、版本、API 用量、已完成槽位数和最近一条实时进度事件；
- `progress.jsonl`：追加式记录 `slot_started`、`slot_prepared`、`slot_completed`、失败、
  中断、恢复和最终完成事件；
- `blueprints.jsonl`、`surfaces.jsonl`、`scenarios.jsonl`：三层权威产物；
- `tasks.public.jsonl` 与 `tasks.oracle.jsonl`：公开任务和隐藏 TaskSpec/Scenario；
- `witnesses.jsonl`：计划、快照、证据和最终 Reward 明细；
- `diversity.json`：OD、天数、人数、路线、Scenario 和约束共现统计；
  其中 `destination_replacements` 单独记录不可行原目的地的确定性补位数量与方向；
- `alignment.json`：profile 配额、硬 Reward、去重、benchmark 原句复用和 Human 文案验收；
- `preference-audit.jsonl`：Preference-like 多 witness 选优审计；
- `polish-audit.jsonl`：每次 LM tool call 的请求、原始响应、解析 payload、验收结果和
  失败原因；
- `preview-*.md`：按任务类型输出的分类预览；
- `quarantine.jsonl`：候选失败阶段和脱敏错误；
- `preview.md`：所有 query 的可读预览；
- `records/`：每个槽位在 witness、题面和最终 Reward 全部通过后立即原子写入的恢复点。

生成过程中以 `records/` 为恢复真相源，不等待整批 witness 全部完成。进程中断后用完全相同的
命令和输出目录重跑，会跳过已有记录并从未完成槽位继续；最终的 `tasks.*.jsonl`、
`witnesses.jsonl` 和统计文件在全部槽位完成后按 slot 顺序汇总。相同输出目录只能用完全一致的
配置续跑，不同配置会拒绝覆盖。`.env`、API key 和模型
reasoning 不写入产物；为分析 validator 误杀，surface polisher 的原始 tool response 会写入
本地 `polish-audit.jsonl`。`data/generated/` 默认被 Git 忽略。

默认 backend 使用 `--witness-concurrency` 个独立进程并行准备槽位，每个进程复用一个
backend；主进程仍是唯一写入者，因此完成一条就立即形成恢复点。注入自定义 backend 的测试
保持串行，避免把不可序列化状态传入子进程。兼容 TaskSpec v2 的逻辑采样规则见
[DSL 多样性 V1](dsl-diversity-v1.md)。

worker 对原目的地只尝试一轮互异 origin；若目的地与时刻、步行聚类或公开可见性组合
结构性不可行，剩余尝试会转向 seed 派生的替代目的地。替换保持该槽位的任务类型、Scenario、
约束配方和难度维度，并写入 `slot_replaced`。任何单槽位失败都不会提前取消同批其他结果；
全部独立 future 收集并持久化后才汇总失败。

程序化 ReAct 轨迹合成遵循相同约束，并把两阶段中间结果分别写入
`<output-stem>.work/capabilities/` 和 `<output-stem>.work/records/`。能力分析或轨迹构造任一条
完成后都会由主进程立即落盘；恢复时只调度缺失槽位，不会重新启动已完成批次的 CPU worker。
DeepSeek 润色和模型 rollout 是网络 I/O，并继续使用线程并发及各自的逐任务恢复目录。

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

`repolish-tasks` 默认只接受 `status=complete` 且 records 数量等于源 manifest `count` 的批次；
确需抽查未完成批次时必须显式使用 `--allow-partial-input`。每条润色结果在 future 完成后立即
原子写入 `records/` 并追加 `progress.jsonl`，中断后使用相同命令和输出目录只会重跑缺失
槽位。单条失败进入 `quarantine.jsonl`，不会丢弃同批其他已经成功、已经产生 API 成本的结果。

混合覆盖版本的标准命令为：

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1 \
  --count 200 \
  --seed 20260808 \
  --max-api-calls 400 \
  --output-dir data/generated/chinatravel-blended-200-v1
```
