# 4000 条独立 Question 与 8:1:1 程序化策略轨迹计划

> 本文已由 `programmatic-sft-4000-v2-plan.md` 取代。V2 合并了本方案的 Question/轨迹设计、
> ChinaTravel 官方兼容审计、统一 Reward v2、完整边界路线证据及首次提交即终止语义。

## 状态与范围

本文记录 4000 条新合成旅行规划 Question 及一题一轨迹的程序化 SFT 数据方案。数据继续使用
`chinatravel_blended_v1_1` 的能力分布、确定性 Scenario、witness-first 合成、TaskSpec 和 Reward
协议。既有约 875 条 Reward=1.0 的 DeepSeek 真实轨迹保持原样，本阶段不将其计入 4000 条，
也不重新调用 API 补量。

整体分为两个阶段：

1. 生成 4000 个不同的 Question、Blueprint 和可行 witness；
2. 每个 Question 只生成一条程序化轨迹，按 `8:1:1` 分配高效成功、循环后恢复和证据充分
   立即提交三类策略。

第一批 500 题完成后暂停扩量，先审计题面和三类轨迹；通过人工检查后再生成其余 3500 题。

## Question 生成

### 协议与批次

沿用完整的 `chinatravel_blended_v1_1` 协议：

```text
Scenario
  → feasible witness
  → Blueprint
  → canonical Surface
  → LLM polish
  → minimal_semantic validation
  → canonical fallback（仅在改写失败时）
```

总量拆为 8 批，每批 500 条，每批使用独立且已记录的 seed。DeepSeek 改写保持 thinking
disabled、`minimal_semantic` 校验和每个批处理入口 256 并发；真实付费调用仅在用户明确授权后
执行。每批保留 manifest、alignment、polish/fallback audit、preference audit 和分类预览。

4000 条总体类型配额为：

| 类型 | 数量 |
|---|---:|
| Easy-like | 1000 |
| Medium-like | 1400 |
| Human-like | 1000 |
| Preference-like | 400 |
| Generalization | 200 |

每个 500 条批次按 profile 等比例分配为 `125 / 175 / 125 / 50 / 25`。Scenario、天数、人数、
约束数量、偏好类型和表层风格仍由该批 seed 确定性派生。

### 唯一性和补位

每批内部、8 批之间以及既有 B01/B02 数据之间检查：

- `task_id` 唯一；
- 规范化 Question 唯一；
- Blueprint `semantic_hash` 唯一；
- 不逐字复用 benchmark Question；
- 不将同一约束组合的纯同义改写计为新的 QA。

规范化 Question 至少统一 Unicode、空白和常见全半角标点。Blueprint 语义签名使用协议中的
`semantic_hash`，包括 trip、硬约束、偏好和冻结世界版本，而不依赖表层措辞。

发现重复时不直接删除。以原批 seed、槽位和补位序号派生新的 replacement seed，重新构造
Scenario、witness、Blueprint 和 Surface，直到该槽位获得唯一且完整通过审计的任务。补位过程
必须记录旧签名、冲突来源、replacement seed 和最终任务 ID。

## 轨迹分层与确定性分配

每批 500 条严格分配：

| 样本家族 | 每批 | 总计 |
|---|---:|---:|
| `efficient_success` | 400 | 3200 |
| `loop_recovery` | 50 | 400 |
| `evidence_ready_submit` | 50 | 400 |

使用总 seed 对任务类型、难度、天数和 Scenario 做确定性分层分配。每个 QA 只属于一个样本
家族，不能因为某类轨迹构造失败而滑动选择下一题。构造失败须隔离并修复该题或使用有记录的
replacement seed 补位。

所有程序化轨迹从环境 `reset` 开始真实执行，observation 使用
`travelweaver-model-tool-response-v1` 的 `delta` 模式生成，禁止从 witness 手工拼接工具返回。
工具 schema 和 arguments 递归使用 required-first 顺序。最终必须正常 `plan_submitted`、
Reward=1.0、`reward_valid=true`、`all_hard_pass=true`，并可确定性重放。

### 高效成功

`efficient_success` 在满足全部题面条件后不包含冗余动作。它不盲目追求最少步数：多日、餐饮、
住宿和市内路线仍须查询并保存完整证据。

```text
search required transport/place/hotel/restaurant evidence  [监督]
save every plan candidate                                [监督]
get every referenced inner-city route                    [监督]
submit_plan(complete witness plan)                        [监督]
```

该家族保持纯 action-only：所有 assistant 普通文本为空，全部 assistant tool-call 回合的 loss mask
为 `true`。验收时拒绝重复搜索、未使用候选、未引用路线、无意义翻页和不影响最终证据的工具动作。

### 循环后恢复

`loop_recovery` 先执行正确证据查询，再在一个无状态搜索工具上原样重复 1 至 3 次。重复查询仍由
环境执行并返回真实结果，不添加虚构的“检测到循环”提示。所有注入循环回合均保留为上下文，
但整个 assistant message 的 loss mask 为 `false`。

循环后的第一个正确动作是恢复点。恢复点在普通 `assistant.content` 中加入一条简短、可见、
受监督的自然语言反思，并在同一 assistant message 中调用下一步正确工具。例如：

```text
assistant.content:
  景点候选已经获得，无需继续重复查询；接下来保存候选并完成其余计划。
assistant.tool_call:
  save_candidate(...)
```

反思文本根据当前已经获得和仍缺失的证据动态生成，不能让所有样本背诵同一句模板。恢复回合的
自然语言和 tool call 一起监督；恢复后的其余正确动作也监督。供应商私有
`reasoning_content` 始终禁止进入数据。

### 证据充分立即提交

`evidence_ready_submit` 执行与高效轨迹相同的必要查询、保存和路线动作，但这些完整前缀只作为
teacher-forced 上下文，loss mask 全部为 `false`。提交点必须是环境状态首次具备完整计划证据的
位置，不能遗漏任何题面要求，也不能为制造样本而提前提交。

最终 assistant message 包含动态生成的可见反思和正确 `submit_plan`，是该样本唯一监督目标：

```text
assistant.content:
  题目要求的交通、景点和路线证据已经齐全，现在可以提交完整计划。
assistant.tool_call:
  submit_plan(...)
```

使用“证据已经齐全，可以提交”而不是“计划已经全部做完”，以准确描述提交前的环境状态。反思
须按实际任务列出关键证据类型；没有住宿或餐饮要求时不得声称已经准备了相应证据。

## 选择性监督与格式

新增 `action_selective` 监督模式，用于合法但不应模仿的循环动作、teacher-forced 正确前缀和带
可见反思的关键正确动作。它与现有模式的边界如下：

| 模式 | 合法动作可 mask | 可见 assistant content | 典型用途 |
|---|---|---|---|
| `action_only` | 否 | 空 | 高效成功 |
| `react_recovery` | 仅 invalid 回合 | 保留 | 真实工具错误恢复 |
| `action_selective` | 是 | 仅关键正确回合可见 | 程序化循环恢复、立即提交 |

`assistant_loss_mask` 继续按 assistant 回合对齐并作用于整个 assistant message：当值为 `false`
时，可见文本和 tool call 均为零 loss；当值为 `true` 时，两者一起监督。system、user 和 tool
message 始终不计算 loss。最终正确动作必须是 `true`。

因为 V4 当前将 mask 与 invalid action 绑定，新增合法 masked 回合和可见关键反思会升级 SFT
格式版本；旧 V1–V4 继续显式兼容，不静默改变旧数据语义。训练 adapter 必须继续验证官方 chat
template 全段一致性、assistant/tool 配对、空 thinking wrapper mask、最大序列长度和不截断。

## Audit sidecar

每条程序化样本至少记录：

- `sample_family` 和来源 Question 批次；
- Blueprint `semantic_hash`；
- 每个 assistant 回合的 `loss_mask`、`mask_reason` 和工具名；
- `visible_reflection`、`reflection_kind` 及其对应环境状态；
- 注入循环的工具、位置、次数和参数摘要；
- 首次证据完备位置；
- 原始动作数、重放动作数、终止原因、Reward 和硬约束结果；
- Question、Blueprint、Scenario、witness 和生成配置的可审计引用。

允许的 `mask_reason` 至少包括：

- `supervised_correct_action`；
- `injected_loop`；
- `teacher_forced_evidence_prefix`；
- `supervised_loop_exit_reflection`；
- `supervised_evidence_ready_reflection`。

Reward、隐藏 TaskSpec、oracle witness 和 `submit_plan` 后的验证明细只能写入 audit sidecar，不得
进入模型上下文。

## 第一批 500 条验收

Question 阶段先输出：

- 500 条完整任务及五类分类预览；
- 数量、类型、Scenario 和约束配额；
- batch 内及跨既有 B01/B02 的 task/query/Blueprint 去重结果；
- benchmark 原句复用检查；
- witness 与 materialized TaskSpec 的 Reward 结果；
- alignment、fallback 率、warning 分类和 Question 长度分布。

轨迹阶段再输出三类完整样例、动作数分布、token 长度和下列专项检查：

- 三类数量严格为 `400 / 50 / 50`；
- 高效成功无冗余动作；
- 循环动作全部 mask，恢复反思和正确动作进入 loss；
- 立即提交仅监督最终反思与 `submit_plan`；
- 恢复反思准确描述当前状态，提交反思只在证据完整后出现；
- 反思与后续 tool call 语义一致，文本重复率和模板化程度可接受；
- 所有用户输入均为纯自然语言，不含 JSON、task type、episode ID、Blueprint ID；
- 全部轨迹正常提交、Reward=1.0、硬约束通过且可确定性重放。

第一批人工审计通过前，不扩展剩余 3500 条，也不与既有 DeepSeek 真实轨迹混合。
