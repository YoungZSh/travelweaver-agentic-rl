# ReAct Recovery SFT V1 设计

## 状态与目标

本文记录 2026-08-09 确认并实现的 ReAct 恢复型轨迹训练方案。转换器和训练 adapter 使用
`travelweaver-sft-v4`；V1/V2 action-only 和 V3 action-only/clean ReAct 仍显式兼容，V4 不静默
改变旧格式的监督含义。

目标不是把最终成功轨迹伪装成从未犯错的轨迹，而是保留闭源模型真实的
“错误动作—工具反馈—反思—修正”链路。invalid assistant 回合只作为 teacher-forced 上下文，
不计算 loss；错误后的自然语言反思和正确工具调用继续作为监督目标。这样既不要求模型复现
invalid action，也能训练 Agent 在真实工具反馈下恢复。

## 三种监督模式

| 模式 | invalid action | 可见 reasoning | 主要能力 |
|---|---|---|---|
| `action_only` | 删除后重放有效动作 | 全部删除 | 正确工具选择和参数生成 |
| `react` | 不接纳含 invalid 的轨迹 | 保留并监督 | 从头到尾的干净 ReAct |
| `react_recovery` | 保留但 mask 对应 assistant 回合 | 保留 | 根据工具错误反思和恢复 |

`react_recovery` 不是 action-only 的别名。两者都不监督 invalid action，但 action-only 会删除错误
及其 observation，也不保留任何自然语言推理；recovery 则将错误和工具反馈保留在上下文中，
并监督后续纠错过程。

## 消息与 loss 语义

典型样本如下：

```text
assistant: 初步分析并提交一个存在错误的计划     # 上下文，loss=0
assistant tool: submit_plan(错误计划)            # 上下文，loss=0
tool: 酒店缺少 route_from_previous_id            # 上下文，loss=0

assistant: 发现住宿前缺少路线，先补充查询       # 监督，loss=1
assistant tool: get_route(...)                   # 监督，loss=1
tool: 返回 route_id                              # 上下文，loss=0

assistant: 使用新路线重新提交                    # 监督，loss=1
assistant tool: submit_plan(正确计划)             # 监督，loss=1
```

具体规则：

1. system、user 和全部 tool message 一律不计算 loss。
2. `valid_action=false` 对应的整个 assistant message 都不计算 loss，包括可见
   `assistant.content` 和 function call。不能只 mask tool arguments 而继续监督导向错误动作的分析。
3. `valid_action=true` 的 assistant message 正常监督可见 content 和 function call。
4. invalid assistant message 虽然被 mask，token 仍参与后续回合的因果上下文；mask 不等于删除。
5. 删除最终成功 `submit_plan` 后的 terminal tool response。中间 invalid action 的 delta error
   response 可以保留，但必须拒绝 Reward、隐藏 TaskSpec、oracle witness 或完整验证明细泄漏。
6. 供应商私有 `reasoning_content` 仍不得进入训练数据。这里训练的是 thinking-disabled rollout
   中模型主动输出的可见自然语言 content。

该目标训练的是“错误已经发生且工具返回错误时如何恢复”，并不对 invalid action 施加对比学习式
负梯度。首次正确调用仍主要由 clean ReAct 和其他正确 assistant 回合提供监督。

## 确定性重放

Recovery 转换不能直接复制原始 messages，也不能先删掉 invalid 行再沿用旧 observation。转换器应：

1. 从同一个 Task、Scenario 和 seed reset 环境；
2. 将来源 assistant tool-call ID 与每个实际执行 step 严格对齐；
3. 按原顺序重放全部 action，包括 invalid action；
4. 重新生成版本化 delta response，并确认每一步的 `valid_action` 分类与来源一致；
5. 确认 invalid action 不改变环境状态；
6. 继续重放后续修正动作，最终仍须正常 `plan_submitted` 且 Reward=1；
7. 保留重放后的消息，按 step 有效性生成逐 assistant 回合监督 mask；
8. 将错误类型、mask 数量、重放差异和最终 Reward 写入 audit sidecar。

若错误代码或有效性分类在重放时发生变化，应隔离样本，不能猜测兼容。cursor 等运行期稳定 ID
允许使用现有显式 remap 机制，但必须记录映射。

Recovery 还必须覆盖两类合法的 masked 上下文边界：模型可能调用未注册工具名，这类调用没有可用
schema，因此仅允许在 `valid_action=false` 且整个 assistant 回合为零 loss 时原样保留 arguments；
有效动作仍必须调用已注册工具并使用 schema 排序。模型也可能传入拼写错误或已失效的分页 cursor；
若来源动作本身为 invalid 且 cursor 没有运行期映射，则原样重放该 opaque value，并确认环境仍返回
同类 invalid error。有效 `next_page` 仍必须完成严格 cursor remap。

## V4 中立格式

V4 新增：

```json
{
  "format_version": "travelweaver-sft-v4",
  "supervision_mode": "react_recovery",
  "assistant_loss_mask": [true, false, true, true]
}
```

`assistant_loss_mask` 按 messages 中 assistant 回合的出现顺序对齐，长度必须与 assistant message
数量完全一致。它放在样本顶层，不向传给模型官方 chat template 的 message 对象添加私有字段。
Parquet 使用独立的 `assistant_loss_mask_json` 字符串列保存，训练 Dataset 在逐回合 tokenization
时将 `false` 回合的 assistant loss mask 清零。

V4 adapter 必须继续验证：

- 分轮拼接 token 与整段官方 chat template 的 token 完全一致；
- `false` assistant 回合没有任何 loss token；
- 后续 `true` 回合的 content 和 tool call 均有监督；
- system、user、tool、空 thinking wrapper 和 terminal response 均不贡献 loss；
- tool-call ID、assistant/tool 配对和 arguments object 结构有效；
- 不截断超长轨迹。

旧 V1/V2 action-only 和 V3 action-only/clean ReAct 输入继续显式兼容；缺少 V4 mask 的 recovery
样本必须拒绝，不能根据 `valid_action` 文本或工具错误内容隐式推断监督范围。

## 首批 100 条实验依据

来源为固定 500 题批次中按题型和 Scenario 比例、由 seed 确定的 100 题 cohort，使用
DeepSeek V4 Flash、thinking disabled、`max_tokens=16384`、256 并发和 delta tool response，
每题 rollout 一次。

结果：

- Reward=1：79 条；失败：21 条；API error：0；
- clean ReAct：39 条；
- 最终成功且包含 invalid action：40 条；
- 40 条恢复轨迹共 54 次 invalid action；
- 其中 45 次为 invalid `submit_plan`，覆盖 36 条轨迹；
- 7 次为不可用 `get_route`，覆盖 2 条；
- 另有 1 次错误 `inspect_place` 和 1 次错误 `save_candidate`。

实现后实际得到 39 条全 true mask 的 clean ReAct 和 40 条含 masked invalid 回合的 recovery
ReAct，共 79 条带可见自然语言推理的 SFT 样本；共保留并 mask 54 个 invalid assistant 回合，
1,995 个有效工具调用保持监督。

完整 79 条经 Qwen3.5-4B 官方模板渲染后，长度为 12,559–46,843 tokens，中位数 28,216，
p90 为 40,990。Qwen3.5-4B checkpoint 的模型上下文上限是 262,144 tokens，模型能力不是当前
瓶颈。训练启动器现已采用 `max_length=65,536`，因此这批样本可完整训练；后续批次仍须在训练前
审计长度，禁止截掉早期证据和错误反馈。

模型侧工具 schema 和 arguments 同时改为递归 required-first 顺序，并在官方模板渲染后验证。
首批 V4 中典型参数顺序为：

- `save_candidate`：`entity_id → purpose → note`；
- `search_intercity_transport`：`origin_city → destination_city → mode`；
- `get_route`：`origin_place_id → destination_place_id → mode → start_time`。

## 验收要求

V4 实现包含以下回归测试：

- 接受最终 Reward=1、thinking disabled 且含可恢复 invalid action 的轨迹；
- invalid assistant content 和 tool call 全部为零 loss；
- invalid tool error 保留在上下文且为零 loss；
- 错误后的反思 content 和正确 tool call 有 loss；
- 多次 invalid action 的 mask 顺序稳定；
- 重放后错误类别变化、状态发生意外变化或最终 Reward 不再为 1 时拒绝；
- invalid delta response 中出现 Reward、TaskSpec、witness 或验证明细时拒绝；
- V3 clean ReAct 和 action-only 兼容结果不变；
- Qwen 官方模板全段一致性、最大长度和不截断检查通过。

## 500 条扩量结果

同一固定 500 题批次续跑后，全部任务均获得 trajectory，API error 为 0。最终 406 条
Reward=1 轨迹通过确定性重放并进入 V4 ReAct Recovery SFT，其中 215 条没有 invalid action，
191 条包含恢复过程；共保留并 mask 270 个 invalid assistant 回合，监督 10,222 个有效工具动作。

按题型的接纳率为 Easy-like 96.0%、Medium-like 85.7%、Human-like 64.8%、Preference-like
82.0%、Generalization 56.0%；按天数看，1–2 天为 94%–96%，5 天为 38.6%。当前批次适合训练，
但后续若要维持困难长任务占比，应按题型和天数对 Reward=1 产物做分层补采，不能只依赖原始题目
配额。

406 条经 Qwen3.5-4B 官方模板渲染后长度为 12,110–56,789 tokens，中位数 28,502，p90 为
39,564。模型上下文上限足够，但训练配置不得继续使用会截断完整轨迹的 32,768 上限；这批数据
应使用至少 65,536 的 `max_length`，或在不截断的前提下隔离更长样本。
