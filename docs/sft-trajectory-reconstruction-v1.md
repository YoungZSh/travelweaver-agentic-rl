# SFT 轨迹重建 V1

## 目标

`travelweaver-sft-v4` 将闭源模型成功 rollout 转换为可审计的 action-only、clean ReAct 或
ReAct Recovery SFT 数据。接纳条件为
正常提交计划、Reward 有效、全部硬约束通过、`rft_accepted=true` 且最终 Reward 为 1.0。

action-only 对恢复型成功轨迹执行清洗：转换器从相同任务和 Scenario reset，跳过 invalid
action，按顺序重放有效 action，并使用环境重新生成 observation。只有重放后仍为 Reward 1.0
的轨迹才进入训练集。ReAct Recovery 则重放并保留 invalid action、工具错误和后续反思，采用逐
assistant 回合 loss mask，详见
[ReAct Recovery SFT V1](react-sft-recovery-v1.md)。

监督模式是显式字段，不能根据文本内容隐式推断：

- `action_only` 是默认模式，允许恢复型成功轨迹，跳过 invalid action 后确定性重放；
- `react` 仅接收来源明确标记为 `thinking=disabled`、全程零 invalid action 的成功轨迹，原样保留
  每个工具调用前模型可见的 `assistant.content`；来源 system prompt 和自然语言 user query 必须与
  重建上下文完全一致，不能在 ReAct 转换时修补题面；
- `react_recovery` 接收 thinking-disabled 的全部 Reward=1 ReAct 轨迹。clean 回合 mask 为 true；
  invalid assistant 回合和对应工具错误保留在上下文中，但该 assistant 回合 mask 为 false，后续
  反思与正确动作继续监督。

## 消息与泄漏边界

- system、初始 user observation 和中间 tool response 仅作上下文。
- 中间 tool response 默认使用 `travelweaver-model-tool-response-v1` 的 `delta` 模式，只保留
  本轮结果、错误和剩余步数；不重复 task、全部 candidates 或累计 visible ID。
- action-only 的 assistant `content` 为空；ReAct 保留来源中的可见自然语言 content。两种模式都
  只保留一个结构化 function call，且所有 arguments 都是 JSON object。
- 删除全部 `reasoning_content`，不训练 DeepSeek 的隐式或显式思维内容。
- 保留最终 `submit_plan` assistant call，但删除其后的 terminal tool response，避免 Reward、隐藏
  TaskSpec、witness 或验证明细进入模型输入。
- delta 响应不包含 Reward、完整 Observation 或 info；adapter 会拒绝混入这些快照字段。
  显式 `snapshot` 兼容模式仍只允许固定 `reward=0` 和 `reward_detail=null`。

## Qwen3.5 与 veRL

neutral JSONL 保留通用 messages/tools 结构。训练 adapter 使用本地 Qwen3.5-4B 官方 chat
template 渲染工具调用，不手写 XML。Parquet 将 messages/tools 保存为 JSON 字符串，避免 Arrow
将不同工具参数合并为带无关 null 字段的 struct；自定义 veRL Dataset 在 tokenization 前恢复对象。

模型对话的首条 user content 是原始自然语言题面，不是 reset observation 的 JSON 包装。
完整 reset 状态只保留在可审计 trajectory 中；后续 tool content 仍为机器协议 JSON。

Qwen 设置 `enable_thinking=false`。官方模板仍可能包含空 `<think></think>` 协议壳，veRL 的
generation-prefix mask 会将其排除，只监督 assistant tool-call token。逐条转换必须通过分轮拼接与
整段 `apply_chat_template` token 完全一致的检查，且不允许截断。

在 ReAct 模式下，可见 `assistant.content` 和 tool call 都进入 assistant loss；供应商私有的
`reasoning_content` 仍被拒绝。训练 adapter 会根据显式 `supervision_mode` 分别校验两种样本。

## 当前 633 条批次

```bash
uv run travelweaver rebuild-sft \
  --source data/generated/chinatravel-blended-200-v1.1-repolished-minimal \
    data/trajectories/chinatravel-blended-200-v1.1-deepseek-v4-flash-thinking-16k-rollout1.jsonl \
  --source data/generated/chinatravel-blended-500-v1.1-b01 \
    data/trajectories/chinatravel-blended-500-v1.1-b01-deepseek-v4-flash-thinking-16k-rollout1.jsonl \
  --output-dir data/sft/chinatravel-qwen3.5-4b-action-633-sft-v2-natural \
  --repair-surface-semantics \
  --supervision-mode action_only \
  --tool-response-mode delta
```

旧题面中，只有餐厅预算缺少其他用餐硬约束时才补“至少一顿用餐”；所有旧版市内交通方式约束
补“至少两个市内地点”。本批共修复 93 条。其他 540 条题面保持不变。当前不切分 train/dev，
全部样本写入 `all.parquet`，待后续批次完成后再按 Blueprint 语义家族分组切分。

633条旧 v3 rollout 使用 delta 协议重建后，Qwen3.5 官方模板统计的序列长度由
`max=101066 / p50=31205 / total=21669520` 降为
`max=22079 / p50=12052 / total=7801719`；assistant loss token 保持 `1139824` 不变，
监督占比由 `5.26%` 提高到 `14.61%`。该历史产物的 system prompt 保留当时
环境的 35 个有效动作硬上限，但不再使用与复杂多日任务冲突的 15 步软目标。
新 rollout 默认为 50 个有效动作和 60 个 API turn，旧产物不会被就地改写。
首条 user content 直接使用原始自然语言题面，不再携带 reset observation JSON。

## ReAct 转换

```bash
uv run travelweaver rebuild-sft \
  --source data/generated/<batch> data/trajectories/<no-thinking-rollout>.jsonl \
  --output-dir data/sft/<react-batch> \
  --supervision-mode react \
  --tool-response-mode delta
```

转换器通过 source tool-call ID 将每个可见 assistant 回合与已执行 action 对齐，然后重放 action 并
重新生成 observation。最终 `submit_plan` 的 assistant content 与 tool call 保留，但其后的 Reward
响应删除。首批 ReAct 数据不接纳任何 invalid action，避免程序性删行造成前后 reasoning 断裂。

V4 的 `react` 保持 clean-only，`react_recovery` 按原顺序重放 invalid action，保留工具错误作为
上下文，对 invalid assistant 回合设零 loss，并监督错误之后的可见反思和正确 function call。
具体序列化、重放与泄漏检查见
[ReAct Recovery SFT V1](react-sft-recovery-v1.md)。

## 工具字段顺序

模型侧工具 schema 与 function arguments 按 schema 递归使用 required-first 顺序。模型可见 JSON
和 Parquet 的 `messages_json` / `tools_json` 不得使用 `sort_keys=True`；用于 hash 和 audit 的
canonical JSON 独立排序。Qwen 官方模板会保持 arguments 的插入顺序，因此转换后必须检查典型
工具至少满足：

- `save_candidate`：`entity_id → purpose → note`；
- `search_intercity_transport`：`origin_city → destination_city → mode → earliest_departure`；
- `get_route`：`origin_place_id → destination_place_id → mode → start_time`。

`submit_plan` 的 plan、itinerary 和 activity 对象同样递归排序，不能只处理顶层参数。
