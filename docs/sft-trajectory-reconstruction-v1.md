# SFT 轨迹重建 V1

## 目标

`travelweaver-sft-v2` 将闭源模型成功 rollout 转换为可审计的 action-only SFT 数据。接纳条件为
正常提交计划、Reward 有效、全部硬约束通过、`rft_accepted=true` 且最终 Reward 为 1.0。

恢复型成功轨迹中的 invalid action 不作为纠错监督。转换器从相同任务和 Scenario reset，跳过
invalid action，按顺序重放有效 action，并使用环境重新生成 observation。只有重放后仍为 Reward
1.0 的轨迹才进入训练集。

## 消息与泄漏边界

- system、初始 user observation 和中间 tool response 仅作上下文。
- 中间 tool response 默认使用 `travelweaver-model-tool-response-v1` 的 `delta` 模式，只保留
  本轮结果、错误和剩余步数；不重复 task、全部 candidates 或累计 visible ID。
- assistant `content` 为空，只保留一个结构化 function call；所有 arguments 都是 JSON object。
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

## 当前 633 条批次

```bash
uv run travelweaver rebuild-sft \
  --source data/generated/chinatravel-blended-200-v1.1-repolished-minimal \
    data/trajectories/chinatravel-blended-200-v1.1-deepseek-v4-flash-thinking-16k-rollout1.jsonl \
  --source data/generated/chinatravel-blended-500-v1.1-b01 \
    data/trajectories/chinatravel-blended-500-v1.1-b01-deepseek-v4-flash-thinking-16k-rollout1.jsonl \
  --output-dir data/sft/chinatravel-qwen3.5-4b-action-633-sft-v2-natural \
  --repair-surface-semantics \
  --tool-response-mode delta
```

旧题面中，只有餐厅预算缺少其他用餐硬约束时才补“至少一顿用餐”；所有旧版市内交通方式约束
补“至少两个市内地点”。本批共修复 93 条。其他 540 条题面保持不变。当前不切分 train/dev，
全部样本写入 `all.parquet`，待后续批次完成后再按 Blueprint 语义家族分组切分。

633条旧 v3 rollout 使用 delta 协议重建后，Qwen3.5 官方模板统计的序列长度由
`max=101066 / p50=31205 / total=21669520` 降为
`max=22079 / p50=12052 / total=7801719`；assistant loss token 保持 `1139824` 不变，
监督占比由 `5.26%` 提高到 `14.61%`。当前 633 条的 system prompt 保留
环境的 35 个有效动作硬上限，但不再使用与复杂多日任务冲突的 15 步软目标。
首条 user content 直接使用原始自然语言题面，不再携带 reset observation JSON。
