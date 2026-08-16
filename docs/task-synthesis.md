# TravelWeaver 任务合成

本文档描述当前唯一受支持的正式合成配置 `chinatravel_blended_v1_1`。历史 pilot、V1 和
official-hybrid 配置已经退出生产流程；已有产物仍可作为审计记录保留，但新批次必须使用本配置。

## 设计边界

任务生成遵循“先证明可行，再生成题面”的顺序：

1. 单一 seed 派生城市、天数、人数、约束、偏好、Scenario 和表层风格；
2. 在固定 ChinaTravel 世界快照中构造满足全部硬约束的 witness；
3. 从 witness 派生版本化 Blueprint、TaskSpec 和 canonical query；
4. LLM 只能改写中文题面，不能改变实体、数值、约束方向或作用域；
5. 重新解析改写结果，并用确定性 Reward 验证 witness；
6. 将失败原因、fallback 和全部中间结果写入可恢复的审计产物。

ChinaTravel benchmark 只提供能力分布与表达风格参考。合成任务不得复制 benchmark 原句，
也不直接复用 benchmark 题目作为训练样本。

## 当前混合分布

`chinatravel_blended_v1_1` 以 200 题配额为比例基线，并可确定性扩展到任意正整数：

| 任务族 | 200 题基线 | 主要目的 |
|---|---:|---|
| `easy_like` | 50 | 单约束和短组合能力 |
| `medium_like` | 70 | 多约束组合与长程工具使用 |
| `human_like` | 50 | 自然表述、隐式上下文和未计分偏好 |
| `preference_like` | 20 | 硬约束可行前提下的独立偏好审计 |
| `generalization` | 10 | 长天数、大团体和较少见组合 |

Scenario 基线为 180 个正常世界、5 个景点关闭、4 个酒店不可用、5 个城际交通取消和
6 个价格变化。Scenario 在 episode 开始前冻结，rollout 中不会随机变化。

## 生成命令

正式批次建议每批 500 题并使用不同 seed：

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1_1 \
  --count 500 \
  --seed 20260811 \
  --max-api-calls 1000 \
  --llm-concurrency 256 \
  --validation-policy minimal_semantic \
  --output-dir data/generated/<batch-name>
```

先做完全离线的结构检查时使用 `--canonical-only`。如果只需要保留已验收 Blueprint 和 witness、
重新改写题面，可运行：

```bash
uv run travelweaver repolish-tasks \
  --input-dir data/generated/<source-batch> \
  --output-dir data/generated/<repolished-batch> \
  --llm-concurrency 256 \
  --validation-policy minimal_semantic \
  --max-api-calls 1000
```

repolish 默认要求完整的当前-profile批次，并按 slot 立即落盘，因此同一命令可以安全恢复。
历史 profile 会被明确拒绝，避免把旧分布静默混入当前训练数据。

## 产物与验收

每个输出目录包含：

- `manifest.json`、`progress.jsonl`：配置、进度、版本和 API 用量；
- `records/`：每个 slot 的完整可恢复记录；
- `tasks.public.jsonl`、`tasks.oracle.jsonl`：公开题面与隐藏任务规范；
- `blueprints.jsonl`、`surfaces.jsonl`、`scenarios.jsonl`、`witnesses.jsonl`；
- `polish-audit.jsonl`、`preference-audit.jsonl`、`quarantine.jsonl`；
- `alignment.json`、`diversity.json` 和分类 Markdown 预览。

批次完成前必须同时满足：数量及任务/Scenario 配额准确、query 与 Blueprint 唯一、无 benchmark
原句复用、全部 witness 的硬 Reward 为 1，以及 preference audit 候选全部硬约束通过。偏好指标
与硬 Reward 保持分离：`unscored_preferences` 不进入训练 Reward。

`data/generated/` 是本地可审计产物目录并被 Git 忽略；不要把数据库、API 响应或生成批次提交到
仓库。
