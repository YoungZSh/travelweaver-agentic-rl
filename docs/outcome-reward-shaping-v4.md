# TravelWeaver Outcome Reward Shaping v4

> 状态：已实现。本文只定义终局 Outcome Reward，不包含逐 Turn credit assignment、工具次数
> 奖励或轨迹长度惩罚。

## 1. 设计目标

Reward v4 解决 v3 的两类信息损失：语义非法的 `submit_plan` 不再全部压成 `-1`；已经能
评价但失败的计划也不再只按硬检查数量粗粒度排序。实现仍满足以下边界：

- 数据构造、离线 rollout、在线 GRPO 和 SFT/RFT audit 调用同一个 `TravelReward`；
- Reward 只读取 rollout 前冻结的 `TravelTaskSpec`、最终提交物和 episode 证据；
- witness 只证明可行，不是唯一答案；LLM Judge 和主观偏好不进入训练 Reward；
- 正式 admission 仍 fail-closed，Reward collect-all 分析不会把非法计划放进环境；
- Reward 是纯终局信号，不读取动作数、搜索顺序、重复调用或 token 数。

现有 654 题 Qwen3.5-4B benchmark rollout 中，201 条提交在 admission 阶段被拒绝，138 条
可评价计划违反硬约束。高频问题包括时间冲突、路线端点、任务元数据、交通 purpose、住宿
容量、营业时间和路线时间窗。这些证据用于确定原子检查覆盖面，不用于按 benchmark 频率设置
Reward 权重。

Shopping GRPO 项目值得借鉴的是“rollout 前冻结 requirements、只归一化 active obligations、
接受非 gold alternative”的工程原则；其 brand/model/options taxonomy、gold bonus、固定失败
阶梯和轨迹终止分数不适合直接移植到旅行执行图。

## 2. Frozen Outcome Contract

`FrozenOutcomeContract` 在评分前从 `TravelTaskSpec` 确定性编译，记录 TaskSpec hash、任务 ID、
active constraint IDs 和独立 contract hash。相同合同、提交物和证据必须产生完全相同的结果。

Reward v4 的状态语义为：

- `pass`：原子检查 `score=1`；
- `fail`：`0 <= score < 1`；
- `blocked`：模型造成的前置条件失败使下游检查无法独立判断；只作诊断，不进入分母；
- `not_applicable`：旧证据或当前任务没有该项，不进入分母；
- `unverifiable`：环境快照、TaskSpec 或 evaluator 自身无法可靠判断，整条样本
  `reward_valid=false`。

模型缺 candidate、route 或字段不是基础设施错误。上游检查正常记 `fail`，依赖它的下游检查
记 `blocked_by=<check_id>`，从而既不重复惩罚，也不会通过缩小分母获益。

## 3. 三个互斥维度

顶层维度固定为 A/V/G，权重均为 `1/3`。每个内建 predicate 在
`reward/registry.py::CHECK_DEFINITIONS` 中只能有一个 owner。

| 维度 | 唯一问题 | 当前内建检查 |
|---|---|---|
| A — Artifact conformance | 最终产物本身是否可评价、局部结构是否合法 | `terminal_plan`, `plan_structure` |
| V — Environment validity | 不考虑用户目标时，冻结证据是否支持这个计划可执行 | chronology、candidate/route grounding、candidate usage、intercity time、opening hours、uniqueness、meal、declared-party quantity、cost、declared-itinerary overnight diagnostic |
| G — Goal satisfaction | 计划是否满足当前冻结任务合同 | user-visible task metadata、requested content/nights/destination/boundaries，以及动态 TaskSpec constraints |

A 不读取 candidate 真伪或用户要求；V 只对计划自己声明的人数和天数检查内部可执行性，不读取
TaskSpec 目标值，也不判断预算、房型等当前任务目标；G 才比较声明值与冻结任务，并且不重复判断
JSON 结构、证据是否存在、时间是否可执行或逐夜住宿位置。TaskSpec 的具体 constraint kind 只在
G 内动态展开，不会成为新的顶层维度。一次修改仍可能同时造成两类独立错误（例如同时填错人数且
没有同步修改票数），但同一个 predicate 只评分一次，也不会跨维度互为前置条件。

层内对可评分 active checks 求平均：

```text
A = mean(active artifact scores)
V = mean(active environment-validity scores)
G = mean(active goal-satisfaction scores)
```

布尔/枚举使用 0/1；数量下限、集合覆盖、预算越界、合法 activity 比例等使用有自然语义的
完成比例。部分分只用于排列失败计划，不放宽严格成功标准。

`overnight_coverage` 是 v4 新诊断：每个非末日恰好一次住宿且末日无住宿。它参与 V shaping，
但第一版不参与 strict-success gate，以保证此前通过 v3 的合格 SFT 轨迹不会因新增规则被静默
改成负样本。正式环境原有“至少覆盖所需住宿晚数”的 admission 规则保持不变。

## 4. 标量映射与 admission

当正式 admission 通过且全部 active hard checks 为 `pass`：

```text
R = 0.5 + 0.5 * S
```

当前没有 scored soft constraint 时 `S=1`，所以既有严格合格 SFT 轨迹在 v4 下仍为 `1.0`。

只要 admission 未通过，或任一 strict hard check 未通过：

```text
R = min(-1 + (A + V + G) / 3, -1e-8)
```

因此失败始终在 `[-1, 0)`，成功在 `[0.5, 1]`。即使 collect-all 暂时漏掉 admission 的某个
原子原因，`admission_passed=false` 也会阻止非法提交进入成功区间。完全没有可评价 plan 时
固定为 `-1`。

`reward`、`rl_reward` 和兼容字段 `terminal_utility` 来自同一个标量。v4 继续输出 v3 的
`hard_score`、`soft_score` 和 `group_results` 供旧分析代码读取，同时新增：

- `dimension_scores` 与 `dimension_coverage`；
- `outcome_contract_hash`；
- `admission_passed`；
- 每个 check 的 `owner_dimension`、`score`、`blocked_by`、`affects_success` 和
  `affects_shaping`。

## 5. 环境与 SFT 兼容

Environment v0.7 在 schema 合法的 `submit_plan` 被拒绝后，使用 raw plan、已保存 candidates
和已查询 routes 调用 `evaluate_submission`。正式 validator 仍返回首个 admission error 用于
用户可读诊断，Reward evaluator 则只读地 collect-all，不修改 episode 状态。

EvidenceBundle v3 新增已使用 candidate 的 `entity_type` 和 `purpose` 元数据，使在线 Reward
可以验证 candidate 使用方式。读取旧 v2 evidence 时，这个新增检查显式记
`not_applicable`，不影响 strict gate 或 shaping；不会猜测旧数据不存在的 purpose。

SFT/RFT 接纳条件未放宽：必须正常 `plan_submitted`、`reward_valid=true`、全部 strict hard
checks 通过且 `sft_accepted=true`。对当前 633 条 action-only SFT 来源轨迹的实际离线重算结果
是 633/633 仍为 `1.0`；其中 22 条触发新的 overnight diagnostic，但不改变 strict gate。
Reward v3 的成功轨迹通过 v4 重算时仍为 1；旧 Reward
明细保留其原始版本，不按 v4 静默改写。Trajectory v10 记录 v4 明细，读取器继续显式支持 v3–v9。

## 6. veRL GRPO 配置

Qwen3.5-4B 的本地双 A800 profile 固定使用：

- 每题 8 个 rollout；一个 optimizer step 消费一个 8 样本 group；
- `algorithm.adv_estimator=grpo`；
- `algorithm.norm_adv_by_std_in_grpo=false`，保留组内中心化但不除以标准差；
- vLLM rollout TP=2，FSDP2 actor/ref，Ulysses SP=2；
- 65,536 总序列上限：8,192 prompt + 57,344 response；
- 多轮 AgentLoop 最多 60 个 assistant/user turn，环境仍最多 50 个有效工具动作；
- 使用 Qwen3.5 官方 chat template 和 `qwen3_coder` XML tool parser，不手工拼接工具协议；
- online AgentLoop 直接把 `TravelReward` 标量写入 `reward_score`，不启用 learned RM。

自定义 replay buffer 过滤所有零方差组，而不只过滤全 0 或全 1：`[c,c,...,c]` 对任意
`c` 都没有 GRPO 排序信号。`reward_valid=false` 或不完整的组也不会训练，但不计入“连续无
信号”次数。

若连续 10 个完整、Reward 有效的 group 都是零方差，且中间没有任何可用 group，训练会：

1. 抛出带明确标记的内部停止信号；
2. 保存当前 actor、optimizer 和 dataloader checkpoint；
3. 写入 `stop-report.json`，包含十个 group 的共同 Reward 和 A/V/G 均值；
4. 以成功状态结束训练入口。一个可用 group 会立即把连续计数归零。

采样器状态写入独立 JSON，并在 launcher 重启时恢复。过滤和停止都使用
`reward_extra_info.travelweaver_reward`，不存在第二套 Reward 函数。

## 7. 数据准备与启动

先把版本化生成任务转换为只含公开 prompt 和隐藏运行句柄的 Parquet：

```bash
PYTHONPATH=training/src:src uv run --project training python \
  training/scripts/prepare_grpo_prompts.py \
  --input-dir data/generated/<batch-1> \
  --input-dir data/generated/<batch-2> \
  --output data/grpo/<combined-batch>/train.parquet
```

manifest 会记录输入 manifest hash、Parquet hash，并声明不包含 witness 和 Reward label。先做
纯 CPU 配置检查，再启动双卡训练：

```bash
TRAIN_FILE=data/grpo/<batch>/train.parquet \
MODEL_PATH=training/outputs/<sft-run>/final-model \
bash training/scripts/run_qwen3_5_4b_travelweaver_grpo.sh --dry-run

TRAIN_FILE=data/grpo/<batch>/train.parquet \
MODEL_PATH=training/outputs/<sft-run>/final-model \
GPU_HOLD_HANDOFF=1 \
bash training/scripts/run_qwen3_5_4b_travelweaver_grpo.sh
```

launcher 的 CPU preflight 会核验数据 hash、隐藏标签隔离、Qwen3.5 模型类型、上下文长度、
veRL 的 std-normalization 参数、custom sampler hook、TransferQueue 版本和 15 个工具 schema。
通过全部 preflight 后才停止 GPU 0/1 holder；退出、失败或收到 `INT`/`TERM` 后恢复同一 holder。

## 8. 首轮实验需要观察

v4 公式和等权是结构性先验，不是已经证明最优的超参数。首个 pilot 应同时报告：零方差组
比例、每组 unique Reward 数、Reward 方差、A/V/G 方差、成功/混合/全失败组比例、filtered
Reward 水平分布和 stop 触发情况。只有在这些数据表明明确瓶颈后，才讨论权重、credit
assignment、过程惩罚或 dynamic sampling；不要一次混入多个算法变化。
