# TravelWeaver Reward 与离线评估协议

本文档冻结 TravelEnv 第一版可训练评分协议。它只面向 TravelWeaver 支持的旅行规划
任务，不试图抽象购物、网页操作等其他 Agent 领域。

## 1. 设计边界

TravelReward 是确定性的终局 Reward。它只读取 rollout 前冻结的任务规格、模型提交的
结构化计划和环境快照证据，不在评分期间调用 LLM，也不执行 benchmark 提供的 Python
约束代码。

```text
ChinaTravel oracle ──安全适配──┐
自由文本任务 ───────LLM 编译───┼──> TravelTaskSpec ──> TravelReward
未来自产任务 ───────Spec 优先──┘
```

ChinaTravel 只是任务来源和回归测试集。Reward 核心不得读取 `hard_logic`、benchmark
split 或 task ID。无法确定性验证的主观要求只进入离线 LLM Judge。

## 2. TravelTaskSpec

`TravelTaskSpec` 是模型不可见、版本化并冻结的任务评分合同，不是标准答案。它描述任务
必须满足的属性，允许多个不同但同样有效的行程。

规格包含：

- `trip`：出发地、目的地、天数、人数和可选日期；
- `constraints`：预算、时间窗口、交通、酒店、景点、餐饮、数量和包含/排除要求；
- `unscored_preferences`：环境无法客观验证但离线 Judge 可以参考的主观偏好；
- `spec_version`、环境快照版本、来源、编译器版本、输入哈希和规格哈希。

每条约束必须包含稳定 `id`、`kind`、`operator`、`value`、`scope`、`hardness` 和
`source_text`。组合约束仅支持 `all_of`、`any_of` 和否定；不允许嵌入任意代码。未知
`kind` 在 rollout 前拒绝，绝不静默视为通过。

自由文本通过 OpenAI-compatible function calling 编译为 `TaskSpecDraft`。编译器最多
尝试两次，随后执行 JSON Schema、字段范围、实体绑定、可支持性和基本可满足性验证。
未通过验证的任务进入隔离区，不参与 RFT/SFT 或 GRPO。ChinaTravel 的已有约束使用
安全 AST 适配器迁移；原始字符串只作为输入数据，永不 `exec`。

## 3. Environment evidence contract

最终计划必须引用本 episode 已保存的候选和已查询路线。`get_route` 返回稳定
`route_id`，相邻的同城活动通过路线引用建立可回放证据。

模型负责显式选择真正具有决策含义的字段，例如酒店房型和房间数；环境根据人数推导
门票、车票和出租车数量。单价、数量、晚数、路线费用和总费用全部由环境重新计算，
不接受模型自报总价。

终态生成：

- `PlanSnapshot`：规范化的最终计划；
- `EvidenceBundle`：实体、营业时间、路线、交通、价格和数量证据；
- `RewardResult`：约束状态、证据引用、标量 Reward 和有效性。

模型提交错误、伪造 ID 或缺少必需引用属于有效负样本；环境数据损坏、规格不受支持或
验证器异常属于基础设施问题，返回 `reward_valid=false`。

## 4. TravelReward v1

每个检查返回 `pass`、`fail` 或 `unverifiable`。硬检查分为两类：

1. 环境不变量：Schema、实体落地、时间无重叠、营业时间、路线可达、城际往返、住宿
   覆盖和费用/数量一致；
2. TaskSpec 约束：预算、时间、交通、酒店、景点、餐饮及包含/排除要求。

第一版全部激活的硬检查等权，全部可评分软约束也等权：

```text
H = passed_hard / active_hard
S = passed_soft / active_soft
```

没有软约束时 `S=1`。终局映射为：

- 基础设施或规格不可验证：`reward=0`、`reward_valid=false`，训练时剔除；
- 未提交可评估计划、步数耗尽或非法终止：`reward=-1`；
- 任一硬检查失败：`reward=-1+H`，范围 `[-1, 0)`；
- 所有硬检查通过：`reward=0.5+0.5*S`，范围 `[0.5, 1]`。

Reward 只在 episode 终态产生，不对普通搜索动作给正奖励，也不向 Actor 暴露隐藏规格
或评分明细。RFT/SFT 只接纳正常提交、`reward_valid=true` 且所有硬检查通过的轨迹；
软分仅保留为分析字段。后续 GRPO 直接使用上述标量 Reward。

## 5. Offline LLM Judge

LLM Judge 只用于离线评价主观质量，不参与在线 Reward 或第一版 RFT 接纳：

- 输入公开 query、脱敏轨迹、最终计划和环境证据摘要；
- 不提供确定性 Reward、ChinaTravel oracle、Gold 行程或原始数据库记录；
- 分别评价任务完成度、行程合理性、偏好满足、工具效率和最终表达质量；
- 每个判断必须引用可见证据，提示注入内容只按数据处理；
- 报告分别展示 Deterministic、Rubric 和 Trajectory 面板，不合成一个总分。

Reward 与 Judge 不一致是需要分析的诊断信号，不由 Judge 覆盖确定性结果。

## 6. 版本与验收

第一版使用独立版本号：TaskSpec、Environment、Tools、Observation、Trajectory、Reward
和 Judge schema 分别版本化。轨迹必须记录其实际使用的全部版本，旧轨迹不得静默按新
协议解释。

验收至少覆盖：否定与范围解析、每日/总计作用域、硬软区分、未知约束隔离、路线与营业
时间、人数/房间/票数、费用重算、相同 Spec 的来源无关性、RFT 严格过滤和 Judge 盲测。
ChinaTravel 的 654 条带 oracle 基础任务作为适配覆盖率与 Reward 回归集，而不是核心
Reward 的运行时依赖。
