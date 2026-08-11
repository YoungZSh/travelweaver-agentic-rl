# 4000 条 Question、官方兼容约束与 8:1:1 程序化 SFT 合并实施计划

## 状态与目标

本文是第一版 4000 条 Question/轨迹方案与后续 ChinaTravel 官方兼容方案的合并版本，作为
本轮实现、合成和验收的唯一基线。既有约 875 条 Reward=1.0 的 DeepSeek 轨迹原样保留，
不计入本轮 4000 条，也不重新调用 API。

本轮分两阶段：

1. 使用 `chinatravel_official_hybrid_v2` 生成 4000 个不同 Question、Blueprint 和可行 witness；
2. 每题只生成一条程序化轨迹，严格分为 3200 条高效成功、400 条循环恢复和 400 条证据完备
   立即提交。

先完成并审计 B01 的 500 题及 500 条轨迹。未通过人工检查前，不扩展其余 3500 条。B01
验收后，用户确认先增加一个 1,000 条的 B02 中间检查点：B01+B02 合计 1,500 条，用于先训练
格式与基础工具能力；B02 通过相同验收后，再决定是否完成剩余 2,500 条。

## 为什么引入 official-hybrid v2

旧 `chinatravel_blended_v1_1` 能较好训练工具使用和 TravelWeaver 硬约束，但与固定版
ChinaTravel evaluator 仍存在四类差异：

- TravelWeaver 计划引用 `candidate_id`/`route_from_previous_id`，官方 schema 要求展开
  `position`、`price`、`cost`、`transports`、`TrainID`/`FlightID`；
- 旧 Reward 未完整覆盖官方对重复景点、重复普通餐厅、餐型时间窗和同日餐型唯一性的检查；
- 旧 witness 偏最低可行解，完整游览日的景点和餐饮密度不足；
- 原合成类型、天数和约束分布比官方 benchmark 更难，适合泛化但不适合全部作为核心分布。

V2 不复制 benchmark 原题，而是采用“官方核心能力 + 可控泛化尾部”的混合分布，并保留
TravelWeaver 更严格的证据契约。

## Question 分布、批次与 seeds

共 8 批，每批 500 条，seed 固定为 `20260821` 至 `20260828`。总体分布为：

| 类型 | 总量 | 用途 |
|---|---:|---|
| Easy-like | 1468 | 官方核心 |
| Medium-like | 732 | 官方核心 |
| Human-like | 754 | 官方自然表达核心 |
| Preference-base | 246 | 无偏好对照基线 |
| Preference-like | 400 | 六类偏好审计 |
| Generalization | 400 | 仅作为受控 Scenario 尾部 |

B01 的精确配额是 `184 / 92 / 94 / 30 / 50 / 50`。其余批次使用文档化的精确整数配额，
八批汇总后达到上表总量。六类偏好总体尽量均衡为 `67 / 67 / 67 / 67 / 66 / 66`。

Generalization 之外的 450 题使用正常世界。每批 50 条 Generalization 分为价格变化 15、
交通取消 13、POI 关闭 12、酒店不可用 10；Scenario 只改变未被 witness 选中的干扰候选，
不得靠故障替换 Question 的真实目标。

天数分布按约 `10% / 45% / 34% / 7% / 3% / 1%` 覆盖 1–6 天。默认完整游览日安排约
2 个不重复景点和 1 顿合理餐饮；首日和末日根据城际交通后的实际可用时长降级，不能为了密度
制造不可行行程。4–6 天长行程默认每个完整日 1 个景点，控制轨迹长度。

## witness-first Question 协议

每个槽位严格执行：

```text
frozen Scenario
  → feasible witness
  → Blueprint
  → canonical Surface
  → DeepSeek polish
  → minimal_semantic validation
  → canonical fallback（仅改写失败时）
```

DeepSeek 只润色自然语言，不得改变城市、实体、人数、天数、数字、约束方向或偏好。所有付费批量
入口默认 256 并发；polisher 固定 thinking disabled。CPU witness 和程序化轨迹构造使用有界
多线程，产物按槽位重新排序，保证并发不改变确定性结果。

Question polisher 与轨迹决策说明 polisher 是两条独立链路。前者只处理用户题面；后者在完整
程序化动作已经确定并回放成功后，为每个 assistant 工具回合润色可见的决策说明。不得用题面
polisher 代替轨迹 polisher，也不得让轨迹 polisher 重新选择动作。

先用 `--canonical-only` 完成 500 题的 witness、Reward、唯一性和官方 parity 预检；只有预检
通过后才调用付费改写。API Key 仅从本地 `.env` 进入适配层，不写入代码、manifest 或日志。

跨 B01–B08 及既有 blended B01/B02 必须检查：

- `task_id` 唯一；
- Unicode/空白/标点规范化后的 Question 唯一；
- Blueprint 语义签名唯一；
- 不逐字复用 benchmark Question；
- 不将同一约束组合的纯同义改写算作新 QA。

冲突槽位使用由原 seed、槽位和补位序号派生的新 seed 重建 witness，不直接删除导致缺量。

## 完整证据、路线和官方 exporter

计划中的相邻地点必须形成完整连续链：

- 去程到达站/机场 → 当天首个本地地点；
- 同日各本地地点之间；
- 酒店 → 次日首个本地地点；
- 最后一个本地地点 → 返程车站/机场。

城际候选公开稳定的起终点 anchor。所有上述边界均通过真实 `get_route` 获得证据，后一个活动
引用 `route_from_previous_id`；相同地点连续活动不强制虚构路线。跨午夜城际交通使用显式日偏移
和绝对分钟比较，不再把 `end_time < start_time` 一律判为非法。

确定性 exporter 从 EvidenceBundle 展开官方字段。每批同时保存官方格式计划，并运行固定 vendor
版本的 schema 和以下八类只读 parity wrapper：

1. 活动实体 grounded；
2. 城际交通正确；
3. 景点正确；
4. 酒店正确；
5. 餐厅和用餐 commonsense 正确；
6. 市内交通正确；
7. 时间连续性正确；
8. 空间连续性正确。

wrapper 用于发现与官方实现的漂移，不进入模型上下文，也不在运行时复制一套平行 Reward。

## 统一 Reward v2 与 SFT 接纳

旧 Reward v1 的必须项与需要保留的 commonsense 合并为 `travelweaver-reward-v2`，不再作为
两套相互矛盾的最终判断。硬检查归入五个可解释组：

| 组 | 内容 |
|---|---|
| protocol/structure | trip 对齐、日程结构、合法终止 |
| evidence/grounding | 候选和完整路线证据 |
| spatiotemporal/commonsense | 时间、空间、营业时间、去重、餐型规则 |
| task constraints | TaskSpec 中所有题面硬约束 |
| quantity/cost | 人数、票/房数量和成本核算 |

SFT 是严格门槛：五组全部通过、`reward_valid=true`、`all_hard_pass=true`、Reward=1.0 才能接纳。
RL 使用同一检查事实产生有限分级信号：不可评测或无计划为 -1；硬失败根据通过的组数给
少量分层；全部硬通过后才进入正奖励区间。粒度停留在五个组，不给每条细规则单独塑形，避免
模型通过刷局部项获得高分。

官方餐型规则固定为早餐 06:00–09:00、午餐 11:00–14:00、晚餐 17:00–20:00；同日同餐型
最多一次，普通餐厅和景点全程不得重复。酒店早餐允许以住宿候选为证据并计 0 元。

`submit_plan` 的 schema 合法调用即为最终答案。内部校验失败时 episode 立即以失败终止，不能
在同一 episode 二次修改后重新提交；schema 本身不合法仍按普通 invalid action 处理。

## 8:1:1 程序化策略轨迹

每批按任务类型确定性分层后严格分为：

| family | 每批 | 总计 |
|---|---:|---:|
| `efficient_success` | 400 | 3200 |
| `loop_recovery` | 50 | 400 |
| `evidence_ready_submit` | 50 | 400 |

每题只进入一个 family。所有 observation 都从 reset 后真实执行得到，使用 delta 响应；不得从
witness 手工拼 observation。

### 工具语义与覆盖策略

模型侧工具协议为 `travelweaver-tools-v4-agent`，共 17 个工具。相同返回契约的附近搜索继续
合并为一个带 `category` 的类型化工具；语义不同的“目录发现 → 搜索 → 详情/营业核验”保持拆分，
包括景点类别、餐厅菜系、按推荐菜搜索、酒店特色、详情和营业时间查询。JSON Schema 已承担参数
发现职责，因此不恢复官方任意 `*_keys`/lambda 入口。

`list_candidates` 放在证据阶段性完备后的复查点；`remove_candidate` 只用于真实的纠错支线：先查询
并检查一个非 witness 备选，保存、列出比较后再删除，最终计划仍只引用 witness。分页、附近检索、
目录查询和核验动作都必须来自模拟器真实结果，不为刷频次手工拼 observation。每批审计要求 17 个
工具全部出现、全部具有至少一个监督目标，且每个工具至少调用批量规模的 10%（500 条时为 50 次）。

### efficient_success

只执行完整计划真正需要的搜索、翻页、保存、路线和一次提交。禁止重复搜索、未使用候选、未使用
路线和无意义翻页。每个工具调用前都必须存在非空、与当前可见状态和
同回合工具决策一致的 `assistant.content`。自然语言和 tool call 使用同一个 true mask 一起监督。
说明长度由决策复杂度决定，可以是一至数句，不强制压缩成固定短句。

### loop_recovery

在一个无状态搜索动作上原样重复 1–3 次，环境返回真实结果。注入循环的完整 assistant 回合保留
为上下文但 mask 为 false。所有正常正确动作都带可见决策说明；注入循环也带一个与重复动作一致
但整体 masked 的错误决策说明。循环后的第一个正确动作带可见反思，例如“同一页结果已经核对过，
无需再次查询；现在继续翻页定位西湖”或“已经定位到西湖，接下来保存候选”。反思必须与当时
是否已看到目标候选一致，并与同回合正确 tool call 一起监督。表达从多个确定性模板中选择并填入
真实实体，再由 DeepSeek 润色，避免单句模板化。
注入循环的说明位于重复工具调用之前，必须使用“我再执行一次相同的查询”一类当前决策表达，
不得写成“我重复执行了查询”等已经完成当前动作的时态。

### evidence_ready_submit

必要查询、保存和路线前缀全部真实执行，每个回合均带完整可见决策说明，但整个 assistant 回合的
mask 为 false。环境首次具备全部证据后，最终
assistant 回合动态列出实际已有的交通、景点、路线、住宿或餐饮证据，并调用 `submit_plan`；
这是唯一监督目标。

## 完整 ReAct 决策说明生成与润色

每条成功程序化轨迹执行以下独立协议：

```text
已确定并可回放的 tool action
  → 基于 action 前可见状态生成 template_rationale
  → 每条轨迹一次 DeepSeek 结构化润色
  → 逐轮语义与时序校验
  → 单轮 template fallback
  → assistant.content + assistant.tool_calls
```

模板不是少量固定句子的简单轮换，而是依据本轮决策动态填入：当前仍缺的证据、真实城市和实体、
候选用途、路线起终点、交通方式、循环状态以及提交前已经具备的证据类型。搜索前只能表达“准备
查询/核实”，不能提前声称已经看到结果；保存前只能引用上一轮已经展示的候选；路线说明只能引用
当前已保存地点；任何说明都不能泄露后续 observation、oracle witness、隐藏 TaskSpec 或 Reward。

DeepSeek 使用 thinking disabled、required function call 和每条轨迹一次请求，把带 `step_index` 的
全部模板草稿作为独立条目润色。返回必须保持回合数量、索引和顺序，且只能返回每轮 rationale，
不得返回或修改工具名、arguments 或计划。第一批最多 500 次调用，默认 256 并发并使用逐 task
原子 checkpoint 恢复；扩展批次沿用相同协议。

逐轮 validator 至少检查：非空和长度、protected literals、阿拉伯数字不新增、工具意图一致、
搜索前不声称已有结果、恢复点明确停止重复、提交点明确提交，以及不得把其他回合的受保护实体
搬到当前回合。某轮失败只退回该轮确定性模板；不得重新规划或改变动作。audit 同时保存
`template_rationale`、`polished_rationale`、validation errors、fallback outcome、模型和 token usage。

## SFT v5 与审计

`travelweaver-sft-v5` 的 `action_selective` 允许合法动作作为 masked context，同时要求显式
`assistant_loss_mask`、最后正确动作必为 true、所有动作均可回放。可见反思和 tool call 使用同一
message 级 mask。供应商私有 `reasoning_content` 永远删除。

每条 sidecar 至少记录 family、Question 批次、Blueprint 签名、每回合 tool/mask/reason、可见
反思、循环工具/位置/次数、首次证据完备位置、动作数、终止原因、Reward 和五组结果。Reward、
TaskSpec、oracle witness 和 submit 后验证明细只在轨迹/audit 保存，不进入模型 messages。

## B01 验收与扩量门槛

B01 必须同时满足：

- 500 个 task_id、规范化 Question 和 Blueprint 签名均唯一，且不复用 benchmark 原句；
- 类型/Scenario/偏好精确达到 B01 配额；
- 500 个 witness 均 Reward=1.0、五组全通过；
- 官方 exporter schema 500/500，八类 commonsense parity 500/500；
- 轨迹 family 严格为 400/50/50，均正常 `plan_submitted` 且确定性回放 Reward=1.0；
- 高效轨迹无冗余，循环动作全部 mask 且恢复动作监督，立即提交只有最终回合监督；
- 所有 assistant 工具回合均有非空可见决策说明，且逐轮通过 rationale validator 或记录模板回退；
- 高效轨迹的全部“决策说明 + tool call”受监督；循环和立即提交严格保持既定 selective mask；
- 输出 Question、动作数、字符/token 长度、fallback/warning、DAV/ATT/DDR、去重、Reward 和
  官方 parity 汇总，并提供三类完整样例；
- user message 只有自然语言 Question，不含 JSON、task type、episode ID、Blueprint ID。

人工审计通过后才使用 seeds `20260822`–`20260828` 生成其余七批。

## B01 实施结果（2026-08-11）

B01 当前离线候选版本为 `natural-r4 / programmatic-v12-grounded-r15-language-final`。它复用同一批
500 个已验收 witness，只重渲染 surface 并从 reset 开始重新执行 24,372 个有效动作；重建
action-selective ReAct SFT 时再次执行全部动作并映射 9,284 个 episode-local cursor，0 个 invalid
action。family 严格为 `400 / 50 / 50`，Reward、全部硬约束、官方 schema 和 commonsense 均为
`500/500`。

为保证模型看到的 user content 是真人自然语言，V1.1 的方括号 provenance metadata 现只保留在
Blueprint/audit，不再出现在题面或 polisher request。B01 的 500 条题面仍全部唯一，`[` 前缀、JSON
花括号、`constraint_id` 等结构化泄漏均为 0；有出行背景的题目改为在正文中自然表达（例如“这次是
独自出行，……”）。此表层重渲染没有调用 DeepSeek，也没有改变 Blueprint、witness、TaskSpec
硬约束或轨迹策略。仅 3 条无额外约束的短题面收到 `query_shorter_than_30_characters` 的非语义 warning。

策略采用可见证据因果图：目录结果必须驱动下一次筛选；名称、菜品、ID、城际时间窗和删除候选的
价格比较均只能来自当时的 user/工具上下文。审计中八项 grounding check 均通过，包含
`search_nearby` 三种类别分布（景点 29、餐厅 63、酒店 14）和 90 条有清单价格依据的
`remove_candidate`。17 个工具均出现且在至少 50 条监督样本中出现；最低调用数为
`search_restaurants_by_food=86`，`remove_candidate=90`。500 条完整工具序列均不重复。

可见决策说明随后已按每条轨迹一次、256 并发的 DeepSeek 结构化调用完成润色。r17 最终版在
24,372 回合中接受 18,669 条润色（76.60%），其余 5,703 条因严格 validator 自动回退模板；
不改变任何 action、参数、mask、工具 observation 或 final plan。DeepSeek 的价格比较同义表达
（如“标价”）由审计识别，缺少明确价格比较的 14 条 `remove_candidate` 说明则本地回退原模板，
无需再次调用 API。r17 的 Reward、官方检查和全部 causal grounding 继续为 500/500。
它不是通过为每个工具添加硬编码分支来堆叠句子，而是让三个高频、过去最模板化的决策显式引用已见
状态：目录调用说明当前已保存候选；翻页说明已查页数与本轮真实筛选范围；提交说明引用已保存的
去程、本地地点、返程等候选以及实际路线数。目录说明为 328/1,400 unique（23.43%，单句最高重复
11），普通翻页为 4,012/6,446（62.24%，最高重复 7），提交说明为 500/500 unique。附近翻页仍
限定为可见锚点、半径和页码；候选清单仅引用已保存候选、用途或比较对象；路线说明显式包含当前
出发时刻。

循环注入仍为 mask=false，但将要重复的搜索和恢复点均使用真实城市/查询范围。重复 1–3 次时采用
确定性轮换措辞，恢复回合明确停止重复并转向下一页或保存候选。长翻页搜索同样以“每个查询流的
稳定起始偏移 + 顺序轮换”选择句式，保证连续页面不复用同一结构，同时不同 task 保留表达差异。
V8–V12 只变更 assistant 的可见自然语言：对 r11、r12、r13、r14、r15 的 500 条动作序列均将
episode-local cursor 归一化后比较，最终动作、计划和 Reward 均为 0 差异。全量 causal audit 继续
确认不泄露 hidden witness 名称。

只有在用户明确同意后，才可以把这一固定轨迹版本按每条一次的方式交给 DeepSeek 做纯语言润色。

模型可见 observation 已升级为 `travelweaver-model-tool-response-v3`：它保留 ID、名称、价格、
类别/菜品提示、营业时间、候选用途、分页 cursor 和路线 ID 等决策证据，移除完整环境仍会保留的
坐标、源 metadata、嵌套候选快照和完整路线分段。此变更将模型响应字符量从 51,725,454 降至
27,371,864（约 47%），不改变环境结果、action、mask、计划或 Reward。

Qwen3.5 官方模板转换后的 r17 序列长度为 9,682–59,000 tokens，p50 为 28,852、p90 为 40,549、
p95 为 43,884、p99 为 51,387；均低于新设定的 65,536 训练上限、两卡 Ulysses sequence-parallel
token budget（32,768 × 2）和模型 262,144-token 上限。loss token 总数为 2,232,174（占全部
14,593,972 tokens 的 15.29%）；最长样本的监督 token 数为 9,363，thinking scaffold 全部 mask。
扩量阶段以 60K 为预警线、65,536 为拒绝线；任何超长样本均隔离并重建 witness/轨迹，绝不截掉
早期证据。

## B02：1,000 条中间扩量（已授权，待执行）

使用新 seed `20260822`、相同 `chinatravel_official_hybrid_v2` profile 和 B01 目录作为跨批去重
输入，生成 1,000 个新 Blueprint、Question 与 witness。B02 中每个 Question 仍只对应一条轨迹，
family 精确为 `800 / 100 / 100`；动作构造、17 工具覆盖、官方 exporter、DeepSeek ReAct 决策说明
润色、逐轮回退、Reward/official/causal audit 和 Qwen 65K 预检均复用 B01 r17 协议。B02 合格后，
B01+B02 共同转换为 1,500 条 action-selective SFT；不混入既有约 875 条 DeepSeek rollout。
训练 launcher 的默认 `MAX_LENGTH` 已同步为 65,536，`MAX_TOKEN_LEN_PER_GPU` 为 32,768（两卡
SP=2）；dry-run 已验证 SFT Parquet hash、FSDP/sequence parallel、fused AdamW、fused linear
cross-entropy 和 `truncation=error`，未启动 `torchrun` 或占用 GPU。
