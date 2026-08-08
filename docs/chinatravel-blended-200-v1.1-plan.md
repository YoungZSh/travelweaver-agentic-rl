# ChinaTravel 混合覆盖 200 题 V1.1 试验计划

本文档记录 `chinatravel_blended_v1_1` 相对 V1 的定向质量修正。V1 的初始配比、约束和
场景配额保持不变，参见
[V1 原始计划](chinatravel-blended-200-v1-plan.md)。V1.1 的目标不是复制 benchmark，
而是在保留确定性、可回放和扩展覆盖的同时，提高对 benchmark 自然输入分布的支持。

## 保持不变

- 总量与类型配比仍为 Easy-like 50、Medium-like 70、Human-like 50、
  Preference-like 20、Generalization 10。
- 每类硬约束数量配额不变。
- Human 元数据 35 条、纯对话 15 条，软偏好数量配额不变。
- 场景配额仍为正常 180、景点关闭 5、酒店无房 4、交通取消 5、价格上涨 6。
- 所有结构随机性只由单一 `--seed` 派生。
- 硬约束 Reward、evidence contract、唯一 mention 和 protected literal 规则不变。

## 类型内 benchmark 主体与覆盖尾部

V1.1 不再把全局天数、人数先验无条件分散到每一种任务类型，而是在类型内部保留
benchmark 主体区域并混入覆盖尾部：

- Easy-like：45/50 为 1–3 天，47/50 为 1–4 人，其余用于长行程或多人覆盖。
- Medium-like：主体为 2–4 天、1–4 人，保留少量 1/5 天和 5/6 人组合。
- Human-like：覆盖 1–5 天，以 2–3 天和 1–3 人为主体。
- Preference-like：以 1–3 天和 1–3 人为主体，保留少量长行程及多人任务。
- Generalization：固定为 4–5 天、5–6 人的 compositional tail，不再与普通类型交换回
  benchmark 主体区域。

Generalization 在本版本中明确表示组合尾部覆盖，而不是未见城市、未见 DSL 或严格的
域外 OOD。

## 自然确定表达

新增显式 `benchmark_natural` validation profile。它与 `strict` 使用相同事实与方向校验，
但允许 benchmark 常见的自然确定句式，例如：

- `往返坐火车`
- `想去西湖`
- `酒店每晚订2间房`
- `全程安排3个景点`

V1.1 中 Easy、Medium、Preference 和 Generalization 按 seed 混合
`benchmark_natural` 与 `strict`，避免训练数据过度依赖“必须、硬性要求”等显式模态词。
Human 继续使用 `human_conservative`，其 canonical fallback 也改用自然确定表达。

## Human 修正

- 方括号元数据改为 benchmark-like 事实前缀，并增加受控人设字段：
  `[当前位置上海,目标位置杭州,旅行人数2,旅行天数3,出行背景情侣出行]`。
- 元数据已经包含人设时，正文不得再次逐字复述同一人设。
- 1 天游程不分配降低住宿支出占比偏好。
- 全步行 witness 不分配少走路偏好；若路线 fallback 最终变为全步行，则重试候选。
- 固定景点数硬约束不与“更多景点”或“轻松行程”偏好组合。
- 保持第一批不生成错别字、事实歧义或元数据冲突的边界。

## Preference 修正

- 硬约束不再局限于去程、返程和市内交通，可使用总预算、去返程时间、景点数和交通
  方式等共同可验证约束。
- 预算与时间约束从整个候选 cohort 的最坏可行边界反推，保证参加审计的候选通过同一组
  硬约束。
- “更多景点”和“轻松行程”候选允许改变景点密度，不生成固定景点数硬约束。
- “少步行”候选允许比较出租车、地铁和步行路线，并按实际步行 segment 分钟数审计。
- 每题至少有两个不同的偏好指标值；指标完全打平时该候选组作废并重试。
- 每个审计候选仍须满足 `reward=1.0` 和 `all_hard_pass=true`。

## 产物与验收

除 V1 验收外，V1.1 新增：

- Human 元数据人设正文重复数必须为 0。
- 20 条 Preference audit 的指标都必须有区分度。
- Preference audit 中所有候选的硬 Reward 必须通过。
- `tasks.public.jsonl` 写入明确的 `task_type` 和 profile tag，避免下游丢失分类。
- 完整运行 `pytest`、`ruff` 和 ChinaTravel smoke test。

试验生成命令：

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1_1 \
  --count 200 \
  --seed 20260808 \
  --max-api-calls 400 \
  --output-dir data/generated/chinatravel-blended-200-v1.1
```

## 试生成结果

2026-08-08 使用上述命令完成了 200 条试生成，共调用模型 288 次。验收结果如下：

- 200/200 条硬约束 Reward 通过，任务类型配额和场景配额均准确。
- 200 条 query 全部唯一，没有复用 benchmark 原句。
- 20 条 Preference audit 均有至少两个不同的指标值；全部候选都通过同一组硬约束，
  最终候选确为指标方向上的最优项。
- 50 条 Human 中，元数据人设在正文中的逐字重复数为 0，禁用模板词出现率为 0，
  相同开头的最大占比为 4%。
- “必须”出现在 43/200 条 query 中，相比 V1 的 172/200 明显降低；V1.1 保留了比
  benchmark 更高的约束密度和组合覆盖，不以逐字拟合 benchmark 为目标。
- canonical fallback 为 65/200，高于 V1 的 39/200。fallback 已使用自然确定模板且通过
  全部语义校验，但后续大规模合成前仍应继续降低该比例，以增加表述多样性。

人工通读全部 Human 和 Preference 样本时发现 4 条步行约束使用了“坐步行/使用步行”的
不自然搭配。生成器已修正为“市内地点之间都步行/必须步行”并补充回归测试；当前输出目录
作为试验快照保留，正式训练批次应从修正后的代码重新生成。

## Surface 并发重写审计

在不改变原有 200 个 Blueprint、Scenario、witness 和 Reward 的情况下，以 256 个 worker
重新运行 surface polisher，输出到
`data/generated/chinatravel-blended-200-v1.1-repolished/`：

- 200 条全部完成，共 288 次 API 调用，canonical fallback 为 61 条；
- `polish-audit.jsonl` 共 349 个事件：139 次接受、149 次拒绝、61 次最终 fallback；
- 每个请求均保存任务输入、原始 tool response、解析后的 payload、attempt 和校验错误；
- 重新核对后，200 个 Blueprint ID、Scenario ID 和 witness snapshot 与输入目录完全一致；
- 200/200 硬 Reward 和全部 alignment checks 继续通过；
- 修正后的结果不再包含“坐步行”或“使用步行”。

149 次拒绝中，主要原因为 protected literal 逐字不匹配 46 次、等值约束关键词不匹配
41 次、实体正则误报 21 次、人设正文重复 7 次。protected literal 拒绝中有 39 次只是
`1人` 改写为“一人/一个人”等人数或天数表面变化；实体误报包括“希望酒店”“住的酒店”
等普通短语。审计结果证明当前 fallback 大部分来自规则误杀，而非真实语义漂移，后续应将
校验改为按 constraint mention 做数值归一化和类型化语义校验。

## Minimal semantic validator 复测

validator 随后拆分为 hard error 和 warning。只有城市、关键实体、归一化数值、交通方式、
上下限方向、去返程作用域以及硬约束明显弱化会拒绝；中文数字形式、短文本、粗粒度实体
正则、人设重复、偏好同义表达和可修复 mention 对齐只产生 warning。

先将上一轮 149 个 rejected payload 离线重放：146 个被新规则接受，剩余 3 个均包含
“最好……”导致硬约束弱化。之后对相同 200 个 Blueprint/witness 重新调用模型：

- 202 次 API 调用，200 次 accepted、2 次首轮 rejected、0 次 canonical fallback；
- fallback 从 strict 审计的 61/200 降为 0/200；
- 54 条 surface 带 warning，主要是 39 次 protected literal 表面变化、15 次旧实体正则
  提示和 12 次全局数字形式变化；
- 200 条 query 唯一，200/200 结构化硬 Reward 和全部适用 alignment checks 通过；
- Blueprint、Scenario 和 witness 仍与原 V1.1 完全一致。

事后使用增强后的“可选词上下文”规则复查发现 1/200 条把指定酒店写成“最好是……”，属于
允许极少量 surface 噪声的试验结果。validator 已补充上下文检测，今后的请求会拒绝这种
弱化；本轮目录保留该样本以便审计。minimal 复测产物位于
`data/generated/chinatravel-blended-200-v1.1-repolished-minimal/`。
