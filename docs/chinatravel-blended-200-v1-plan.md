# ChinaTravel 混合覆盖 200 题合成计划

本文档冻结 `chinatravel_blended_v1` 的原始设计计划，作为实现、生成和验收的依据。
质量复盘及后续修改应另行记录，不回写或改写本计划的初始目标。

## 配比

新增 `chinatravel_blended_v1`，全部结构变化由单一 `--seed` 派生。

| 类型 | 数量 |
| --- | ---: |
| Easy-like | 50 |
| Medium-like | 70 |
| Human-like | 50 |
| Preference-like | 20 |
| Generalization | 10 |

## Human-like

- 35 条保留方括号元数据前缀，15 条使用纯自然对话。
- 软偏好数量：无偏好 10 条、1 个 20 条、2 个 15 条、3 个 5 条。
- 额外硬约束数量：`{0: 5, 1: 10, 2: 13, 3: 12, 4: 6, 5: 3, 6: 1}`。
- Blueprint 增加受控人设上下文，如独自、情侣、朋友、亲子出行；LLM 只能使用已分配背景。
- 第一批不生成元数据冲突、错别字和事实歧义。

Human 单独使用 `human_conservative` 改写模式：

- 城市、实体名、数字、单位、时间和交通方式仍须逐字保留。
- 每个硬约束仍需返回唯一、连续且不重叠的 mention。
- 允许调整语序、拆合句子、连接词、请求语气和上下文组织。
- 放宽等值约束的强制词检查，接受符合真人表达的确定句式，例如：
  - `往返坐高铁`
  - `想去西湖`
  - `酒店订2间房`
- 上下限仍必须明确保留“不超过、以内、至少、之后”等方向，不接受“左右、大概、尽量”替代硬边界。
- 禁止增加新的数字、城市、POI、日期、年龄和约束。
- 偏好允许使用受控同义表达，但必须返回 preference mention，不能遗漏或改变方向。
- Human 使用专门的自然 canonical fallback，避免退回“硬性条件为……”式模板。

Easy、Medium、Preference 和 Generalization 继续使用现有严格模式。

## Preference-like

20 条分为：

- 14 条官方六类偏好：每类至少 2 条，剩余 2 条由总 seed 平衡分配。
  - 更多景点
  - 更少市内交通时间
  - 更短的就餐交通时间
  - 提高餐饮支出占比
  - 降低住宿支出占比
  - 靠近指定 POI
- 6 条从当前可审计扩展池中选择不同偏好，如少步行、降低总成本、轻松行程和费用比例倾向。
- 每题只有一个主要偏好，写入 `unscored_preferences`，本轮不进入训练 Reward。
- witness 从多个满足硬约束的方案中按偏好指标选择更优候选，并写入 `preference-audit.jsonl`。
- 暂不生成缺少可靠数据标签的室内外、热门/冷门偏好。

## 约束和场景

- Easy：`{1: 28, 2: 15, 3: 7}`。
- Medium：`{3: 10, 4: 18, 5: 18, 6: 14, 7: 7, 8: 3}`。
- Preference：`{1: 10, 2: 7, 3: 3}`。
- Generalization：`{1: 1, 2: 2, 3: 3, 4: 2, 5: 1, 6: 1}`。
- 天数、人数、城市与约束家族使用：
  `0.65 × ChinaTravel 固定先验 + 0.35 × 均匀覆盖先验`。
- 将内部 `difficulty` 改名为 `tightness`，仅控制预算和时间余量。
- 场景配额：正常 180、景点关闭 5、酒店无房 4、交通取消 5、价格上涨 6。

## 接口、测试与验收

- CLI 增加 `--profile chinatravel_blended_v1`。
- 输出到 `data/generated/chinatravel-blended-200-v1/`，增加 `alignment.json`、
  `preference-audit.jsonl` 和分类预览。
- 为 polisher 增加显式 `validation_profile`，不根据文本内容隐式判断模式。
- 测试 Human 保守模式接受自然确定句式，同时拒绝：
  - 数值或实体变化
  - 硬约束变成可选项
  - 上下限方向弱化
  - 新增事实
  - 丢失硬约束或偏好
- 200 条硬约束 Reward 通过率必须为 100%，且无重复或 benchmark 原句复用。
- Human 中至少 70% 不出现“硬性条件、必须满足以下要求”等模板措辞；相同开头占比不超过 10%。
- 运行完整 `pytest`、`ruff` 和 ChinaTravel smoke test 后生成：

```bash
uv run travelweaver synthesize-tasks \
  --profile chinatravel_blended_v1 \
  --count 200 \
  --seed 20260808 \
  --max-api-calls 400 \
  --output-dir data/generated/chinatravel-blended-200-v1
```

