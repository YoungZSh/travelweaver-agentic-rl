# DSL 多样性 V1

## 目标

第一阶段只启用 `travelweaver-task-spec-v3` 已经能够完整解析和确定性验证的语义，
不改变 TaskSpec 字段含义，也不引入可执行 Python DSL。生成器版本升级为
`travelweaver-synthesis-v11`。

`chinatravel_blended_v1_1` 当前采样四种逻辑形态：

- `attraction_categories_all`：一个行程同时覆盖两个不同景点类别；
- `attraction_categories_any`：两个景点类别任选其一；
- `exclude_attraction`：排除一个具名景点；
- `allowed_innercity_modes`：通过排除未允许方式表达市内交通允许集合。

每道题最多出现一种特殊逻辑形态。它们参与主配方图的自然采样，不单独生成一批
“低频工具题”。目标是生成后审计分布，而不是在生成阶段把命中率写死：当前 500 槽位的
确定性目录约 72% 为普通合取，四种特殊形态各约 7%。多值类别只分配给至少两天的行程。

## 端到端语义

- `{"values": ["A", "B"]}` 表示同时满足 A 和 B；
- `{"any_of": [["A"], ["B"]]}` 表示满足 A 或 B；
- 负向实体约束只生成单个名称，避免 `NOT(A AND B)` 与
  `NOT A AND NOT B` 的歧义；
- 市内交通允许集合使用固定工具域 `{taxi, metro, walk}` 的补集表达，并显式要求至少
  两个市内地点，防止路线约束被架空；
- canonical renderer、polisher 校验、Reward、程序化 teacher 和 diversity audit 必须对
  同一组值使用一致语义。
- `travelweaver-zh-polisher-v7` 把组内合取、组间析取和负向允许集合视为 hard error；
  即使所有类别或交通方式文字仍然存在，也不接受把“且”改成“或”或反向改写。

## 暂缓到 TaskSpec V3 的选择

以下能力会改变规格语义或需要新的 witness/Reward 协议，本阶段只记录，不实现：

- 显式 `quantifier: exists | forall | count`；
- 带 `day_index` 的结构化 selector；
- `sum/avg/max/count/set` 聚合；
- POI 距离、路线时长、活动顺序和跨实体关系；
- 任意嵌套的 `all_of/any_of/not`。

V3 开始前需要同时设计 TaskSpec、Blueprint、Surface、Reward、快照迁移和版本兼容测试，
不能只扩 sampler。

## 性能与可恢复性

槽位准备使用独立进程并行，每个 worker 复用一个 ChinaTravel backend。目录分配的 origin
先直接尝试；不再为每个槽位预先扫描所有城市的往返交通。主进程仍是唯一产物写入者，
每个完成槽位立即写入 `records/<slot>.json` 和 `progress.jsonl`，中断后按 slot 恢复。
同一目的地最多按不同可行 origin 各尝试一次；若组合仍不可行，后续预算用于确定性的替代
目的地，而不是对同一结构重复随机搜索。替换不改变 task type、Scenario、约束配方、天数和
交通要求，并通过 `slot_replaced` 事件及 `destination_replacements` 分布审计。单槽位失败不会
中止其他 future，主进程会先持久化所有独立成功结果再汇总报告失败。
