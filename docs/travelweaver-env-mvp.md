# TravelWeaverEnv Agent 闭环 MVP

本文档记录第一阶段已经落地的查询环境、数据准备流程和版本边界。更完整的候选管理、行程提交、Reward、SFT 和 GRPO 设计见[项目设计记录](project-design.md)。

## 1. 目标与边界

MVP 将 ChinaTravel 的固定数据快照包装为一个可回放的进程内 Python 环境：

```text
reset(task) -> 查询证据 -> 管理候选 -> submit_plan / finish_without_plan -> terminal
```

环境已经包含完整的 13 工具状态闭环、确定性 Demo Agent、OpenAI-compatible
function-calling rollout、通用 TravelTaskSpec、终局 TravelReward 和严格 RFT 过滤。
普通查询动作 Reward 为 `0.0`，只有提交、退出或截断产生终局 Reward。SFT 数据加工与
veRL/GRPO 训练循环暂不实现，项目不引入 MCP。

版本基线：

- Python `3.10.19`，项目约束 `>=3.10,<3.11`；
- ChinaTravel submodule commit `456b60a28ce0626875a968666c07094e3c90520e`；
- 后续训练使用独立 Linux/CUDA 环境和 `verl==0.8.0`；
- 环境、Observation、工具协议分别为 `travelweaver-environment-v0.3`、
  `travelweaver-observation-v3`、`travelweaver-tools-v2-agent`；
- TaskSpec、Reward 和轨迹协议分别为 `travelweaver-task-spec-v1`、
  `travelweaver-reward-v1`、`travelweaver-trajectory-v6`；模型可见工具返回协议为
  `travelweaver-model-tool-response-v1`。

## 2. 安装与数据准备

```bash
git submodule update --init --recursive
uv sync --dev
uv run travelweaver bootstrap chinatravel
uv run travelweaver import-tasks --split benchmark
```

`bootstrap chinatravel` 会先检查 ChinaTravel 官方数据库；缺失时尝试下载官方 Google Drive 文件夹。也可以先从官方 Google Drive 或 NJU Drive 手动下载压缩包，再执行：

```bash
uv run travelweaver bootstrap chinatravel --archive /absolute/path/to/database.zip
```

如果已手工解压到 `vendor/ChinaTravel/chinatravel/environment/database`，可以只验证：

```bash
uv run travelweaver bootstrap chinatravel --verify-only
```

校验清单包含：10 份景点、10 份餐厅、10 份酒店、10 份 POI、90 份有向火车数据、航班数据和地铁数据，共 132 个必要文件。

任务导入器固定使用 Hugging Face `LAMDA-NeSy/ChinaTravel` revision
`802b18d9844a4a9927bb5750edd155e918c20913`。`benchmark` 会校验并合并 Easy 300、
Medium 150、Human 154 和 Preference base 50，共 654 行，然后生成：

- `data/tasks/benchmark.public.jsonl`：Agent 可见的中文查询和基础元数据；
- `data/tasks/benchmark.oracle.jsonl`：环境内部使用的约束字符串；
- 四个原始 split 各自对应的 public/oracle 文件。

Preference base 50 的原始 UID 与 Easy 重复；合并快照将重复行命名为
`preference_base50:<source_uid>` 并保留 `source_uid`，因此 654 行均可独立寻址。

导入器使用 `ast.literal_eval` 将 `hard_logic_py` 解析为字符串列表，但绝不执行约束源码。

## 3. Python API

```python
from travelweaver.data import JsonlTaskStore
from travelweaver.env import ChinaTravelBackend, TravelWeaverEnv

backend = ChinaTravelBackend()
tasks = JsonlTaskStore.default(split="easy")
env = TravelWeaverEnv(backend, tasks)

observation = env.reset(seed=0)
result = env.step({
    "tool": "search_attractions",
    "arguments": {"city": observation.task["target_city"]},
})
env.close()
```

公开方法：

- `reset(task_id=None, seed=None) -> Observation`
- `step({"tool": ..., "arguments": ...}) -> StepResult`
- `tool_schemas() -> list[dict]`
- `close()`

工具参数按 JSON Schema 严格校验，多余字段也会被拒绝。连续 3 个非法动作终止 episode；50 个有效动作后以 truncated 结束。`reset` 清空可见实体、cursor、候选集和错误计数。Observation 会返回候选摘要，但不暴露隐藏 oracle。

## 4. Agent 工具

| 工具 | 用途 |
|---|---|
| `search_attractions` | 搜索景点及开放时间、票价、建议游玩时长 |
| `search_restaurants` | 搜索餐厅、菜系、推荐菜、营业时间和人均价格 |
| `search_hotels` | 搜索酒店特色、床位数和价格 |
| `search_intercity_transport` | 查询火车或航班快照 |
| `search_nearby` | 在已见地点附近按类别搜索 |
| `inspect_place` | 查看已见地点的完整规范化证据 |
| `get_route` | 查询两个同城已见地点的步行、出租车或地铁路线 |
| `next_page` | 使用一次性 cursor 获取下一页 |
| `save_candidate` | 保存已见地点或城际交通及其快照证据 |
| `list_candidates` | 查看当前 episode 的候选集 |
| `remove_candidate` | 删除候选 |
| `submit_plan` | 提交引用已保存候选的结构化多日行程并终止 |
| `finish_without_plan` | 无法形成方案时说明原因并终止 |

每页默认 10 条。Cursor 与 episode、查询和偏移绑定，使用一次后失效，不能跨轨迹复用。`inspect_place`、`search_nearby`、`get_route` 只接受本 episode 已经展示过的 `place_id`；`save_candidate` 也只能保存本 episode 已见的地点或交通 ID。

`get_route` 返回 episode 内登记的稳定 `route_id`。同一天相邻同城地点活动必须在后一
活动的 `route_from_previous_id` 中引用该路线；城际活动时间必须与车次/航班证据一致。
住宿活动显式提交 `rooms` 和 `room_type`，门票、车票和出租车数量由环境按人数推导。

`submit_plan` 验证任务元数据、天数、活动时序、候选和路线引用、营业时间、往返城际
交通、景点、多日住宿及房间容量。环境重新计算所有单价、数量和总费用，并生成
`PlanSnapshot` 与 `EvidenceBundle` 后调用确定性 Reward。

稳定 ID 规则：景点和餐厅优先使用城市、类型和上游数字 ID；酒店使用城市、类型和规范化名称哈希；火车和航班使用独立 `transport_id`；完整路线内容使用 `route_id`。

## 5. TaskSpec、Reward 与 RFT

每次 `reset` 都在 Actor 不可见的边界解析并冻结 `TravelTaskSpec`。ChinaTravel oracle
通过安全 AST 适配器转换，654 行当前全部覆盖且绝不执行源码；自由文本任务可通过
`LLMTaskSpecCompiler` 的单一 function call 生成候选 Spec，再以确定性校验失败关闭。

所有检查返回 `pass | fail | unverifiable`。设硬检查通过率为 `H`、软约束满足率为
`S`：无可评估提交为 `-1`；硬约束失败为 `-1+H`；全部硬约束通过为
`0.5+0.5×S`；基础设施不可验证为 `0` 且 `reward_valid=false`。严格 RFT 过滤只接纳
正常提交、Reward 有效且全部硬约束通过的轨迹。完整协议见
[Reward 与离线评估协议](reward-and-evaluation.md)。

## 6. 验证

无需真实数据库即可运行完整单元测试：

```bash
uv run pytest
uv run ruff check .
```

安装真实数据库和任务快照后运行：

```bash
uv run travelweaver smoke-env --task-id e20241028160248698752
uv run travelweaver run-agent --task-id e20241028160248698752
```

`smoke-env` 检查真实查询，`run-agent` 使用只调用公开工具的确定性策略跑通查询、候选管理和计划提交，并输出终局方案。测试套件覆盖全部 13 个工具、稳定 ID、过滤排序、无结果、分页、候选隔离、提交校验、两种终止路径、跨 episode 越权、连续非法动作、动作上限、任务 public/oracle 隔离和数据库清单。

使用官方 DeepSeek API 运行一条真实模型轨迹：

```bash
uv sync --extra api --dev
cp .env.example .env
# 在 .env 中填写 DEEPSEEK_API_KEY
uv run travelweaver rollout-api --task-id e20241028160248698752
```

默认模型为 `deepseek-v4-flash`，完整轨迹按 `travelweaver-trajectory-v6` 写入
`data/trajectories/deepseek-v4-flash.jsonl`。轨迹以标准 OpenAI-compatible
`messages + tools` 作为可重放对话，同时独立保存已执行 `steps`、审计事件、终止状态
和 token usage，但不会包含 API key。每个 assistant 回合只执行一个工具；若模型返回
并行调用，规范化消息只保留第一个，其余调用只进入审计事件，避免出现未响应的
`tool_call_id`。

V6 对 malformed 或解析后非 object 的 function arguments 使用可恢复历史：原始坏字符串写入
`tool_argument_normalization` 审计事件和 step 的 `raw_tool_call`，实际环境动作与下一轮模型历史
统一规范为 `{}`。该动作仍会得到 invalid tool response，模型可以在下一轮修正，同时避免把坏
JSON 原样发回 OpenAI-compatible 服务而触发 HTTP 400。

模型可见的中间 tool message 默认使用 `--tool-response-mode delta`：只包含本轮
`tool_result`、可选 `error`、`valid_action` 和 `remaining_steps`，不会重复初始 task、全部
candidates 或累计 visible ID。完整 `StepResult` 始终保存在 `steps[].result`，确定性回放、
Reward 和审计不依赖精简消息。`--tool-response-mode snapshot` 保留 v3 的旧模型上下文行为。

代码层由通用 `OpenAICompatibleConfig`、`OpenAICompatibleChatClient` 和
`ToolCallingAgent` 完成协议处理，`DeepSeekConfig` 是当前官方 API 的配置入口。
DeepSeek 的 `thinking` 是可选供应商字段，不属于环境工具
协议；环境与其他 OpenAI-compatible 模型不依赖它。

离线主观评价使用 `OfflineTravelJudge`：它只能看到公开 query、压缩后的工具摘要、
最终计划和证据摘要，看不到 Reward、oracle 或原始数据库记录。确定性结果、Judge
Rubric 和轨迹指标作为三个独立面板输出，不合成总分。

## 7. 许可与训练环境

ChinaTravel 上游源码以 Git submodule 引用，不复制到 TravelWeaver 包内。官方旅行数据不会由本项目重新分发；ChinaTravel 数据集卡标注为 CC BY-NC-SA 4.0，商业使用或再分发前需要单独检查授权。

`verl==0.8.0`、PyTorch、vLLM、Ray 和 CUDA 不属于查询 MVP 的本地依赖。后续训练镜像仍使用 Python 3.10，但应维护单独的 Linux/CUDA 锁文件，避免把 GPU 依赖带入 macOS/CPU 环境。
