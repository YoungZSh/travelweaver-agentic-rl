# TravelWeaver 仓库协作指南

## 项目目标

TravelWeaver 是面向长程旅行规划 Agent 的确定性、可回放环境。当前重点是先稳定环境、工具协议、任务规格与 Reward，再在此基础上进行闭源模型 rollout、RFT/SFT 数据构造和 GRPO 训练。

## 开发环境

- 使用 Python 3.10 和 `uv` 管理环境，不要直接向系统 Python 安装依赖。
- 初始化仓库：`git submodule update --init --recursive`
- 安装依赖：`uv sync --dev`
- API rollout 额外安装：`uv sync --extra api --dev`
- 准备 ChinaTravel 数据：`uv run travelweaver bootstrap chinatravel`
- 导入 benchmark 任务：`uv run travelweaver import-tasks --split benchmark`
- 环境冒烟测试：`uv run travelweaver smoke-env`
- 运行测试：`uv run pytest`
- 静态检查：`uv run ruff check .`

## 代码结构

- `src/travelweaver/env/`：episode 状态机、13 个公开工具、稳定 ID 和 ChinaTravel backend。
- `src/travelweaver/data/`：数据库准备、校验以及任务快照导入。
- `src/travelweaver/tasks/`：与 benchmark 来源解耦的 `TravelTaskSpec`、编译和解析。
- `src/travelweaver/rollout/`：模型客户端、Agent 循环和可回放轨迹。
- `src/travelweaver/reward/`：确定性约束验证、Reward 和严格 RFT 过滤。
- `src/travelweaver/evaluation/`：与训练 Reward 分离的盲测 LLM Judge。
- `src/travelweaver/cli/`：命令行入口。
- `tests/`：按照上述模块镜像组织测试。
- `docs/`：架构、环境协议及 Reward/评估设计文档。
- `vendor/ChinaTravel/`：固定版本的上游子模块，除非任务明确要求，否则不要直接修改。

## 核心设计约束

1. 环境必须保持确定性和可回放。排序、分页、ID、路线及 Reward 不得依赖未固定的随机状态。
2. `tool_schemas.py` 是面向模型的统一 Function Calling 协议，不是 ChinaTravel 原始 API 的逐字段复制。上游字段转换应集中在 backend 层。
3. Agent 只能引用本 episode 已展示的实体；计划只能引用已保存候选和已查询路线。不要绕过 evidence contract 直接读取 oracle 数据。
4. ChinaTravel benchmark 只是首个任务来源。新任务约束应进入通用 `TravelTaskSpec`，不要把 benchmark 特有逻辑硬编码进 Reward。
5. 训练 Reward 必须是确定性、可审计的；LLM Judge 仅用于离线评估，不参与训练 Reward，也不要把两者合并成单一分数。
6. 环境层保持轻量，不要在当前 Python 环境中引入 CUDA、veRL 或训练框架依赖；训练栈使用独立环境。
7. 默认继续使用进程内 Function Calling，不引入 MCP，除非项目范围明确改变。

## 修改规范

- Python 代码使用 4 空格缩进、完整类型标注和 100 字符行宽。
- 优先使用已有 dataclass、模型和错误类型，不要新增重复的数据结构。
- 新增或修改公开工具时，同时更新：
  - `src/travelweaver/env/tool_schemas.py`
  - `src/travelweaver/env/environment.py` 中的执行与状态逻辑
  - 对应测试和协议文档
  - 必要的工具、环境或轨迹协议版本
- 修改 TaskSpec、Reward、轨迹或评估输出时，检查序列化兼容性、版本常量和已有快照。
- 外部模型调用保持 provider-neutral；供应商特有参数应留在配置适配层，不要污染通用 Agent 循环。
- 不提交 `.env`、API Key、模型权重、下载缓存、数据库副本或 `data/trajectories/` 生成轨迹。
- 不用真实 API 完成普通单元测试；使用 fake client 或 mock，保证测试离线、快速且稳定。

## 验证要求

- 小改动至少运行相关测试文件。
- 环境协议、Reward、TaskSpec 或 rollout 改动应运行完整的 `uv run pytest` 和 `uv run ruff check .`。
- 涉及真实 ChinaTravel backend 时，再运行 `uv run travelweaver smoke-env`；只有用户明确要求时才调用付费外部 API。
- Bug 修复应补充能够复现问题的回归测试。

## 文档与提交

- 设计决策优先写入 `docs/`，README 保持为安装和入口说明。
- 注释解释约束或原因，不重复代码表面行为。
- 提交应按逻辑拆分，使用简洁的 Conventional Commit 风格，如 `feat:`、`fix:`、`test:`、`docs:`、`chore:`。
- 不覆盖或回退与当前任务无关的用户改动。
