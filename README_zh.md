# 🐝 HiveCore

> **扩展 AgentScope 生态——一个用于补齐 AS（AgentScope）在运行时、编排与记忆层缺口的模块化核心框架。**

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-early--alpha-yellow)
![Build](https://img.shields.io/badge/build-passing-success)

---

## 🌐 概览

**HiveCore** 是面向 **AgentScope (AS)** 的开源扩展框架，旨在通过强化运行时管理、沙箱编排与记忆协同来补齐 AS 的生态缺失。

它为 AS Agents 提供统一的运行、观测与扩展基础设施。  
名称 “HiveCore” 的含义是：*众多 Agent，共享一座蜂巢——协同进化的群体智能。*

---

## 🚀 愿景

> *让孤立的 Agent 进化为可编排的智能群体。*

AgentScope 已具备 Agent 定义与通信的基础能力，而 **HiveCore** 专注在 AS 仍待完善的能力上：
- 🧩 **动态运行时管理**：为每个 Agent 提供隔离沙箱  
- 🔄 **多 Agent 协同编排**：基于角色的调度与协作  
- 🧠 **记忆与上下文持久化**：借鉴 Memu/Mem0 的混合记忆层  
- 🧱 **可扩展插件系统**：允许社区模块挂载到运行时  

长期目标：**让 AgentScope 演进为完整、自我迭代的 Agent 运行时标准。**

---

## 🧬 架构

```
                +--------+
                |  用户  |
                +--------+
                    |
                    v
+---------------------------------------------+
| AssistantAgent (AA)                         |
| - 持久化记忆存储 + 外部 resolver            |
+------------------+--------------------------+
                    | 绑定 / 路由用户
                    v
+---------------------------------------------+
| SystemRegistry & UserProfile（进程内）      |
+------------------+--------------------------+
                    |
                    v
+---------------------------------------------+
| Assistant Orchestrator（选择器 + 规划器）    |
+------------------+--------------------------+
                    | 排名 / 计划
                    v
+---------------------------------------------+      +--------------------------------+
| TaskGraph Builder + Execution Loop          |<---->| AgentScope Agent 库 / Runtime  |
+------------------+--------------------------+      | （AS 原生运行时 / 沙箱）       |
                    | 任务状态                        +--------------------------------+
                    v
+---------------------------------------------+
| KPI Tracker & 交付汇报                       |
+---------------------------------------------+

[下一步增强]
- 将广播接口与实时 MsgHub 参与者桥接
- 交付适配器接入真实部署 / 存储基础设施
```

HiveCore 添加的 AA、规划/验收、项目上下文与 MsgHub 等组件构建在 AgentScope 原生 Agent 抽象与消息通道之上。运行时与沙箱依旧采用 AgentScope 自带的 runtime，HiveCore 负责编排策略、记忆与交付逻辑，而非替换底层 AS 执行环境。图中标注 “计划中” 的组件目前仅有桩代码，仍需补充真实的存储或调度逻辑。

---

## 🛠 实现状态

**当前可用**
- AA 选型 + 编排骨架（selector、orchestrator、task graph builder、KPI tracker）。
- `AAMemoryStore` 持久化保存用户 Prompt、知识条目与完整对话记录，可在多轮会话中复用。
- SystemRegistry + UserProfile 映射并可基于记忆库自动装配用户绑定关系。
- ExecutionLoop 自动注册项目、写入 `MemoryPool` 的轮次摘要，供新增 Agent 即时读取。
- 轮次摘要可通过 `MsgHubBroadcaster` 接口推送给订阅方（面板/在线 Agent）。
- ≥90% 验收阈值的轮次交付：每轮记录 KPI + 任务状态，未达标会自动重规划并重试。
- Web/媒体交付适配器（`WebDeployAdapter`, `MediaPackageAdapter`）在验收通过时生成可访问的 URL / URI，并由 AA 回传给用户。
- KPI 记录能力，可在 AA 回复中输出成本/时长对比。

**进行中**
- 将广播接口与运行时 MsgHub 参与者打通，实现真实 Agent 间广播。
- 与 AgentScope runtime 沙箱更深度的联动：资源策略、执行元数据、审计挂钩。
- 交付适配器对接真实部署管道或对象存储，而非示例 URI。

**尚未开始**
- 规划器可观测性面板 + 插件 API，方便第三方扩展。
- 多项目视图与跨项目知识同步。

## 🧪 端到端 CLI 演示

- 脚本：`scripts/full_user_flow_cli.py`
- 环境：在 `~/agentscope/.env` 配置 `SILICONFLOW_API_KEY`、`SILICONFLOW_BASE_URL`、`SILICONFLOW_MODEL`。
- 流程：
  1. AA 调用真实 LLM 与用户多轮对话，直至产出包含详尽验收标准的 `READY::JSON`。
  2. Planner / Designer / Developer / QA 四个 Agent 逐个调用 LLM 协同工作，每轮都会输出真实文案、实现计划与评审结果，不含任何写死数据。
  3. QA 对每条验收标准给出 JSON 判定，按 “通过条数 / 总条数” 计算通过率，不达标则携带反馈进入下一轮。
- 示例：
  ```bash
  python scripts/full_user_flow_cli.py \
    -r "我要一个展示新品发布的单页网站，包含报名表单" \
    --auto-answers "新品是AI智能手表||主要面向科技媒体及核心用户||报名需要姓名、邮箱、媒体名称"
  ```

---

## 🔁 用户流程

以下为目标流程（现已部分实现：持久记忆 + 轮次验收 + 交付适配器）：

1. **AssistantAgent 面向用户长期存在**：AA 通过 `AAMemoryStore` 将 Prompt、知识、完整对话落盘，进程重启也能延续上下文。  
2. **创建项目时初始化共享上下文**：ExecutionLoop 自动注册项目，并把每轮摘要写入 `MemoryPool`，同时通过广播接口推给订阅者（即将对接实时 MsgHub）。  
3. **AA 与用户循环打磨需求**：已在规划层实现，AA 通过 resolver 解析需求，orchestrator 从 AgentScope Agent 库评估并组建团队，同时可利用记忆库中的偏好。  
4. **按轮次交付并校验**：已实现，ExecutionLoop 记录每轮 KPI 与任务状态，按 ≥90% 阈值验收，未通过则自动 replanning + 再执行。  
5. **交付结果即所得**：Web/媒体交付适配器已经能给出可访问的 URL / URI；下一步会接入真实部署与包管理。  

---

## ⚙️ 能力归属：AgentScope vs HiveCore

| 能力 | AgentScope 原生 | HiveCore 定制 |
|------|----------------|---------------|
| Agent 抽象、工具调用、基础消息通道 | ✅ | 作为底座复用 |
| AssistantAgent 人设 + 用户级记忆/Prompt/知识库 | — | ✅ |
| 项目级记忆、知识库与 MsgHub 广播池 | ⚙️ 基础数据接口 | ✅ 强化编排 |
| 团队规划器与交付评审（≥90% 阈值、返工循环） | — | ✅ |
| Agent 库与角色模版 | ✅ | ✅ 增补领域模版 |
| 运行时沙箱编排（Docker/Fargate/本地） | 能力有限 | ✅ 自研层 |
| REST/WebSocket 控制面 | ✅ | ✅ 补充端点 |

---

## 🧰 技术栈

- **Python 3.10+**
- **FastAPI** —— API 层  
- **SQLAlchemy + PostgreSQL** —— 元数据持久化  
- **Docker / Fargate** —— 沙箱运行时  
- **Mem0 / LangChain / Chroma** —— 记忆层  
- **AgentScope (AS)** —— 基础依赖  

---

## 🏗️ 安装

```bash
git clone https://github.com/yourname/hivecore.git
cd hivecore
pip install -e .
```

或即将上线的 PyPI：

```bash
pip install hivecore
```

---

## ⚡ 快速开始

```python
from hivecore import HiveRuntime
from agentscope import Agent

# 加载 Agents
agents = [Agent("coder"), Agent("reviewer")]

# 创建运行时
runtime = HiveRuntime(agents)
runtime.run("Build a weather app using React and Django")
```

---

## 🛣️ 路线图

| 版本 | 关注点 | 状态 |
|------|--------|------|
| v0.1 | 运行时 + 沙箱原型 | ✅ 进行中 |
| v0.2 | 记忆层集成 | ⏳ 规划中 |
| v0.3 | 插件 + 观察者钩子 | ⏳ 规划中 |
| v0.4 | 与 AS 全量兼容 | 🔜 |
| v1.0 | 稳定版（含社区插件） | 🚀 未来 |

---

## 🤝 参与共建

HiveCore 欢迎社区贡献！  
如果你正在扩展 AgentScope 或构建运行时工具，建议：
1. Fork 仓库  
2. 创建新分支（`feature/your-feature`）  
3. 提交 Pull Request  

我们提倡清晰的架构、强类型与最小耦合。

---

## 📜 许可协议

以 **MIT License** 发布，欢迎自由地 Fork、修改与集成。

---

## 🌍 作者与链接

**项目负责人：** [Prayer](https://github.com/prayert)  
**网站：** [django.prayert.cn](https://django.prayert.cn)  
**相关项目：** AgentScope、Memu、Mem0  

---

> “群落中的每一位 Agent 都能自治，却共享一致的方向。”
