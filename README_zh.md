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

```mermaid
flowchart LR
  User((用户))

  subgraph HiveCore 定制层
    AA[AssistantAgent (AA)<br/>用户级秘书<br/>长期记忆 + Prompt + 私有知识库]
    Planner[团队规划器 / 交付评审器]
    Project[(项目上下文<br/>项目记忆 + 知识库 + MsgHub)]
  end

  subgraph AgentScope 原生能力
    AgentLib[[Agent 库与角色模版]]
    Agents{{项目内 Agents}}
  end

  User -->|需求 / 反馈| AA
  AA -->|澄清规格 & 交付标准| Planner
  Planner -->|写入计划| Project
  Planner -->|组建团队| AgentLib
  AgentLib -->|实例化角色| Agents
  Agents -->|通过 MsgHub 广播| Project
  Project -->|轮次交付快照| Planner
  Planner -->|≥ 90%?<br/>自动质检| AA
  Planner -->|< 90%| Project
  AA -->|最终交付| User
```

HiveCore 添加的 AA、规划/验收、项目上下文与 MsgHub 等组件构建在 AgentScope 原生 Agent 抽象与消息通道之上，既保持 AS 兼容，又实现持久化秘书与项目级协同。

---

## 🔁 用户流程

1. **AssistantAgent 面向用户长期存在**：独立于项目之外，保存用户记忆、专属 Prompt 与私有知识库，保证需求不会从零开始。  
2. **创建项目时初始化共享上下文**：每个项目都拥有独立的记忆库、知识库与 `MsgHub` 广播池，后续加入的 Agent 也能即时了解进度。  
3. **AA 与用户循环打磨需求**：AA 与用户共同完善需求与验收标准，经确认后由规划器从 AgentScope Agent 库挑选合适角色，组队后开始执行。  
4. **按轮次交付并校验**：每一批任务形成“轮次交付”，AA 根据约定标准（默认 ≥90%）验收；若未达标，规划器会重新制订计划，并可替换/新增 Agent 进入下一轮。  
5. **交付结果即所得**：若用户要网站则完成部署并返回 URL；若要视频/素材则直接提供文件，AA 作为唯一接口确保“所求即所得”。  

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
