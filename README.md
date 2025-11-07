# 🐝 HiveCore

> **Extending the AgentScope ecosystem — A modular core framework designed to fill the missing runtime, orchestration, and memory layers of AS (AgentScope).**

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-early--alpha-yellow)
![Build](https://img.shields.io/badge/build-passing-success)

---

## 🌐 Overview

**HiveCore** is an open-source extension framework for **AgentScope (AS)** — aiming to **complete** the AS ecosystem with enhanced runtime management, sandbox orchestration, and memory coordination.

It provides a unified infrastructure to run, observe, and extend AS Agents dynamically.  
The name “HiveCore” reflects its design principle: *many agents, one hive — a collective intelligence system.*

---

## 🚀 Vision

> *From isolated Agents to orchestrated intelligence.*

While AgentScope provides a foundation for agent definitions and communication, **HiveCore** focuses on what’s missing in AS:
- 🧩 **Dynamic Runtime Management** — isolated sandbox per Agent  
- 🔄 **Multi-Agent Orchestration** — role-based coordination and scheduling  
- 🧠 **Memory & Context Persistence** — Memu/Mem0-like hybrid memory layer  
- 🧱 **Extensible Plugin System** — allows community modules to hook into the runtime  

The long-term goal: **make AgentScope a complete, self-evolving agent runtime standard.**

---

## 🧬 Architecture

```
                +-------+
                | User  |
                +-------+
                    |
                    v
+---------------------------------------------+
| AssistantAgent (AA)                         |
| - persistent memory store + resolvers       |
+------------------+--------------------------+
                    | route / bind user
                    v
+---------------------------------------------+
| SystemRegistry & UserProfile store          |
+------------------+--------------------------+
                    |
                    v
+---------------------------------------------+
| Assistant Orchestrator (selector + planner) |
+------------------+--------------------------+
                    | rankings / plans
                    v
+---------------------------------------------+      +--------------------------------+
| TaskGraph Builder + Execution Loop          |<---->| AgentScope Agent Library / RT  |
+------------------+--------------------------+      | (native AS runtime & sandbox)  |
                    | task status                      +--------------------------------+
                    v
+---------------------------------------------+
| KPI Tracker & Delivery Reporter             |
+---------------------------------------------+

[Next enhancements]
- Bridge broadcaster interface to live MsgHub participants
- Wire artifact adapters to real infra (deployments, media packaging)
```

HiveCore layers (AA, planner/evaluator, project context, MsgHub) sit on top of AgentScope’s core agent abstractions (agent library, messaging, tool APIs). The runtime and sandbox execution still rely on the AgentScope runtime; HiveCore orchestrates policies, memory, and delivery logic without replacing the underlying AS environment. Components marked “planned” exist only as stubs today and still need real storage or orchestration wiring.

---

## 🛠 Implementation Status

**Available Today**
- AA selection + orchestration scaffolding (selector, orchestrator, task graph builder, KPI tracker).
- Persistent AA memory store (`AAMemoryStore`) that records prompts, knowledge entries, and full conversation logs per user.
- SystemRegistry + UserProfile bookkeeping to bind AA instances to users (and auto-bootstrap from the memory store).
- Project-level context snapshots via `ProjectPool` + `MemoryPool` plus per-round summaries saved for any project the AA spins up.
- Round updates automatically broadcast through the `MsgHubBroadcaster` interface so live dashboards / agents can subscribe.
- Multi-round delivery gating: ExecutionLoop evaluates ≥90% quality, persists each round, and automatically replans/retries with improved metrics.
- Artifact adapters (`WebDeployAdapter`, `MediaPackageAdapter`) produce concrete URLs/files and are attached to AA responses when acceptance succeeds.
- KPI tracking hooks for cost/time deltas (baseline vs observed) surfaced through AA responses.

**Work in Progress**
- Bridging the broadcaster interface with runtime MsgHub instances to push live agent-to-agent announcements.
- Deeper integration with the AgentScope runtime sandbox: resource policies, execution metadata, audit hooks.
- Artifact adapters with real infrastructure hooks (Docker/K8s deployers, storage services) instead of mock URIs.

**Not Started**
- Planner observability dashboards + plugin APIs for third-party extensions.
- Multi-project portfolio view with cross-project knowledge sync.

## 🧪 End-to-End CLI Demo

- Script: `scripts/full_user_flow_cli.py`
- Requirements: `~/agentscope/.env` must provide `SILICONFLOW_API_KEY`, `SILICONFLOW_BASE_URL`, `SILICONFLOW_MODEL`.
- Flow:
  1. AA 多轮澄清，每轮都会输出完整的 `requirements + acceptance_map`，并等待用户输入 `confirm/确认`（无人值守可加 `--auto-confirm`）。没有确认信号就不会进入执行阶段。
  2. 执行阶段按需求拆分：Planner / Designer / Developer / QA 分别调用 LLM 生成 Blueprint、真实交付内容（HTML/API 规格/脚本等）和验收日志，交付文件保存在 `deliverables/<需求ID>.ext`。
  3. QA 逐条返回 JSON 判定（pass/fail + reason + recommendation），统计 “通过条数 / 总条数”。若任一需求未达设定阈值（默认 95%），系统会携反馈进入下一轮。
- Example:
  ```bash
  python scripts/full_user_flow_cli.py \
    -r "我要一个展示新品发布的单页网站，包含报名表单" \
    --auto-answers "全球媒体||移动优先||confirm" \
    --auto-confirm
  ```

---

## 🔁 User Flow

Target flow (now partially implemented with persistent memory + round gating + artifact adapters):

1. **AssistantAgent (AA) exists per user** — AA now loads/saves long-term memory (prompt + knowledge + dialogue) through `AAMemoryStore`, so every conversation turn is persisted beyond process restarts.  
2. **Project creation seeds shared context** — ExecutionLoop auto-registers projects, stores per-round summaries in `MemoryPool`, and exposes them through project tags; broadcaster hooks mirror each summary to live subscribers and will bridge to runtime MsgHub instances next.  
3. **AA ↔ user refinement loop** — implemented: AA resolves requirements via injected resolvers, orchestrator selects agents from the AgentScope library, and user preferences from the memory store feed into the planner.  
4. **Round-based work & gating** — implemented: every round records KPI + task status, evaluates against the ≥90% quality bar, and triggers replanning until either accepted or max rounds reached.  
5. **Delivery matches the artifact** — implemented for web/media: once accepted, the artifact delivery manager spins up mocked deployments or packages and returns a concrete URL/URI that AA relays to the user (infra-backed adapters are the next step).

---

## ⚙️ Capabilities: AgentScope vs HiveCore

| Capability | AgentScope Native | HiveCore Customization |
|------------|------------------|------------------------|
| Agent abstractions, tool calling, base messaging | ✅ | Uses as foundation |
| AssistantAgent persona with persistent user memory/prompt/KB | — | ✅ |
| Project-level memory, knowledge base, MsgHub broadcast pool | ⚙️ basic data hooks | ✅ enriched orchestration |
| Team planner & delivery evaluator (≥90% gating, replan loop) | — | ✅ |
| Agent library & role templates | ✅ | ✅ extended with domain presets |
| Runtime sandbox orchestration (Docker/Fargate/local) | Limited | ✅ dedicated layer |
| REST/WebSocket control plane | ✅ | ✅ additional endpoints |

---

## 🧰 Tech Stack

- **Python 3.10+**
- **FastAPI** — API Layer  
- **SQLAlchemy + PostgreSQL** — Metadata persistence  
- **Docker / Fargate** — Sandbox Runtime  
- **Mem0 / LangChain / Chroma** — Memory layer  
- **AgentScope (AS)** — Base dependency  

---

## 🏗️ Installation

```bash
git clone https://github.com/yourname/hivecore.git
cd hivecore
pip install -e .
```

Or from PyPI (coming soon):

```bash
pip install hivecore
```

---

## ⚡ Quick Start

```python
from hivecore import HiveRuntime
from agentscope import Agent

# Load agents
agents = [Agent("coder"), Agent("reviewer")]

# Create runtime
runtime = HiveRuntime(agents)
runtime.run("Build a weather app using React and Django")
```

---

## 🛣️ Roadmap

| Phase | Focus | Status |
|-------|--------|--------|
| v0.1 | Runtime + Sandbox prototype | ✅ In progress |
| v0.2 | Memory layer integration | ⏳ Planned |
| v0.3 | Plugin + Observer hooks | ⏳ Planned |
| v0.4 | Full AS compatibility | 🔜 |
| v1.0 | Stable release (with community plugins) | 🚀 Future |

---

## 🤝 Contributing

HiveCore welcomes collaboration!  
If you’re extending AgentScope or building runtime tools, feel free to:
1. Fork the repo  
2. Create a new branch (`feature/your-feature`)  
3. Submit a pull request

We encourage clean architecture, strong typing, and minimal coupling.

---

## 📜 License

Released under the **MIT License** — feel free to fork, modify, and integrate.

---

## 🌍 Author & Project Links

**Project Lead:** [Prayer](https://github.com/prayert)  
**Website:** [django.prayert.cn](https://django.prayert.cn)  
**Related Projects:** AgentScope, Memu, Mem0  

---

> “A hive of agents — each autonomous, all aligned.”
