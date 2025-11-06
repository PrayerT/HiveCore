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
| - history buffer + injected resolvers       |
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

[Planned / partially implemented]
- Project context persistence (ProjectPool + MemoryPool + ResourceLibrary)
- MsgHub-based broadcast for active agents
- Artifact-specific delivery adapters (deployments, media packaging)
```

HiveCore layers (AA, planner/evaluator, project context, MsgHub) sit on top of AgentScope’s core agent abstractions (agent library, messaging, tool APIs). The runtime and sandbox execution still rely on the AgentScope runtime; HiveCore orchestrates policies, memory, and delivery logic without replacing the underlying AS environment. Components marked “planned” exist only as stubs today and still need real storage or orchestration wiring.

---

## 🛠 Implementation Status

**Available Today**
- AA selection + orchestration scaffolding (selector, orchestrator, task graph builder, KPI tracker).
- SystemRegistry + UserProfile bookkeeping to bind AA instances to users (process memory only).
- KPI tracking hooks for cost/time deltas (baseline vs observed) surfaced through AA responses.

**Work in Progress**
- Persistent AA memory/prompt/knowledge-base store (current implementation is in-memory only).
- Project-level memory/knowledge base plus MsgHub orchestration wired into ExecutionLoop.
- Round-based delivery gating with configurable ≥90% acceptance and automatic replanning loops.
- Deeper integration with AgentScope runtime sandbox: resource policies, execution metadata, audit hooks.
- Artifact-specific delivery adapters (auto deployments, media/file packaging) initiated from AA.

**Not Started**
- Planner observability dashboards + plugin APIs for third-party extensions.
- Multi-project portfolio view with cross-project knowledge sync.

---

## 🔁 User Flow

Target flow (current prototype implements only parts of steps 1 & 3):

1. **AssistantAgent (AA) exists per user** — today AA maintains an in-memory history and routes to the orchestrator; persistent memory/prompt/KB remain TODO.  
2. **Project creation seeds shared context** — planned: per-project memory store + knowledge base + `MsgHub` to keep late-joining agents informed. (Stubs exist in `ProjectPool`/`MemoryPool` but are not wired in yet.)  
3. **AA ↔ user refinement loop** — implemented at the planner level: AA resolves requirements via injected resolvers, then the orchestrator selects agents from the AgentScope library.  
4. **Round-based work & gating** — planned: aggregate task outputs per round and replan if acceptance < 90%. (ExecutionLoop currently marks tasks complete without QA.)  
5. **Delivery matches the artifact** — planned: AA triggers deployment/file-packaging adapters so the delivered artifact matches user intent.

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
