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
      +-------+        +-------------------------------------------+
      | User  |<-----> | AssistantAgent (AA)                       |
      +-------+        | - persistent memory / prompt / KB         |
                       +--------------------+----------------------+
                                            |
                                            v
                       +-------------------------------------------+
                       | Team Planner & Delivery Evaluator         |
                       +--------------------+----------------------+
                                            |
            +-------------------------------+------------------------------+
            |                                                              |
            v                                                              v
+-----------------------------+                         +-------------------------------+
| Project Context             |                         | AgentScope Agent Library      |
| (memory, KB, MsgHub)        |<----------------------->| & Role Templates (native AS)  |
+-----------------------------+   instantiate agents    +---------------+---------------+
            |                                                              |
            v                                                              v
    +-------------------+                                +-------------------------------+
    | Project Agents    |<----------- MsgHub ----------->| Other Project Agents          |
    +-------------------+                                +-------------------------------+
            |
            v
    Round delivery snapshot --> Planner --> AA auto-QA (>=90%?). If fail, replan; else AA ships result to user.
```

HiveCore layers (AA, planner/evaluator, project context, MsgHub) sit on top of AgentScope’s core agent abstractions (agent library, messaging, tool APIs), so we stay AS-compatible while adding persistent assistants and project-scoped collaboration.

---

## 🔁 User Flow

1. **AssistantAgent (AA) exists per user** — it is outside any single project and stores long-term memory, a dedicated prompt, and a personal knowledge base.  
2. **Project creation seeds shared context** — every project gets its own memory store, knowledge base, and `MsgHub` broadcast pool so new agents always know the latest progress.  
3. **AA ↔ user refinement loop** — AA co-edits requirements and delivery criteria with the user; once approved, the planner pulls suitable roles from the AgentScope agent library to form the execution team.  
4. **Round-based work & gating** — after each batch, outputs are aggregated into a “round delivery.” AA checks against the acceptance bar (default ≥90%); if not met, the planner regenerates the plan and can swap or add agents for the next loop.  
5. **Delivery matches the artifact** — websites get deployed with URLs returned, media assets are provided as files; AA stays accountable for “what you ask is what you receive.”

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
