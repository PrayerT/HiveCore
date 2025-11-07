# -*- coding: utf-8 -*-
"""
CLI 演示：使用 SiliconFlow OpenAI 兼容接口走完 HiveCore 用户流程。

步骤：
1. 读取 `~/agentscope/.env` 获取 SILICONFLOW_* 配置。
2. 调用大模型与用户多轮对话，直到生成结构化需求与验收标准。
3. 将需求映射到 HiveCore 的 `AASystemAgent` + `ExecutionLoop`，自动组建团队并运行多轮交付。
4. 输出每轮广播、最终交付链接以及 AA 的验收回复。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from pathlib import Path

from agentscope.aa import AgentCapabilities, AgentProfile, RoleRequirement, StaticScore
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.ones import (
    AAMemoryStore,
    AASystemAgent,
    AcceptanceCriteria,
    ArtifactDeliveryManager,
    CollaborationLayer,
    ExecutionLoop,
    ExperienceLayer,
    InMemoryMsgHub,
    IntentLayer,
    KPITracker,
    MediaPackageAdapter,
    MemoryPool,
    OpenQuestionTracker,
    ProjectDescriptor,
    ProjectPool,
    ResourceLibrary,
    SlaLayer,
    SupervisionLayer,
    SystemMission,
    SystemProfile,
    SystemRegistry,
    TaskGraphBuilder,
    UserProfile,
    WebDeployAdapter,
    AssistantOrchestrator,
)


def load_siliconflow_env() -> dict[str, str]:
    """Load SiliconFlow credentials from ~/agentscope/.env (if present)."""
    env_path = Path.home() / "agentscope" / ".env"
    if env_path.exists():
        with env_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    required = ("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL", "SILICONFLOW_MODEL")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError(f"缺少环境变量: {', '.join(missing)}")
    return {
        "api_key": os.environ["SILICONFLOW_API_KEY"],
        "base_url": os.environ["SILICONFLOW_BASE_URL"],
        "model": os.environ["SILICONFLOW_MODEL"],
    }


async def gather_spec(
    llm: OpenAIChatModel,
    initial_requirement: str,
    scripted_answers: list[str] | None = None,
) -> dict:
    """Use the LLM to iteratively collect requirement spec."""
    system_prompt = textwrap.dedent(
        """
        你是 HiveCore 的 AssistantAgent。与用户多轮对话，逐步完善项目需求。
        当你需要更多信息时，请提出一个具体问题。
        当你觉得信息足够时，请输出:
        READY::{"project_name": "...","summary": "...","requirements":[{"id":"task-1","role":"Planner","skills":["planning"],"tools":["notion"],"description":"..."}],"acceptance":{"quality":0.85},"artifact_type":"web"}
        JSON 中不要出现换行或注释。
        """
    ).strip()
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_requirement},
    ]
    print("\nAA: 已记录初始需求，我将提出问题来澄清细节。")
    while True:
        response = await llm(conversation, stream=False, temperature=0.2)
        text = "".join(
            block.get("text", "")
            for block in response.content
            if block.get("type") == "text"
        ).strip()
        conversation.append({"role": "assistant", "content": text})
        if text.startswith("READY::"):
            payload = text.split("READY::", 1)[1].strip()
            spec = json.loads(payload)
            print("AA: 需求已完善，开始进入执行阶段。")
            return spec
        print(f"AA: {text}")
        if scripted_answers:
            user_reply = scripted_answers.pop(0)
            print(f"用户(自动): {user_reply}")
        else:
            user_reply = input("用户: ").strip()
            if not user_reply:
                user_reply = "先按你的理解继续。"
        conversation.append({"role": "user", "content": user_reply})


def build_requirements(spec: dict) -> dict[str, RoleRequirement]:
    requirements: dict[str, RoleRequirement] = {}
    for item in spec.get("requirements", []):
        node_id = item.get("id") or f"task-{len(requirements) + 1}"
        role = item.get("role", "Dev")
        requirements[node_id] = RoleRequirement(
            role=role,
            skills=set(item.get("skills", [])),
            tools=set(item.get("tools", [])),
            domains=set(item.get("domains", [])),
            languages=set(item.get("languages", [])),
        )
    if not requirements:
        requirements["task-1"] = RoleRequirement(
            role="Dev",
            skills={"python"},
            tools={"docker"},
        )
    return requirements


def seed_agent_library(orchestrator: AssistantOrchestrator, requirements: dict[str, RoleRequirement]) -> None:
    """Register a simple agent library covering the requested roles."""
    base_profiles = {
        "Planner": AgentProfile(
            agent_id="planner-pro",
            name="PlannerPro",
            role="Planner",
            static_score=StaticScore(performance=0.9, brand=0.8, recognition=0.85),
            capabilities=AgentCapabilities(
                skills={"planning", "spec"},
                tools={"notion", "slack"},
                domains={"product"},
            ),
        ),
        "Designer": AgentProfile(
            agent_id="designer-pro",
            name="UXSpark",
            role="Designer",
            static_score=StaticScore(performance=0.88, brand=0.75, recognition=0.82),
            capabilities=AgentCapabilities(
                skills={"design", "figma"},
                tools={"figma"},
                domains={"web"},
            ),
        ),
        "Dev": AgentProfile(
            agent_id="dev-fleet",
            name="DevFleet",
            role="Dev",
            static_score=StaticScore(performance=0.92, brand=0.77, recognition=0.8),
            capabilities=AgentCapabilities(
                skills={"python", "frontend", "backend"},
                tools={"docker", "git"},
                domains={"web", "api"},
            ),
        ),
        "QA": AgentProfile(
            agent_id="qa-guardian",
            name="QAGuardian",
            role="QA",
            static_score=StaticScore(performance=0.85, brand=0.7, recognition=0.78),
            capabilities=AgentCapabilities(
                skills={"testing", "automation"},
                tools={"pytest"},
                domains={"web"},
            ),
        ),
    }
    fallback = base_profiles["Dev"]

    for requirement in requirements.values():
        role = requirement.role
        profile = base_profiles.get(role)
        if profile is None:
            profile = AgentProfile(
                agent_id=f"{role.lower()}-auto",
                name=f"{role}Auto",
                role=role,
                static_score=fallback.static_score,
                capabilities=AgentCapabilities(
                    skills=set(fallback.capabilities.skills),
                    tools=set(fallback.capabilities.tools),
                    domains=set(fallback.capabilities.domains),
                    languages=set(fallback.capabilities.languages),
                    regions=set(fallback.capabilities.regions),
                    compliance_tags=set(fallback.capabilities.compliance_tags),
                    certifications=set(fallback.capabilities.certifications),
                ),
            )
        orchestrator.register_candidates(role, [profile])


async def run_cli(
    initial_requirement: str,
    scripted_answers: list[str] | None = None,
) -> None:
    creds = load_siliconflow_env()
    llm = OpenAIChatModel(
        model_name=creds["model"],
        api_key=creds["api_key"],
        stream=False,
        client_args={"base_url": creds["base_url"]},
    )

    spec = await gather_spec(llm, initial_requirement, scripted_answers)
    requirements = build_requirements(spec)
    acceptance = AcceptanceCriteria(
        description=spec.get("summary", "HiveCore 验收"),
        metrics={"quality": spec.get("acceptance", {}).get("quality", 0.85)},
    )
    artifact_type = spec.get("artifact_type", "web")

    registry = SystemRegistry()
    orchestrator = AssistantOrchestrator(system_registry=registry)
    seed_agent_library(orchestrator, requirements)

    project_pool = ProjectPool()
    project_pool.register(
        ProjectDescriptor(
            project_id="proj-cli",
            name=spec.get("project_name", "CLI Project"),
            metadata={"source": "cli"},
        ),
    )
    memory_pool = MemoryPool()
    resource_library = ResourceLibrary()
    kpi_tracker = KPITracker(target_reduction=0.85)
    broadcast_sink = InMemoryMsgHub()
    delivery_manager = ArtifactDeliveryManager([WebDeployAdapter(), MediaPackageAdapter()])

    executor = ExecutionLoop(
        project_pool=project_pool,
        memory_pool=memory_pool,
        resource_library=resource_library,
        orchestrator=orchestrator,
        task_graph_builder=TaskGraphBuilder(),
        kpi_tracker=kpi_tracker,
        msg_hub_factory=lambda _project_id: broadcast_sink,
        delivery_manager=delivery_manager,
        max_rounds=3,
    )

    memory_store = AAMemoryStore()
    user_profile = UserProfile(user_id="cli-user")
    registry.register_user(user_profile, "aa-cli")

    requirement_resolver = lambda _: requirements
    acceptance_resolver = lambda _: acceptance

    metrics_resolver = lambda _: (120.0, 260.0, 120.0, 200.0)

    agent = AASystemAgent(
        name="HiveCore-AA",
        user_id="cli-user",
        orchestrator=orchestrator,
        execution_loop=executor,
        requirement_resolver=requirement_resolver,
        acceptance_resolver=acceptance_resolver,
        metrics_resolver=metrics_resolver,
        project_resolver=lambda _: "proj-cli",
        user_profile=user_profile,
        memory_store=memory_store,
    )

    user_brief = f"{spec.get('project_name','项目')}，需求：{spec.get('summary','')}。"
    response = await agent.reply(
        Msg(
            name="user",
            role="user",
            content=user_brief,
        ),
    )

    print("\n========== AA 最终回复 ==========")
    print(response.get_text_content())
    if response.metadata.get("deliverable"):
        print("\n交付结果：", response.metadata["deliverable"])

    print("\n========== 广播快照 ==========")
    for update in broadcast_sink.updates:
        print(f"[Round {update.round_index}] {update.summary}")

    print("\n========== 项目记忆摘要 ==========")
    for entry in memory_pool.query_by_tag("project:proj-cli"):
        print(entry.content)

    print("\n========== KPI 记录 ==========")
    for record in kpi_tracker.records:
        print(
            f"成本优化: {record.cost_reduction:.0%}, 时长优化: {record.time_reduction:.0%}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="HiveCore 用户流程 CLI 演示")
    parser.add_argument(
        "--requirement",
        "-r",
        dest="requirement",
        help="初始需求说明（可选，默认交互输入）",
    )
    parser.add_argument(
        "--auto-answers",
        dest="auto_answers",
        help="使用 '||' 分隔的预设回答序列，便于自动化测试",
    )
    args = parser.parse_args()
    requirement = args.requirement or input("请输入你的项目需求：").strip()
    scripted = None
    if args.auto_answers:
        scripted = [item.strip() for item in args.auto_answers.split("||")]
    asyncio.run(run_cli(requirement, scripted))


if __name__ == "__main__":
    main()
