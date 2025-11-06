# -*- coding: utf-8 -*-
"""Tests for the concrete AA system agent."""
from agentscope.aa import AgentCapabilities, AgentProfile, RoleRequirement, StaticScore
from agentscope.message import Msg
from agentscope.ones import (
    AASystemAgent,
    AcceptanceCriteria,
    AssistantOrchestrator,
    KPITracker,
    MemoryPool,
    ProjectDescriptor,
    ProjectPool,
    ResourceLibrary,
    SystemRegistry,
    TaskGraphBuilder,
    UserProfile,
    ExecutionLoop,
)
from agentscope.ones.memory import ResourceHandle
from agentscope.ones.storage import AAMemoryStore
import pytest


def _mock_agent(agent_id: str) -> AgentProfile:
    return AgentProfile(
        agent_id=agent_id,
        name=agent_id,
        role="Dev",
        static_score=StaticScore(performance=0.9, brand=0.8, recognition=0.85),
        capabilities=AgentCapabilities(
            skills={"python", "rag"},
            tools={"docker"},
            domains={"infra"},
            languages={"zh"},
            regions={"cn"},
            compliance_tags={"standard"},
            certifications={"iso"},
        ),
    )


@pytest.mark.asyncio
async def test_aa_agent_reply_generates_summary() -> None:
    registry = SystemRegistry()
    orchestrator = AssistantOrchestrator(system_registry=registry)
    orchestrator.register_candidates("Dev", [_mock_agent("dev-aa")])

    project_pool = ProjectPool()
    project_pool.register(ProjectDescriptor(project_id="proj-aa", name="AA Test"))
    memory_pool = MemoryPool()
    resource_library = ResourceLibrary()
    resource_library.register(ResourceHandle(identifier="tool-1", type="mcp", uri="http://example"))
    kpi_tracker = KPITracker(target_reduction=0.2)
    execution_loop = ExecutionLoop(
        project_pool=project_pool,
        memory_pool=memory_pool,
        resource_library=resource_library,
        orchestrator=orchestrator,
        task_graph_builder=TaskGraphBuilder(),
        kpi_tracker=kpi_tracker,
    )

    def requirement_resolver(_: str) -> dict[str, RoleRequirement]:
        return {
            "task-1": RoleRequirement(role="Dev", skills={"python"}, tools={"docker"}),
        }

    def acceptance_resolver(_: str) -> AcceptanceCriteria:
        return AcceptanceCriteria(description="QoS", metrics={"quality": 0.8})

    def metrics_resolver(_: str) -> tuple[float, float, float, float]:
        return (100.0, 20.0, 100.0, 40.0)

    def project_resolver(_: str) -> str:
        return "proj-aa"

    user_profile = UserProfile(user_id="u-aa")
    agent = AASystemAgent(
        name="AA",
        user_id="u-aa",
        orchestrator=orchestrator,
        execution_loop=execution_loop,
        requirement_resolver=requirement_resolver,
        acceptance_resolver=acceptance_resolver,
        metrics_resolver=metrics_resolver,
        project_resolver=project_resolver,
        user_profile=user_profile,
    )

    msg = Msg(name="user", role="user", content="需要一个RAG团队")
    response = await agent.reply(msg)

    assert response.role == "assistant"
    assert response.metadata["accepted"] is True
    text = response.get_text_content()
    assert text is not None and "proj-aa" in text
    assert "task-1" in text


@pytest.mark.asyncio
async def test_aa_agent_persists_conversation(tmp_path) -> None:
    registry = SystemRegistry()
    orchestrator = AssistantOrchestrator(system_registry=registry)
    orchestrator.register_candidates("Dev", [_mock_agent("dev-aa")])

    memory_store = AAMemoryStore(path=tmp_path / "aa.json")
    project_pool = ProjectPool()
    memory_pool = MemoryPool()
    resource_library = ResourceLibrary()
    execution_loop = ExecutionLoop(
        project_pool=project_pool,
        memory_pool=memory_pool,
        resource_library=resource_library,
        orchestrator=orchestrator,
        task_graph_builder=TaskGraphBuilder(),
        kpi_tracker=KPITracker(target_reduction=0.1),
    )

    agent = AASystemAgent(
        name="AA",
        user_id="u-memory",
        orchestrator=orchestrator,
        execution_loop=execution_loop,
        requirement_resolver=lambda _: {
            "task-1": RoleRequirement(role="Dev", skills={"python"})
        },
        acceptance_resolver=lambda _: AcceptanceCriteria(description="q", metrics={"quality": 0.5}),
        metrics_resolver=lambda _: (100.0, 120.0, 50.0, 60.0),
        memory_store=memory_store,
    )

    msg = Msg(name="user", role="user", content="测试持久化")
    await agent.reply(msg)

    record = memory_store.load("u-memory")
    assert record.conversation_log
    roles = {entry["role"] for entry in record.conversation_log}
    assert "user" in roles and "assistant" in roles
