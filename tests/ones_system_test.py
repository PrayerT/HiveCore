# -*- coding: utf-8 -*-
"""Integration style tests for the One·s modules."""
from agentscope.aa import AgentCapabilities, AgentProfile, StaticScore, RoleRequirement
from agentscope.ones import (
    AcceptanceCriteria,
    AssistantOrchestrator,
    DeliveryStack,
    ExecutionLoop,
    ExperienceLayer,
    IntentLayer,
    IntentRequest,
    KPITracker,
    MemoryPool,
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
    build_summary,
    OpenQuestionTracker,
    CollaborationLayer,
)


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


def test_end_to_end_cycle() -> None:
    mission = SystemMission(
        name="One·s",
        value_proposition="所求即所得",
        goal_statement="Build self-driven AI delivery",
    )
    profile = SystemProfile(
        project_name="One·s",
        mission=mission,
        aa_description="AA orchestrates intent + acceptance",
    )
    registry = SystemRegistry()
    orchestrator = AssistantOrchestrator(system_registry=registry)
    orchestrator.register_candidates("Dev", [_mock_agent("dev-1")])

    intent = IntentRequest(
        user_id="u-1",
        utterance="需要部署一个RAG服务",
        project_id="proj-1",
        role_requirements={
            "task-1": RoleRequirement(role="Dev", skills={"python", "rag"}, tools={"docker"}),
        },
    )
    acceptance = AcceptanceCriteria(
        description="Quality >= 0.8",
        metrics={"quality": 0.8},
    )

    project_pool = ProjectPool()
    project_pool.register(ProjectDescriptor(project_id="proj-1", name="RAG"))
    memory_pool = MemoryPool()
    resource_library = ResourceLibrary()
    delivery_stack = DeliveryStack(
        intent=IntentLayer(),
        sla=SlaLayer(),
        supervision=SupervisionLayer(),
        collaboration=CollaborationLayer(),
        experience=ExperienceLayer(),
    )
    assert delivery_stack.execute("hello") == "hello"
    kpi_tracker = KPITracker(target_reduction=0.5)
    executor = ExecutionLoop(
        project_pool=project_pool,
        memory_pool=memory_pool,
        resource_library=resource_library,
        orchestrator=orchestrator,
        task_graph_builder=TaskGraphBuilder(),
        kpi_tracker=kpi_tracker,
    )

    registry.register_user(UserProfile(user_id="u-1"), "aa-u-1")
    report = executor.run_cycle(
        intent,
        acceptance,
        baseline_cost=100,
        observed_cost=20,
        baseline_time=100,
        observed_time=30,
    )

    assert report.accepted is True
    assert all(status == "completed" for status in report.task_status.values())
    assert "task-1" in report.plan.decision

    question_tracker = OpenQuestionTracker()
    summary = build_summary(profile, kpi_tracker, question_tracker)
    assert summary.overview.startswith("Project One·s")
    assert summary.unresolved_questions == 0


def test_round_persistence_and_replan(tmp_path) -> None:
    registry = SystemRegistry()
    orchestrator = AssistantOrchestrator(system_registry=registry)
    orchestrator.register_candidates(
        "Dev",
        [
            AgentProfile(
                agent_id="dev-round",
                name="dev-round",
                role="Dev",
                static_score=StaticScore(performance=0.8, brand=0.7, recognition=0.75),
                capabilities=AgentCapabilities(skills={"python"}, tools={"docker"}),
            ),
        ],
    )

    intent = IntentRequest(
        user_id="u-round",
        utterance="需要多轮次交付",
        role_requirements={
            "task-1": RoleRequirement(role="Dev", skills={"python"}, tools={"docker"}),
        },
    )
    acceptance = AcceptanceCriteria(description="Quality", metrics={"quality": 0.8})

    project_pool = ProjectPool()
    memory_pool = MemoryPool()
    resource_library = ResourceLibrary()
    executor = ExecutionLoop(
        project_pool=project_pool,
        memory_pool=memory_pool,
        resource_library=resource_library,
        orchestrator=orchestrator,
        task_graph_builder=TaskGraphBuilder(),
        kpi_tracker=KPITracker(target_reduction=0.9),
        max_rounds=2,
    )

    report = executor.run_cycle(
        intent,
        acceptance,
        baseline_cost=100,
        observed_cost=400,
        baseline_time=80,
        observed_time=200,
    )

    assert report.accepted is True
    project_id = report.project_id
    assert project_id is not None
    round_entries = memory_pool.query_by_tag(f"project:{project_id}")
    assert len(round_entries) == 2
