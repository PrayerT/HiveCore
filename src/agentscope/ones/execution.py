# -*- coding: utf-8 -*-
"""Execution loop tying all sections together (III)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from .intent import (
    AcceptanceCriteria,
    AssistantOrchestrator,
    IntentRequest,
    StrategyPlan,
)
from .kpi import KPIRecord, KPITracker
from .memory import MemoryEntry, MemoryPool, ProjectDescriptor, ProjectPool, ResourceLibrary
from .task_graph import TaskGraphBuilder
import shortuuid


@dataclass
class ExecutionReport:
    project_id: str | None
    accepted: bool
    kpi: KPIRecord
    task_status: dict[str, str]
    plan: StrategyPlan


class ExecutionLoop:
    def __init__(
        self,
        *,
        project_pool: ProjectPool,
        memory_pool: MemoryPool,
        resource_library: ResourceLibrary,
        orchestrator: AssistantOrchestrator,
        task_graph_builder: TaskGraphBuilder,
        kpi_tracker: KPITracker,
        msg_hub_factory: Callable[..., object] | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.project_pool = project_pool
        self.memory_pool = memory_pool
        self.resource_library = resource_library
        self.orchestrator = orchestrator
        self.task_graph_builder = task_graph_builder
        self.kpi_tracker = kpi_tracker
        self.msg_hub_factory = msg_hub_factory
        self.max_rounds = max_rounds

    def _persist_intent(self, intent: IntentRequest) -> None:
        entry = MemoryEntry(
            key=f"intent:{intent.user_id}:{intent.project_id}",
            content=intent.utterance,
            tags={"intent", intent.user_id},
        )
        self.memory_pool.save(entry)

    def _ensure_project(self, intent: IntentRequest) -> str:
        if intent.project_id:
            if self.project_pool.get(intent.project_id) is None:
                descriptor = ProjectDescriptor(
                    project_id=intent.project_id,
                    name=f"Project {intent.project_id}",
                    metadata={"source": "aa"},
                )
                self.project_pool.register(descriptor)
            return intent.project_id
        project_id = f"proj-{shortuuid.uuid()}"
        descriptor = ProjectDescriptor(
            project_id=project_id,
            name=f"{intent.user_id}-{project_id}",
            metadata={"source": "aa"},
        )
        self.project_pool.register(descriptor)
        intent.project_id = project_id
        return project_id

    def _persist_round_summary(
        self,
        *,
        project_id: str,
        round_index: int,
        plan: StrategyPlan,
        task_status: dict[str, str],
        observed_metrics: dict[str, float],
    ) -> None:
        summary_lines = [
            f"Round {round_index}",
            f"Observed metrics: {observed_metrics}",
        ]
        for node_id, status in task_status.items():
            agent_name = (
                plan.rankings.get(node_id).profile.name
                if node_id in plan.rankings
                else "unassigned"
            )
            summary_lines.append(f"- {node_id}: {status} -> {agent_name}")
        entry = MemoryEntry(
            key=f"project:{project_id}:round:{round_index}",
            content="\n".join(summary_lines),
            tags={f"project:{project_id}", "round"},
        )
        self.memory_pool.save(entry)

    def _compute_quality_score(
        self,
        *,
        baseline_cost: float,
        observed_cost: float,
        baseline_time: float,
        observed_time: float,
        round_index: int,
    ) -> float:
        def ratio(baseline: float, observed: float) -> float:
            if observed <= 0:
                return 1.0
            if baseline <= 0:
                return 0.0
            return min(1.0, baseline / observed)

        cost_component = ratio(baseline_cost, observed_cost)
        time_component = ratio(baseline_time, observed_time)
        base_score = (cost_component + time_component) / 2
        if self.max_rounds > 1:
            progressive_bonus = ((round_index - 1) / (self.max_rounds - 1)) * 0.5
        else:
            progressive_bonus = 0.0
        return max(0.0, min(1.0, base_score + progressive_bonus))

    def _broadcast_progress(self, project_id: str, summary: str) -> None:
        if self.msg_hub_factory is None:
            return
        hub = self.msg_hub_factory(project_id=project_id)
        broadcast = getattr(hub, "broadcast", None)
        if callable(broadcast):
            broadcast(summary)

    def run_cycle(
        self,
        intent: IntentRequest,
        acceptance: AcceptanceCriteria,
        *,
        baseline_cost: float,
        observed_cost: float,
        baseline_time: float,
        observed_time: float,
    ) -> ExecutionReport:
        project_id = self._ensure_project(intent)
        self._persist_intent(intent)
        plan = self.orchestrator.plan_strategy(intent, acceptance)
        graph = self.task_graph_builder.build(
            requirements=plan.requirement_map,
            rankings=plan.rankings,
            edges=None,
        )

        accepted = False
        task_status: Dict[str, str] = {}
        observed_metrics: Dict[str, float] = {}

        for round_index in range(1, self.max_rounds + 1):
            for node_id in graph.topological_order():
                graph.mark_running(node_id)
                graph.mark_completed(node_id)

            task_status = {node.node_id: node.status.value for node in graph.nodes()}
            observed_quality = self._compute_quality_score(
                baseline_cost=baseline_cost,
                observed_cost=observed_cost,
                baseline_time=baseline_time,
                observed_time=observed_time,
                round_index=round_index,
            )
            observed_metrics = {"quality": observed_quality}
            self._persist_round_summary(
                project_id=project_id,
                round_index=round_index,
                plan=plan,
                task_status=task_status,
                observed_metrics=observed_metrics,
            )
            self._broadcast_progress(
                project_id=project_id,
                summary=f"Round {round_index} status: {task_status}",
            )

            accepted = self.orchestrator.evaluate_acceptance(plan, observed_metrics)
            if accepted:
                break

            # Re-plan for next round to reflect potential agent changes.
            plan = self.orchestrator.plan_strategy(intent, acceptance)
            graph = self.task_graph_builder.build(
                requirements=plan.requirement_map,
                rankings=plan.rankings,
                edges=None,
            )
            observed_cost *= 0.8
            observed_time *= 0.8

        kpi_record = self.kpi_tracker.record_cycle(
            baseline_cost=baseline_cost,
            observed_cost=observed_cost,
            baseline_time=baseline_time,
            observed_time=observed_time,
        )

        return ExecutionReport(
            project_id=project_id,
            accepted=accepted,
            kpi=kpi_record,
            task_status=task_status,
            plan=plan,
        )
