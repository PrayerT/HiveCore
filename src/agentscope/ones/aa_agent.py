# -*- coding: utf-8 -*-
"""Concrete AA agent that bridges user utterances and the One·s stack."""
from __future__ import annotations

from typing import Callable, Sequence

from ..agent import AgentBase
from ..message import Msg, TextBlock
from ..aa import RoleRequirement
from .execution import ExecutionLoop
from .intent import AcceptanceCriteria, AssistantOrchestrator, IntentRequest
from ._system import UserProfile
from .storage import AAMemoryStore

RequirementResolver = Callable[[str], dict[str, RoleRequirement]]
AcceptanceResolver = Callable[[str], AcceptanceCriteria]
MetricsResolver = Callable[[str], tuple[float, float, float, float]]
ProjectResolver = Callable[[str], str | None]


def _default_metrics(_: str) -> tuple[float, float, float, float]:
    return (100.0, 30.0, 100.0, 30.0)


class AASystemAgent(AgentBase):
    """Top-level AA agent that owns requirement routing and acceptance."""

    def __init__(
        self,
        *,
        name: str,
        user_id: str,
        orchestrator: AssistantOrchestrator,
        execution_loop: ExecutionLoop,
        requirement_resolver: RequirementResolver,
        acceptance_resolver: AcceptanceResolver,
        metrics_resolver: MetricsResolver | None = None,
        project_resolver: ProjectResolver | None = None,
        user_profile: UserProfile | None = None,
        memory_store: AAMemoryStore | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.user_id = user_id
        self.orchestrator = orchestrator
        self.execution_loop = execution_loop
        self.requirement_resolver = requirement_resolver
        self.acceptance_resolver = acceptance_resolver
        self.metrics_resolver = metrics_resolver or _default_metrics
        self.project_resolver = project_resolver or (lambda _: None)
        self._history: list[Msg] = []
        self.memory_store = memory_store
        if self.memory_store is not None:
            record = self.memory_store.load(user_id)
            if initial_prompt:
                record.prompt = initial_prompt
                self.memory_store.save(record)

        if user_profile is not None:
            self.orchestrator.route_user(user_profile)

    async def observe(self, msg: Msg | list[Msg] | None) -> None:
        if msg is None:
            return
        if isinstance(msg, list):
            self._history.extend(msg)
        else:
            self._history.append(msg)
        if self.memory_store is not None and msg is not None:
            messages = msg if isinstance(msg, list) else [msg]
            for item in messages:
                text = item.get_text_content()
                if text:
                    self.memory_store.append(self.user_id, item.role or "unknown", text)

    def _ensure_messages(
        self,
        msg: Msg | list[Msg] | None,
    ) -> Sequence[Msg]:
        if msg is None:
            return self._history[-1:] if self._history else []
        if isinstance(msg, list):
            return msg
        return [msg]

    @staticmethod
    def _extract_text(message: Msg) -> str:
        text = message.get_text_content()
        if text is None:
            raise ValueError("AA agent expects textual content")
        return text

    def _build_intent(self, utterance: str, role_requirements: dict[str, RoleRequirement]) -> IntentRequest:
        project_id = self.project_resolver(utterance)
        return IntentRequest(
            user_id=self.user_id,
            utterance=utterance,
            project_id=project_id,
            role_requirements=role_requirements,
        )

    async def reply(self, msg: Msg | list[Msg] | None = None, **kwargs) -> Msg:
        if msg is not None:
            await self.observe(msg)
        messages = self._ensure_messages(msg)
        if not messages:
            raise ValueError("AA agent requires at least one message to reply")
        utterance = self._extract_text(messages[-1])

        role_requirements = self.requirement_resolver(utterance)
        acceptance = self.acceptance_resolver(utterance)
        intent = self._build_intent(utterance, role_requirements)
        baseline_cost, observed_cost, baseline_time, observed_time = self.metrics_resolver(utterance)

        report = self.execution_loop.run_cycle(
            intent=intent,
            acceptance=acceptance,
            baseline_cost=baseline_cost,
            observed_cost=observed_cost,
            baseline_time=baseline_time,
            observed_time=observed_time,
        )

        lines = [
            f"项目: {report.project_id or '未指定'}",
            f"AA验收: {'通过' if report.accepted else '未通过'}",
            f"成本优化: {report.kpi.cost_reduction:.0%}",
            f"时长优化: {report.kpi.time_reduction:.0%}",
            "任务结果:",
        ]
        for node_id, status in report.task_status.items():
            agent_name = (
                report.plan.rankings[node_id].profile.name
                if node_id in report.plan.rankings
                else "未分配"
            )
            lines.append(f"- {node_id}: {status} -> {agent_name}")

        content = "\n".join(lines)
        metadata = {
            "accepted": report.accepted,
            "project_id": report.project_id,
            "task_status": report.task_status,
        }
        response = Msg(
            name=self.name,
            role="assistant",
            content=[TextBlock(type="text", text=content)],
            metadata=metadata,
        )
        if self.memory_store is not None:
            self.memory_store.append(self.user_id, response.role, content)
        await self.observe(response)
        return response
