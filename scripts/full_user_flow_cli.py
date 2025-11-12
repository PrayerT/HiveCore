# -*- coding: utf-8 -*-
"""
HiveCore CLI: 多场景需求澄清 + 多模块交付 + 细粒度验收
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agentscope.aa import AgentCapabilities, AgentProfile, RoleRequirement, StaticScore
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel, OllamaChatModel
from agentscope.mcp import HttpStatelessClient
from agentscope.ones import (
    AASystemAgent,
    AcceptanceCriteria,
    AssistantOrchestrator,
    ArtifactDeliveryManager,
    ExecutionLoop,
    InMemoryMsgHub,
    KPITracker,
    MemoryPool,
    ProjectPool,
    ResourceHandle,
    ResourceLibrary,
    SystemRegistry,
    TaskGraphBuilder,
    UserProfile,
    WebDeployAdapter,
)
from agentscope.ones.storage import AAMemoryStore

DELIVERABLE_DIR = Path("deliverables")
MCP_SUMMARY_MAX_TOOLS = 5


@dataclass
class MCPServerConfig:
    name: str
    transport: str
    url: str


@dataclass
class RuntimeHarness:
    orchestrator: AssistantOrchestrator
    execution_loop: ExecutionLoop
    registry: SystemRegistry
    project_pool: ProjectPool
    memory_pool: MemoryPool
    resource_library: ResourceLibrary
    msg_hub: InMemoryMsgHub | None
    kpi_tracker: KPITracker
    aa_agent: AASystemAgent
    project_id: str
    resource_handles: list[ResourceHandle] = field(default_factory=list)


def parse_mcp_server(value: str) -> MCPServerConfig:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == 2:
        name, url = parts
        transport = "streamable_http"
    elif len(parts) == 3:
        name, transport, url = parts
    else:
        raise ValueError(
            "MCP server格式应为 'name,url' 或 'name,transport,url'",
        )
    if not name or not url:
        raise ValueError("MCP server 参数缺少 name 或 url")
    return MCPServerConfig(name=name, transport=transport, url=url)


def build_mcp_clients(configs: list[MCPServerConfig] | None) -> list[HttpStatelessClient]:
    clients: list[HttpStatelessClient] = []
    for cfg in configs or []:
        transport = cfg.transport.strip().lower()
        if transport not in {"streamable_http", "sse"}:
            print(f"[WARN] MCP {cfg.name} 使用未知 transport {cfg.transport}，默认 streamable_http")
            transport = "streamable_http"
        try:
            client = HttpStatelessClient(
                name=cfg.name,
                transport=transport,  # type: ignore[arg-type]
                url=cfg.url,
            )
        except Exception as exc:
            print(f"[WARN] 初始化 MCP 客户端 {cfg.name} 失败: {exc}")
            continue
        clients.append(client)
    return clients


async def summarize_mcp_clients(
    clients: list[HttpStatelessClient],
) -> tuple[str, list[ResourceHandle]]:
    lines: list[str] = []
    handles: list[ResourceHandle] = []
    for client in clients:
        try:
            tools = await client.list_tools()
        except Exception as exc:
            lines.append(f"{client.name} ({client.transport}) 工具获取失败: {exc}")
            metadata = {"error": str(exc)}
        else:
            lines.append(f"{client.name} ({client.transport}) 可用工具 {len(tools)} 个:")
            for tool in tools[:MCP_SUMMARY_MAX_TOOLS]:
                lines.append(f"  - {tool.name}: {tool.description}")
            remaining = len(tools) - MCP_SUMMARY_MAX_TOOLS
            if remaining > 0:
                lines.append(f"  - ... 其余 {remaining} 个工具省略")
            metadata = {
                "transport": client.transport,
                "tool_count": str(len(tools)),
                "tools_preview": ", ".join(tool.name for tool in tools[:MCP_SUMMARY_MAX_TOOLS]),
            }
        uri = ""
        if hasattr(client, "client_config"):
            uri = client.client_config.get("url", "")
        handle = ResourceHandle(
            identifier=f"mcp::{client.name}",
            type="mcp",
            uri=uri,
            tags={"mcp", client.transport},
            metadata=metadata,
        )
        handles.append(handle)
    summary = "\n".join(lines).strip()
    return summary, handles


def _profile(
    *,
    agent_id: str,
    role: str,
    skills: set[str],
    tools: set[str],
    domains: set[str],
) -> AgentProfile:
    return AgentProfile(
        agent_id=agent_id,
        name=agent_id.replace("_", " ").title(),
        role=role,
        static_score=StaticScore(
            performance=0.82,
            brand=0.78,
            recognition=0.75,
        ),
        capabilities=AgentCapabilities(
            skills=skills,
            tools=tools,
            domains=domains,
            languages={"zh", "en"},
            regions={"global"},
            compliance_tags={"standard"},
            certifications={"iso9001"},
        ),
    )


def default_agent_profiles() -> dict[str, list[AgentProfile]]:
    catalog = {
        "Strategy": [
            _profile(
                agent_id="strategy_chief",
                role="Strategy",
                skills={"roadmap", "ops"},
                tools={"miro", "dovetail"},
                domains={"platform"},
            ),
        ],
        "Product": [
            _profile(
                agent_id="product_lead",
                role="Product",
                skills={"mvp", "sla"},
                tools={"notion", "figjam"},
                domains={"experience"},
            ),
        ],
        "Builder": [
            _profile(
                agent_id="builder_core",
                role="Builder",
                skills={"ai", "automation"},
                tools={"python", "bash"},
                domains={"delivery"},
            ),
        ],
        "Frontend": [
            _profile(
                agent_id="frontend_crafter",
                role="Frontend",
                skills={"web", "react"},
                tools={"vite", "tailwind"},
                domains={"experience"},
            ),
        ],
        "Backend": [
            _profile(
                agent_id="backend_foundry",
                role="Backend",
                skills={"api", "infra"},
                tools={"fastapi", "docker"},
                domains={"platform"},
            ),
        ],
        "Ux": [
            _profile(
                agent_id="ux_curator",
                role="Ux",
                skills={"research", "wireframe"},
                tools={"figma"},
                domains={"experience"},
            ),
        ],
        "QA": [
            _profile(
                agent_id="qa_guardian",
                role="QA",
                skills={"test", "monitor"},
                tools={"pytest", "playwright"},
                domains={"quality"},
            ),
        ],
    }
    return catalog


def _contains_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in keywords)


def infer_role(requirement: dict[str, Any]) -> str:
    text = " ".join(
        str(requirement.get(key, "")) for key in ("title", "type", "details")
    ).lower()
    if _contains_any(text, ["策略", "roadmap", "规划"]):
        return "Strategy"
    if _contains_any(text, ["产品", "业务", "mvp", "slo", "sla"]):
        return "Product"
    if _contains_any(text, ["体验", "交互", "设计", "ux", "ui"]):
        return "Ux"
    if _contains_any(text, ["前端", "web", "页面", "h5", "组件"]):
        return "Frontend"
    if _contains_any(text, ["后端", "api", "service", "服务", "数据", "pipeline"]):
        return "Backend"
    if _contains_any(text, ["测试", "验收", "质量", "qa", "监控"]):
        return "QA"
    return "Builder"


def _infer_tags(requirement: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    text = " ".join(
        str(requirement.get(key, "")) for key in ("title", "type", "details")
    ).lower()
    skills: set[str] = set()
    tools: set[str] = set()
    domains: set[str] = set()
    if _contains_any(text, ["ai", "llm", "agent"]):
        skills.add("ai")
        tools.add("python")
        domains.add("ai")
    if _contains_any(text, ["web", "页面", "h5", "landing"]):
        skills.add("web")
        tools.add("react")
        domains.add("experience")
    if _contains_any(text, ["api", "service", "接口", "微服务"]):
        skills.add("api")
        tools.add("fastapi")
        domains.add("platform")
    if _contains_any(text, ["测试", "验收", "监控", "qa"]):
        skills.add("qa")
        tools.add("pytest")
        domains.add("quality")
    if _contains_any(text, ["数据", "分析", "etl", "pipeline"]):
        skills.add("data")
        tools.add("spark")
        domains.add("data")
    if not skills:
        skills.add("generalist")
    if not tools:
        tools.add("notion")
    if not domains:
        domains.add("delivery")
    return skills, tools, domains


def build_runtime_requirements(spec: dict[str, Any]) -> dict[str, RoleRequirement]:
    requirements: dict[str, RoleRequirement] = {}
    for req in spec.get("requirements", []):
        rid = req.get("id") or sanitize_filename(req.get("title", "R"))
        role = infer_role(req)
        skills, tools, domains = _infer_tags(req)
        requirement = RoleRequirement(
            role=role,
            skills=skills,
            tools=tools,
            domains=domains,
            languages={"zh", "en"},
            regions={"global"},
            notes=req.get("details"),
        )
        requirements[rid] = requirement
    if not requirements:
        requirements["R1"] = RoleRequirement(
            role="Builder",
            skills={"generalist"},
            tools={"notion"},
            domains={"delivery"},
        )
    return requirements


def build_runtime_acceptance(spec: dict[str, Any]) -> AcceptanceCriteria:
    config = spec.get("acceptance", {}) or {}
    target = config.get("overall_target")
    if target is None:
        collected: list[float] = []
        for mapping in spec.get("acceptance_map", []):
            for criterion in mapping.get("criteria", []):
                try:
                    value = float(criterion.get("target"))
                except (TypeError, ValueError):
                    continue
                collected.append(value)
        if collected:
            target = min(max(min(collected), 0.5), 0.99)
        else:
            target = 0.9
    target = float(target)
    return AcceptanceCriteria(
        description="HiveCore CLI 自动验收",
        metrics={"quality": target},
    )


def compute_runtime_metrics(spec: dict[str, Any]) -> tuple[float, float, float, float]:
    count = max(len(spec.get("requirements", [])), 1)
    baseline_cost = 120.0 * count
    observed_cost = baseline_cost * 0.65
    baseline_time = 80.0 * count
    observed_time = baseline_time * 0.6
    return (baseline_cost, observed_cost, baseline_time, observed_time)


def derive_project_id(spec: dict[str, Any], hint: str | None) -> str:
    base = hint or spec.get("summary") or "hivecore_project"
    return sanitize_filename(base)[:24] or "hivecore_project"


def build_runtime_harness(
    spec: dict[str, Any],
    *,
    user_id: str,
    project_hint: str | None,
    resource_handles: list[ResourceHandle],
) -> RuntimeHarness:
    registry = SystemRegistry()
    orchestrator = AssistantOrchestrator(system_registry=registry)
    for role, candidates in default_agent_profiles().items():
        orchestrator.register_candidates(role, candidates)

    project_pool = ProjectPool()
    memory_pool = MemoryPool()
    resource_library = ResourceLibrary()
    for handle in resource_handles:
        resource_library.register(handle)
    msg_hub = InMemoryMsgHub()
    kpi_tracker = KPITracker(target_reduction=0.3)
    delivery_manager = ArtifactDeliveryManager([WebDeployAdapter()])

    execution_loop = ExecutionLoop(
        project_pool=project_pool,
        memory_pool=memory_pool,
        resource_library=resource_library,
        orchestrator=orchestrator,
        task_graph_builder=TaskGraphBuilder(),
        kpi_tracker=kpi_tracker,
        msg_hub_factory=lambda _: msg_hub,
        delivery_manager=delivery_manager,
    )

    def requirement_resolver(_: str) -> dict[str, RoleRequirement]:
        return build_runtime_requirements(spec)

    def acceptance_resolver(_: str) -> AcceptanceCriteria:
        return build_runtime_acceptance(spec)

    def metrics_resolver(_: str) -> tuple[float, float, float, float]:
        return compute_runtime_metrics(spec)

    project_id = derive_project_id(spec, project_hint)

    def project_resolver(_: str) -> str:
        return project_id

    user_profile = UserProfile(user_id=user_id)
    aa_agent = AASystemAgent(
        name="Hive-AA",
        user_id=user_id,
        orchestrator=orchestrator,
        execution_loop=execution_loop,
        requirement_resolver=requirement_resolver,
        acceptance_resolver=acceptance_resolver,
        metrics_resolver=metrics_resolver,
        project_resolver=project_resolver,
        user_profile=user_profile,
        memory_store=AAMemoryStore(),
    )
    return RuntimeHarness(
        orchestrator=orchestrator,
        execution_loop=execution_loop,
        registry=registry,
        project_pool=project_pool,
        memory_pool=memory_pool,
        resource_library=resource_library,
        msg_hub=msg_hub,
        kpi_tracker=kpi_tracker,
        aa_agent=aa_agent,
        project_id=project_id,
        resource_handles=resource_handles,
    )


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def load_siliconflow_env() -> dict[str, str]:
    env_path = Path.home() / "agentscope" / ".env"
    if env_path.exists():
        with env_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    creds = {}
    for key in ("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL", "SILICONFLOW_MODEL"):
        if os.environ.get(key):
            creds[key] = os.environ[key]
    return creds


def initialize_llm(
    provider: str,
    silicon_creds: dict[str, str],
    *,
    ollama_model: str,
    ollama_host: str,
):
    have_silicon = all(key in silicon_creds for key in ("SILICONFLOW_API_KEY", "SILICONFLOW_MODEL"))
    provider = provider.lower()

    if provider == "siliconflow":
        if not have_silicon:
            raise RuntimeError("未检测到硅基流动配置，请在 ~/agentscope/.env 设置相关变量。")
        return OpenAIChatModel(
            model_name=silicon_creds["SILICONFLOW_MODEL"],
            api_key=silicon_creds["SILICONFLOW_API_KEY"],
            stream=False,
            client_args={
                "base_url": silicon_creds.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            },
        ), "siliconflow"

    if provider == "ollama":
        return (
            OllamaChatModel(
                model_name=ollama_model,
                stream=False,
                host=ollama_host,
            ),
            "ollama",
        )

    if have_silicon:
        return initialize_llm(
            "siliconflow",
            silicon_creds,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
        )
    return initialize_llm(
        "ollama",
        silicon_creds,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )


async def call_llm_raw(
    llm: Any,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
) -> str:
    kwargs = {"stream": False}
    if isinstance(llm, OllamaChatModel):
        kwargs["options"] = {"temperature": temperature}
    else:
        kwargs["temperature"] = temperature
    resp = await llm(messages, **kwargs)
    return "".join(
        block.get("text", "")
        for block in resp.content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


async def call_llm_json(
    llm: Any,
    base_messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    retries: int = 3,
) -> tuple[Any, str]:
    messages = list(base_messages)
    for attempt in range(retries):
        raw = await call_llm_raw(llm, messages, temperature=temperature)
        try:
            return parse_json_from_text(raw), raw
        except Exception:
            messages = base_messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "上述回答不是有效 JSON。请严格按照规定格式输出 JSON，不要附加额外文本。",
                },
            ]
    raise RuntimeError("LLM 未能返回合法 JSON")


def extract_json_block(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未找到 JSON 区块")
    return text[start : end + 1]


def parse_json_from_text(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return json.loads(extract_json_block(text))


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "deliverable"


def ensure_requirement_ids(spec: dict[str, Any]) -> None:
    for idx, req in enumerate(spec.get("requirements", []), start=1):
        req.setdefault("id", f"R{idx}")


def print_requirements(spec: dict[str, Any]) -> None:
    print("\n当前需求列表:")
    for req in spec.get("requirements", []):
        print(f"- {req.get('id')}: {req.get('title')} [{req.get('type')}]")
        print(f"  说明: {req.get('details')}")
    print("\n当前交付标准映射:")
    for mapping in spec.get("acceptance_map", []):
        rid = mapping.get("requirement_id")
        criteria = mapping.get("criteria", [])
        print(f"- {rid}: {len(criteria)} 条标准")


# ---------------------------------------------------------------------------
# 需求澄清循环（需显式确认）
# ---------------------------------------------------------------------------
async def collect_spec(
    llm: OpenAIChatModel,
    initial_requirement: str,
    scripted_inputs: list[str] | None = None,
    auto_confirm: bool = False,
) -> dict[str, Any]:
    conversation = [
        {
            "role": "system",
            "content": textwrap.dedent(
                """
                你是 HiveCore 的产品需求协调员。
                每轮回复必须输出 JSON:
                {
                  "summary": "...",
                  "requirements": [
                    {"id":"R1","title":"...","type":"功能/流程/...","details":"...","priority":"P1/P2等"}
                  ],
                  "acceptance_map": [
                    {"requirement_id":"R1","criteria":[{"id":"R1-C1","description":"...","target":0.95}, ...]}
                  ],
                  "questions": ["下一步需要用户澄清的问题列表"]
                }
                - 无论用户输入什么，都要复述完整 JSON。
                - 若用户说“继续”或提供更多信息，更新 JSON。
                - 当用户输入 confirm/确认 时，表示当前 JSON 可以进入执行阶段。
                """,
            ).strip(),
        },
        {"role": "user", "content": initial_requirement},
    ]
    scripted_inputs = scripted_inputs[:] if scripted_inputs else []

    while True:
        spec, raw_text = await call_llm_json(llm, conversation)
        ensure_requirement_ids(spec)
        print_requirements(spec)
        if spec.get("questions"):
            print("\nAA 缺少的信息:")
            for q in spec["questions"]:
                print(f"- {q}")

        user_input: str
        if scripted_inputs:
            user_input = scripted_inputs.pop(0)
            print(f"\n[自动输入] {user_input}")
        elif auto_confirm:
            user_input = "confirm"
            print("\n[自动确认] confirm")
        else:
            user_input = input(
                "\n请输入补充信息或输入 confirm/确认 以进入执行阶段: ",
            ).strip()

        if user_input.lower() in {"confirm", "确认"}:
            print("\n需求已确认，进入执行阶段。")
            return spec

        if not user_input:
            user_input = "请继续询问缺失信息。"

        conversation.append({"role": "assistant", "content": raw_text})
        conversation.append({"role": "user", "content": user_input})


# ---------------------------------------------------------------------------
# 验收标准生成
# ---------------------------------------------------------------------------
async def enrich_acceptance_map(llm: OpenAIChatModel, spec: dict[str, Any]) -> None:
    reqs = spec.get("requirements", [])
    cleaned_map: dict[str, dict[str, Any]] = {}
    for item in spec.get("acceptance_map", []):
        rid = item.get("requirement_id")
        if not rid:
            continue
        cleaned_map.setdefault(rid, {"requirement_id": rid, "criteria": []})
        cleaned_map[rid]["criteria"].extend(item.get("criteria", []))
    spec["acceptance_map"] = list(cleaned_map.values())
    mapping = cleaned_map

    def determine_standard_count(requirement: dict[str, Any]) -> int:
        text = " ".join(str(requirement.get(k, "")) for k in ("title", "type", "details"))
        base = 10
        if any(keyword in text.lower() for keyword in ["全站", "平台", "后台", "app", "系统", "自动化"]):
            base = 14
        return base

    for req in reqs:
        rid = req["id"]
        min_count = determine_standard_count(req)
        entry = mapping.setdefault(rid, {"requirement_id": rid, "criteria": []})
        prompt = textwrap.dedent(
            f"""
            需求:
            {json.dumps(req, ensure_ascii=False, indent=2)}

            请基于该需求输出 JSON:
            {{
              "standards": [
                {{
                  "id": "{rid}.1",
                  "title": "子标准标题",
                  "description": "对需求的具体补充说明，必须可验证",
                  "category": "内容/交互/性能/安全/数据/可观测性/国际化/运维/合规 等分类之一",
                  "checklist": [
                    {{"item": "检查项1", "method": "如何验证"}},
                    {{"item": "检查项2", "method": "如何验证"}},
                    {{"item": "检查项3", "method": "如何验证"}}
                  ],
                  "target": 0.95
                }}
              ]
            }}

            要求:
            - 使用层级编号，例如 {rid}.1, {rid}.2, {rid}.3 ...；若需求本身已有编号，可在小数位继续延伸。
            - 至少 {min_count} 条，且必须覆盖内容、交互、性能、安全、数据准确性、可观测性、国际化/多语言、权限/合规、可扩展性/运维等不同角度。
            - checklist 中至少 3 个检查项，每个包含 item + method。
            - 标准描述必须明确“如何判断通过/不通过”，避免泛泛而谈。
            """,
        )
        data, _ = await call_llm_json(
            llm,
            [
                {
                    "role": "system",
                    "content": "你是严格的交付验收官，负责补齐细粒度标准，输出合法 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
        )
        standards = data.get("standards") or data.get("criteria") or []
        entry["criteria"] = standards


def criteria_for_requirement(spec: dict[str, Any], rid: str) -> list[dict[str, Any]]:
    for item in spec.get("acceptance_map", []):
        if item.get("requirement_id") == rid:
            return item.get("criteria", [])
    return []


def resolve_artifact_type(requirement: dict[str, Any]) -> str:
    text = " ".join(
        str(requirement.get(key, ""))
        for key in ("title", "type", "details", "category")
    ).lower()
    if any(word in text for word in ["页面", "网站", "web", "landing", "portal"]):
        return "web"
    if any(word in text for word in ["api", "接口", "后台", "service", "微服务"]):
        return "api"
    if any(word in text for word in ["自动化", "脚本", "pipeline", "etl", "batch"]):
        return "script"
    return "document"


# ---------------------------------------------------------------------------
# Agent 角色
# ---------------------------------------------------------------------------
async def design_requirement(
    llm,
    requirement: dict[str, Any],
    feedback: str,
    passed_ids: set[str],
    failed_criteria: list[dict[str, Any]],
    prev_blueprint: dict[str, Any] | None,
    contextual_notes: str | None = None,
) -> dict[str, Any]:
    artifact_type = resolve_artifact_type(requirement)
    prompt = textwrap.dedent(
        f"""
        需求对象:
        {json.dumps(requirement, ensure_ascii=False, indent=2)}

        已通过的标准:
        {sorted(passed_ids) if passed_ids else "无"}

        仍需改进的标准:
        {json.dumps(failed_criteria, ensure_ascii=False, indent=2) if failed_criteria else "无"}

        上一版 Blueprint (如有):
        {json.dumps(prev_blueprint, ensure_ascii=False, indent=2) if prev_blueprint else "无"}

        之前 QA 的反馈 (如有):
        {feedback or "无"}

        请输出 Blueprint（JSON），字段包括:
        {{
          "requirement_id": "{requirement['id']}",
          "artifact_type": "{artifact_type}",
          "deliverable_pitch": "...",
          "structure": [...],
          "data_sources": [...],
          "constraints": [...],
          "recommended_stack": "...",
          "artifact_spec": {{
             "format": "html|markdown|json|python|notebook|diagram|... ",
             "sections": [{{"id":"...","title":"...","content_outline":["..."]}}]
          }}
        }}
        输出合法 JSON。
        """
    )
    if contextual_notes:
        prompt += "\n共享上下文（Runtime/MCP）:\n" + contextual_notes
    blueprint, _ = await call_llm_json(
        llm,
        [
            {"role": "system", "content": "你是资深架构/体验设计师，输出 Blueprint JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
    )
    return blueprint


async def implement_requirement(
    llm,
    requirement: dict[str, Any],
    blueprint: dict[str, Any],
    feedback: str,
    passed_ids: set[str],
    failed_criteria: list[dict[str, Any]],
    previous_artifact: str,
    contextual_notes: str | None = None,
) -> dict[str, Any]:
    artifact_type = blueprint.get("artifact_spec", {}).get("format") or resolve_artifact_type(requirement)
    prompt = textwrap.dedent(
        f"""
        需求:
        {json.dumps(requirement, ensure_ascii=False, indent=2)}

        Blueprint:
        {json.dumps(blueprint, ensure_ascii=False, indent=2)}

        已通过的标准:
        {sorted(passed_ids) if passed_ids else "无"}

        需修复的标准:
        {json.dumps(failed_criteria, ensure_ascii=False, indent=2) if failed_criteria else "无"}

        上一版交付片段:
        {previous_artifact[:1200] if previous_artifact else "无"}

        QA 反馈:
        {feedback or "无"}

        请输出 JSON:
        {{
          "summary": "...",
          "artifact_extension": "html|md|json|py|txt",
          "artifact_content": "..."
        }}
        - 如果 format=html，请生成完整可运行的 HTML/CSS/JS 单页，包含表单/交互逻辑。
        - 如果 format=json，输出带缩进的 JSON。
        - 如果 format=markdown，输出结构化文档。
        - 其他格式同理。
        """
    )
    if contextual_notes:
        prompt += "\n可引用的 Runtime / MCP 见解:\n" + contextual_notes
    impl, _ = await call_llm_json(
        llm,
        [
            {"role": "system", "content": "你是交付工程师，需要生成最终产物。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
    )
    return impl


async def qa_requirement(
    llm: OpenAIChatModel,
    requirement: dict[str, Any],
    blueprint: dict[str, Any],
    artifact_path: Path,
    criteria: list[dict[str, Any]],
    round_index: int,
) -> dict[str, Any]:
    artifact_content = artifact_path.read_text(encoding="utf-8")
    prompt = textwrap.dedent(
        f"""
        需求:
        {json.dumps(requirement, ensure_ascii=False, indent=2)}

        Blueprint:
        {json.dumps(blueprint, ensure_ascii=False, indent=2)}

        验收标准:
        {json.dumps(criteria, ensure_ascii=False, indent=2)}

        交付物 (文件 {artifact_path.name}，截断前 6000 字符):
        {artifact_content[:6000]}

        请输出 JSON:
        {{
          "round": {round_index},
          "criteria": [
            {{
              "id": "...",
              "name": "...",
              "pass": true/false,
              "reason": "...",
              "recommendation": "...",
              "checklist_review": [
                 {{"item": "原 checklist 内容", "pass": true/false, "remark": "逐项说明"}}
              ]
            }}
          ],
          "improvements": "如果未通过，需要改进什么"
        }}
        要求逐条给出理由与整改建议。
        """
    )
    qa_report, _ = await call_llm_json(
        llm,
        [
            {
                "role": "system",
                "content": "你是资深 QA，逐条对照验收标准给出 pass/fail，输出 JSON。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return qa_report


# ---------------------------------------------------------------------------
# 执行与验收
# ---------------------------------------------------------------------------
async def run_execution(
    llm: OpenAIChatModel,
    spec: dict[str, Any],
    max_rounds: int = 3,
    verbose: bool = False,
    runtime_summary: str | None = None,
    mcp_context: str | None = None,
) -> dict[str, Any]:
    requirements = spec.get("requirements", [])
    overall_target = spec.get("acceptance", {}).get("overall_target", 0.95)
    feedback_map = {req["id"]: "" for req in requirements}
    req_state = {
        req["id"]: {
            "passed": set(),
            "artifact": "",
            "summary": "",
            "feedback": "",
            "blueprint": None,
            "path": None,
        }
        for req in requirements
    }
    rounds: list[dict[str, Any]] = []
    final_paths: dict[str, Path] = {}
    notes_parts: list[str] = []
    if runtime_summary:
        notes_parts.append(f"Runtime 摘要:\n{runtime_summary}")
    if mcp_context:
        notes_parts.append(f"MCP 工具概览:\n{mcp_context}")
    contextual_notes = "\n\n".join(notes_parts)

    for round_idx in range(1, max_rounds + 1):
        print(f"\n---- 执行轮次 Round {round_idx} ----")
        round_entry = {"round": round_idx, "results": []}
        requirement_pass_flags = []

        for requirement in requirements:
            rid = requirement["id"]
            criteria = criteria_for_requirement(spec, rid)
            for idx, item in enumerate(criteria, 1):
                item.setdefault("id", f"{rid}.{idx}")

            state = req_state[rid]
            passed_ids = state["passed"]
            failed_criteria = [c for c in criteria if c.get("id") not in passed_ids]

            if not failed_criteria:
                print(f"- {rid} 已全部通过，沿用上一轮成果")
                requirement_pass_flags.append(True)
                round_entry["results"].append(
                    {
                        "requirement_id": rid,
                        "blueprint": state.get("blueprint"),
                        "implementation": {
                            "summary": state.get("summary", "上一轮产物"),
                            "path": str(state.get("path") or ""),
                        },
                        "qa": {"criteria": []},
                        "pass_ratio": 1.0,
                    },
                )
                final_paths[rid] = state.get("path") or final_paths.get(rid)
                continue

            blueprint = await design_requirement(
                llm,
                requirement,
                feedback_map[rid],
                passed_ids,
                failed_criteria,
                state.get("blueprint"),
                contextual_notes=contextual_notes or None,
            )
            print(f"\n[{rid}] Blueprint 摘要：{blueprint.get('deliverable_pitch', '')}")
            if verbose:
                print(
                    f"[{rid}] Blueprint 详情：\n"
                    f"{json.dumps(blueprint, ensure_ascii=False, indent=2)}"
                )
            impl = await implement_requirement(
                llm,
                requirement,
                blueprint,
                feedback_map[rid],
                passed_ids,
                failed_criteria,
                state.get("artifact", ""),
                contextual_notes=contextual_notes or None,
            )
            print(f"[{rid}] Developer Summary：{impl.get('summary', '')}")
            if verbose:
                print(
                    f"[{rid}] Developer 输出：\n"
                    f"{json.dumps(impl, ensure_ascii=False, indent=2)}"
                )

            DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
            ext = impl.get("artifact_extension", "txt").lstrip(".")
            path = DELIVERABLE_DIR / f"{sanitize_filename(rid)}_{round_idx}.{ext}"
            artifact_content = impl.get("artifact_content", "")
            if isinstance(artifact_content, (dict, list)):
                artifact_content = json.dumps(artifact_content, ensure_ascii=False, indent=2)
            path.write_text(str(artifact_content), encoding="utf-8")
            final_paths[rid] = path
            state.update(
                {
                    "artifact": str(artifact_content),
                    "path": path,
                    "summary": impl.get("summary", ""),
                    "blueprint": blueprint,
                },
            )

            qa_report = await qa_requirement(
                llm=llm,
                requirement=requirement,
                blueprint=blueprint,
                artifact_path=path,
                criteria=criteria,
                round_index=round_idx,
            )
            print(f"[{rid}] QA 判定共 {len(qa_report.get('criteria', []))} 条标准")
            if verbose:
                print(
                    f"[{rid}] QA 输出：\n"
                    f"{json.dumps(qa_report, ensure_ascii=False, indent=2)}"
                )

            crit = qa_report.get("criteria", [])
            passed = sum(1 for item in crit if item.get("pass"))
            total = max(len(crit), 1)
            pass_ratio = passed / total
            requirement_pass_flags.append(pass_ratio >= overall_target and passed == total)
            for item in crit:
                if item.get("pass") and item.get("id"):
                    state["passed"].add(item["id"])

            if pass_ratio >= overall_target and passed == total:
                feedback_map[rid] = ""
                state["feedback"] = ""
            else:
                feedback = qa_report.get("improvements", "")
                feedback_map[rid] = feedback
                state["feedback"] = feedback

            round_entry["results"].append(
                {
                    "requirement_id": rid,
                    "blueprint": blueprint,
                    "implementation": {
                        "summary": impl.get("summary", ""),
                        "path": str(path),
                    },
                    "qa": qa_report,
                    "pass_ratio": pass_ratio,
                },
            )

            print(
                f"- {rid} 通过 {passed}/{total} = {pass_ratio:.2%} "
                f"-> {'通过' if pass_ratio >= overall_target and passed == total else '未通过'}",
            )

        rounds.append(round_entry)

        if all(requirement_pass_flags):
            print("所有需求均达标，结束执行。")
            break
        else:
            print("仍有需求未达标，进入下一轮。")

    return {"rounds": rounds, "deliverables": final_paths}


# ---------------------------------------------------------------------------
# CLI 主流程
# ---------------------------------------------------------------------------
async def run_cli(
    initial_requirement: str,
    scripted_inputs: list[str] | None = None,
    auto_confirm: bool = False,
    provider: str = "auto",
    ollama_model: str = "qwen3:30b",
    ollama_host: str = "http://localhost:11434",
    verbose: bool = False,
    user_id: str = "cli-user",
    project_id: str | None = None,
    mcp_configs: list[MCPServerConfig] | None = None,
) -> None:
    silicon_creds = load_siliconflow_env()
    llm, provider_used = initialize_llm(
        provider,
        silicon_creds,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )
    print(f"使用 LLM 提供方: {provider_used}")

    spec = await collect_spec(llm, initial_requirement, scripted_inputs, auto_confirm)
    await enrich_acceptance_map(llm, spec)
    mcp_clients = build_mcp_clients(mcp_configs)
    mcp_context, resource_handles = await summarize_mcp_clients(mcp_clients)
    runtime = build_runtime_harness(
        spec,
        user_id=user_id,
        project_hint=project_id,
        resource_handles=resource_handles,
    )
    runtime_payload = json.dumps(
        {
            "initial_requirement": initial_requirement,
            "summary": spec.get("summary"),
            "requirements": spec.get("requirements", []),
        },
        ensure_ascii=False,
        indent=2,
    )
    runtime_msg = await runtime.aa_agent.reply(
        Msg(name=user_id, role="user", content=runtime_payload),
    )
    runtime_text = runtime_msg.get_text_content() or ""
    print("\n========== Hive Runtime (AA) ==========")
    print(runtime_text or "[无文本]")
    runtime_meta = runtime_msg.metadata or {}
    deliverable_meta = runtime_meta.get("deliverable")
    if deliverable_meta:
        print(
            f"Runtime 交付模拟: {deliverable_meta.get('type')} -> "
            f"{deliverable_meta.get('uri')}",
        )
    if mcp_context:
        print("\n检测到 MCP 资源：")
        print(mcp_context)

    result = await run_execution(
        llm,
        spec,
        verbose=verbose,
        runtime_summary=runtime_text,
        mcp_context=mcp_context or None,
    )

    print("\n========== 最终交付 ==========")
    for rid, path in result["deliverables"].items():
        print(f"- {rid}: file://{path.resolve()}")

    last_round = result["rounds"][-1]
    print("\n========== 验收结果 ==========")
    for item in last_round["results"]:
        rid = item["requirement_id"]
        qa = item["qa"]
        print(f"\n[{rid}]")
        for criterion in qa.get("criteria", []):
            status = "通过" if criterion.get("pass") else "不通过"
            print(f"{status} - {criterion.get('name')}: {criterion.get('reason')}")
        print(f"通过率: {item['pass_ratio']:.2%}")


def parse_auto_inputs(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [token.strip() for token in value.split("||") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="HiveCore 全流程 CLI")
    parser.add_argument("-r", "--requirement", dest="requirement", help="初始需求描述")
    parser.add_argument("--auto-answers", dest="auto_answers", help="使用 '||' 分隔的自动输入（含 confirm）")
    parser.add_argument("--auto-confirm", dest="auto_confirm", action="store_true", help="自动确认需求（适用于无人值守测试）")
    parser.add_argument(
        "--provider",
        choices=["auto", "siliconflow", "ollama"],
        default="auto",
        help="选择 LLM 提供方，auto 表示优先使用硅基流动，否则回退到本地 Ollama",
    )
    parser.add_argument("--ollama-model", dest="ollama_model", default="qwen3:30b", help="Ollama 模型名称")
    parser.add_argument("--ollama-host", dest="ollama_host", default="http://localhost:11434", help="Ollama 服务地址")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印完整的 Agent 输出内容")
    parser.add_argument("--user-id", dest="user_id", default="cli-user", help="绑定到 runtime 的用户 ID")
    parser.add_argument("--project-id", dest="project_id", help="指定项目 ID（若不提供则自动生成）")
    parser.add_argument(
        "--mcp-server",
        dest="mcp_servers",
        action="append",
        help="注册 MCP 服务，格式 name,url 或 name,transport,url，可重复使用",
    )
    args = parser.parse_args()

    requirement = args.requirement or input("请输入你的项目需求：").strip()
    scripted = parse_auto_inputs(args.auto_answers)
    mcp_configs = None
    if args.mcp_servers:
        parsed: list[MCPServerConfig] = []
        for raw in args.mcp_servers:
            try:
                parsed.append(parse_mcp_server(raw))
            except ValueError as exc:
                print(f"[WARN] 忽略无效 MCP 参数 '{raw}': {exc}")
        mcp_configs = parsed or None
    asyncio.run(
        run_cli(
            requirement,
            scripted,
            args.auto_confirm,
            provider=args.provider,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            verbose=args.verbose,
            user_id=args.user_id,
            project_id=args.project_id,
            mcp_configs=mcp_configs,
        ),
    )


if __name__ == "__main__":
    main()
