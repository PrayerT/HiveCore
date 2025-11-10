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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentscope.model import OpenAIChatModel, OllamaChatModel

DELIVERABLE_DIR = Path("deliverables")


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
    llm: OpenAIChatModel,
    requirement: dict[str, Any],
    feedback: str,
) -> dict[str, Any]:
    artifact_type = resolve_artifact_type(requirement)
    prompt = textwrap.dedent(
        f"""
        需求对象:
        {json.dumps(requirement, ensure_ascii=False, indent=2)}

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
    llm: OpenAIChatModel,
    requirement: dict[str, Any],
    blueprint: dict[str, Any],
    feedback: str,
) -> dict[str, Any]:
    artifact_type = blueprint.get("artifact_spec", {}).get("format") or resolve_artifact_type(requirement)
    prompt = textwrap.dedent(
        f"""
        需求:
        {json.dumps(requirement, ensure_ascii=False, indent=2)}

        Blueprint:
        {json.dumps(blueprint, ensure_ascii=False, indent=2)}

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
async def run_execution(llm: OpenAIChatModel, spec: dict[str, Any], max_rounds: int = 3) -> dict[str, Any]:
    requirements = spec.get("requirements", [])
    overall_target = spec.get("acceptance", {}).get("overall_target", 0.95)
    feedback_map = {req["id"]: "" for req in requirements}
    rounds: list[dict[str, Any]] = []
    final_paths: dict[str, Path] = {}

    for round_idx in range(1, max_rounds + 1):
        print(f"\n---- 执行轮次 Round {round_idx} ----")
        round_entry = {"round": round_idx, "results": []}
        requirement_pass_flags = []

        for requirement in requirements:
            rid = requirement["id"]
            criteria = criteria_for_requirement(spec, rid)

            blueprint = await design_requirement(llm, requirement, feedback_map[rid])
            print(f"\n[{rid}] Blueprint 摘要：{blueprint.get('deliverable_pitch', '')}")
            impl = await implement_requirement(llm, requirement, blueprint, feedback_map[rid])
            print(f"[{rid}] Developer Summary：{impl.get('summary', '')}")

            DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
            ext = impl.get("artifact_extension", "txt").lstrip(".")
            path = DELIVERABLE_DIR / f"{sanitize_filename(rid)}_{round_idx}.{ext}"
            artifact_content = impl.get("artifact_content", "")
            if isinstance(artifact_content, (dict, list)):
                artifact_content = json.dumps(artifact_content, ensure_ascii=False, indent=2)
            path.write_text(str(artifact_content), encoding="utf-8")
            final_paths[rid] = path

            qa_report = await qa_requirement(
                llm=llm,
                requirement=requirement,
                blueprint=blueprint,
                artifact_path=path,
                criteria=criteria,
                round_index=round_idx,
            )
            print(f"[{rid}] QA 判定共 {len(qa_report.get('criteria', []))} 条标准")

            crit = qa_report.get("criteria", [])
            passed = sum(1 for item in crit if item.get("pass"))
            total = max(len(crit), 1)
            pass_ratio = passed / total
            requirement_pass_flags.append(pass_ratio >= overall_target and passed == total)

            if pass_ratio >= overall_target and passed == total:
                feedback_map[rid] = ""
            else:
                feedback_map[rid] = qa_report.get("improvements", "")

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
    result = await run_execution(llm, spec)

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
    args = parser.parse_args()

    requirement = args.requirement or input("请输入你的项目需求：").strip()
    scripted = parse_auto_inputs(args.auto_answers)
    asyncio.run(
        run_cli(
            requirement,
            scripted,
            args.auto_confirm,
            provider=args.provider,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
        ),
    )


if __name__ == "__main__":
    main()
