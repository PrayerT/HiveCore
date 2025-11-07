# -*- coding: utf-8 -*-
"""
HiveCore CLI：真实 LLM 多 Agent 协作流程演示

特性
-----
- 自动读取 `~/agentscope/.env` 中的 SILICONFLOW_* 配置，调用硅基流动 OpenAI 兼容接口。
- AA 通过多轮追问补齐需求，输出包含细粒度验收标准的 JSON。
- Planner / Designer / Developer / QA 四个 LLM Agent 共享上下文，协同产出与评审。
- QA 按照 “通过条数/总条数” 计算验收质量；若未达阈值会带着反馈进入下一轮迭代。
- 全部文案、任务、交付和验收结论均由 LLM 实时生成，无任何写死数据。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from agentscope.model import OpenAIChatModel


# ---------------------------------------------------------------------------
# 环境与 LLM 辅助
# ---------------------------------------------------------------------------

def load_siliconflow_env() -> dict[str, str]:
    """Load SiliconFlow credentials from ~/agentscope/.env."""
    env_path = Path.home() / "agentscope" / ".env"
    if env_path.exists():
        with env_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
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


async def call_llm_text(
    llm: OpenAIChatModel,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.3,
) -> str:
    """Call LLM and return plain text content."""
    resp = await llm(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        stream=False,
        temperature=temperature,
    )
    text = "".join(
        block.get("text", "")
        for block in resp.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    return text.strip()


def extract_json_block(text: str) -> str:
    """Extract JSON substring from text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未找到 JSON 区块")
    return text[start : end + 1]


# ---------------------------------------------------------------------------
# 需求澄清阶段
# ---------------------------------------------------------------------------

async def gather_spec(
    llm: OpenAIChatModel,
    initial_requirement: str,
    scripted_answers: list[str] | None = None,
) -> dict[str, Any]:
    """Iteratively clarify requirements and return structured spec."""
    system_prompt = textwrap.dedent(
        """
        你是 HiveCore 的 AssistantAgent，需与用户多轮对话收集需求。
        如果信息不足，请提出一个问题；如果足够，请输出：
        READY::{"project_name": "...","summary": "...",
                 "requirements":[{"id":"task-1","role":"Planner",
                                  "skills":["planning"],"tools":["figma"],
                                  "goal":"..."}],
                 "acceptance":{"criteria":[
                    {"name":"内容深度","description":"...","target":0.9},
                    {"name":"报名字段完整","description":"...","target":1.0}
                 ],"overall_target":0.9},
                 "artifact_type":"web"}
        JSON 必须有效、紧凑且不含注释。
        """
    ).strip()
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_requirement},
    ]
    print("\nAA: 已记录初始需求，开始澄清。")
    while True:
        response = await llm(conversation, stream=False, temperature=0.2)
        text = "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        conversation.append({"role": "assistant", "content": text})
        if text.startswith("READY::"):
            payload = text.split("READY::", 1)[1].strip()
            spec = json.loads(payload)
            print("AA: 需求收集完成。")
            return spec
        print(f"AA: {text}")
        if scripted_answers:
            user_reply = scripted_answers.pop(0)
            print(f"用户(自动): {user_reply}")
        else:
            user_reply = input("用户: ").strip() or "请继续完善。"
        conversation.append({"role": "user", "content": user_reply})


# ---------------------------------------------------------------------------
# 多 Agent 协作
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    role: str
    content: str


async def planner_agent(llm: OpenAIChatModel, spec: dict, feedback: str) -> AgentOutput:
    prompt = textwrap.dedent(
        f"""
        需求摘要:
        {json.dumps(spec, ensure_ascii=False, indent=2)}

        前一轮 QA 反馈（如有）:
        {feedback or "无"}

        任务:
        - 拆解为 3-5 个明确可执行的任务节点，涵盖文案/视觉/开发。
        - 说明每个任务的负责人、输入、输出。
        - 使用 Markdown bullet 列表。
        """
    )
    content = await call_llm_text(
        llm,
        "你是项目规划师，擅长将需求拆解为可执行任务。",
        prompt,
        temperature=0.3,
    )
    return AgentOutput("Planner", content)


async def designer_agent(
    llm: OpenAIChatModel,
    spec: dict,
    planner_notes: str,
    feedback: str,
) -> AgentOutput:
    prompt = textwrap.dedent(
        f"""
        参考需求与 Planner 任务:
        {planner_notes}

        具体要求:
        - 生成新品官网的文案结构 (Hero / 核心卖点 / 报名区域)。
        - 以 Markdown 标题 + 列表形式输出，给出 CTA 文案。
        - 若 QA 有反馈，请对相应部分做出改进: {feedback or "无"}。
        """
    )
    content = await call_llm_text(
        llm,
        "你是资深 UX / 内容设计师，负责撰写网站文案与结构。",
        prompt,
        temperature=0.4,
    )
    return AgentOutput("Designer", content)


async def developer_agent(
    llm: OpenAIChatModel,
    spec: dict,
    planner_notes: str,
    designer_notes: str,
    feedback: str,
) -> AgentOutput:
    prompt = textwrap.dedent(
        f"""
        需求和规划:
        {planner_notes}

        文案与结构:
        {designer_notes}

        任务:
        - 输出页面信息架构与组件清单（section、模块、表单字段）。
        - 给出实现建议（如技术栈、组件划分）。
        - 若 QA 有反馈，请重点解决: {feedback or "无"}。
        - 使用 Markdown，有条理地展示。
        """
    )
    content = await call_llm_text(
        llm,
        "你是全栈实现负责，需将方案转为可执行实现计划。",
        prompt,
        temperature=0.35,
    )
    return AgentOutput("Developer", content)


async def qa_agent(
    llm: OpenAIChatModel,
    spec: dict,
    deliverable: str,
    round_index: int,
) -> dict[str, Any]:
    acceptance = spec.get("acceptance", {})
    criteria = acceptance.get("criteria") or [
        {"name": "整体质量", "description": spec.get("summary", ""), "target": 0.9},
    ]
    prompt = textwrap.dedent(
        f"""
        验收标准 (JSON):
        {json.dumps(criteria, ensure_ascii=False, indent=2)}

        交付物 (Round {round_index}):
        {deliverable}

        请严格按照以下 JSON 格式输出：
        {{
          "round": {round_index},
          "criteria": [
            {{"name": "...", "pass": true/false, "reason": "...", "recommendation": "..."}}
          ]
        }}
        不要输出额外文本。
        """
    )
    raw = await call_llm_text(
        llm,
        "你是严格的 QA，需要逐条核对并返回 JSON 结果。",
        prompt,
        temperature=0.2,
    )
    try:
        data = json.loads(extract_json_block(raw))
    except Exception as exc:  # pragma: no cover - fallback parsing
        raise RuntimeError(f"QA 输出无法解析为 JSON:\n{raw}") from exc
    return data


async def run_collaboration(
    llm: OpenAIChatModel,
    spec: dict,
    *,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Run multi-agent collaboration rounds until acceptance met."""
    rounds: list[dict[str, Any]] = []
    feedback = ""
    overall_target = spec.get("acceptance", {}).get("overall_target", 0.9)

    for round_index in range(1, max_rounds + 1):
        print(f"\n---- Round {round_index} ----")

        planner_output = await planner_agent(llm, spec, feedback)
        print("\n[Planner]\n", planner_output.content)

        designer_output = await designer_agent(llm, spec, planner_output.content, feedback)
        print("\n[Designer]\n", designer_output.content)

        developer_output = await developer_agent(
            llm,
            spec,
            planner_output.content,
            designer_output.content,
            feedback,
        )
        print("\n[Developer]\n", developer_output.content)

        qa_data = await qa_agent(llm, spec, developer_output.content, round_index)
        criteria = qa_data.get("criteria", [])
        passed = sum(1 for item in criteria if item.get("pass"))
        total = max(len(criteria), 1)
        pass_ratio = passed / total

        rounds.append(
            {
                "round": round_index,
                "planner": planner_output.content,
                "designer": designer_output.content,
                "developer": developer_output.content,
                "qa": qa_data,
                "pass_ratio": pass_ratio,
                "passed": passed,
                "total": total,
            },
        )

        print("\n[QA]")
        for item in criteria:
            status = "✅" if item.get("pass") else "❌"
            print(
                f"{status} {item.get('name')}: {item.get('reason')} | 建议: {item.get('recommendation')}",
            )
        print(f"Round {round_index} 通过率: {passed}/{total} = {pass_ratio:.2%}")

        if pass_ratio >= overall_target:
            print("验收达标，结束迭代。")
            break

        failing = [
            f"- {item.get('name')}: {item.get('recommendation')}"
            for item in criteria
            if not item.get("pass")
        ]
        feedback = "\n".join(failing)
        print("需要改进：\n", feedback)

    return {
        "rounds": rounds,
        "final_pass_ratio": rounds[-1]["pass_ratio"],
        "final_deliverable": rounds[-1]["developer"],
        "final_qa": rounds[-1]["qa"],
    }


# ---------------------------------------------------------------------------
# CLI 主流程
# ---------------------------------------------------------------------------

async def run_cli(initial_requirement: str, scripted_answers: list[str] | None = None) -> None:
    creds = load_siliconflow_env()
    llm = OpenAIChatModel(
        model_name=creds["model"],
        api_key=creds["api_key"],
        stream=False,
        client_args={"base_url": creds["base_url"]},
    )

    spec = await gather_spec(llm, initial_requirement, scripted_answers)
    result = await run_collaboration(llm, spec)

    print("\n========== 最终交付 ==========")
    print(result["final_deliverable"])

    final_qa = result["final_qa"]
    criteria = final_qa.get("criteria", [])
    print("\n========== 验收结果 ==========")
    for item in criteria:
        status = "通过" if item.get("pass") else "不通过"
        print(f"{item.get('name')}: {status} — {item.get('reason')}")
    print(
        f"整体通过率: {result['rounds'][-1]['passed']}/{result['rounds'][-1]['total']} "
        f"= {result['final_pass_ratio']:.2%}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HiveCore 用户全流程 CLI")
    parser.add_argument(
        "--requirement",
        "-r",
        dest="requirement",
        help="初始需求描述（未提供则交互输入）",
    )
    parser.add_argument(
        "--auto-answers",
        dest="auto_answers",
        help="使用 '||' 分隔的预设回答，方便自动化测试",
    )
    args = parser.parse_args()
    requirement = args.requirement or input("请输入你的项目需求：").strip()
    scripted = None
    if args.auto_answers:
        scripted = [item.strip() for item in args.auto_answers.split("||")]
    asyncio.run(run_cli(requirement, scripted))


if __name__ == "__main__":
    main()
