from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


PAPERS = {
    "P-101": {
        "title": "忆阻器神经元的动力学分析",
        "evidence": "第 5 页：使用分岔图和最大 Lyapunov 指数。",
    },
    "P-202": {
        "title": "分数阶 Jerk 系统的动力学分析",
        "evidence": "第 8 页：使用 Poincaré 截面观察吸引子。",
    },
}


class PaperState(TypedDict, total=False):
    title_hint: str
    matched_ids: list[str]
    evidence: str
    message: str
    approved: bool


def find_paper(state: PaperState) -> PaperState:
    keyword = state["title_hint"].strip().casefold()
    matched_ids = [
        paper_id
        for paper_id, paper in PAPERS.items()
        if keyword and keyword in paper["title"].casefold()
    ]
    return {"matched_ids": matched_ids}


def route_after_search(state: PaperState) -> Literal["read", "stop"]:
    if len(state["matched_ids"]) == 1:
        return "read"
    return "stop"


def confirm_paper(state: PaperState) -> PaperState:
    paper_id = state["matched_ids"][0]
    title = PAPERS[paper_id]["title"]

    decision = interrupt({
        "paper_id": paper_id,
        "title": title,
        "question": f"确认读取《{title}》的证据吗？",
    })

    return {"approved": decision is True}


def route_after_confirmation(state: PaperState) -> Literal["read", "cancel"]:
    if state["approved"]:
        return "read"
    return "cancel"


def cancel_reading(state: PaperState) -> PaperState:
    return {"message": "已取消证据读取。"}


def read_evidence(state: PaperState) -> PaperState:
    paper_id = state["matched_ids"][0]
    return {"evidence": PAPERS[paper_id]["evidence"]}


def explain_no_unique_match(state: PaperState) -> PaperState:
    if not state["matched_ids"]:
        return {"message": "未找到匹配的论文。"}
    return {"message": "找到多篇论文，请提供更具体的标题关键词。"}


builder = StateGraph(PaperState)
builder.add_node("find_paper", find_paper)
builder.add_node("confirm_paper", confirm_paper)
builder.add_node("read_evidence", read_evidence)
builder.add_node("cancel_reading", cancel_reading)
builder.add_node("explain_no_unique_match", explain_no_unique_match)

builder.add_edge(START, "find_paper")

builder.add_conditional_edges(
    "find_paper",
    route_after_search,
    {
        "read": "confirm_paper",
        "stop": "explain_no_unique_match",
    },
)

builder.add_conditional_edges(
    "confirm_paper",
    route_after_confirmation,
    {
        "read": "read_evidence",
        "cancel": "cancel_reading",
    },
)

builder.add_edge("read_evidence", END)
builder.add_edge("cancel_reading", END)
builder.add_edge("explain_no_unique_match", END)

workflow = builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    keyword = input("请输入标题关键词：").strip()
    config = {"configurable": {"thread_id": str(uuid4())}}

    result = workflow.invoke({"title_hint": keyword}, config=config)
    pending = result.get("__interrupt__", ())

    if pending:
        payload = pending[0].value
        print("暂停时的匹配结果", result["matched_ids"])
        assert "evidence" not in result

        answer = input(f"{payload['question']} [y/n] ").strip().lower()
        while answer not in {"y", "n"}:
            answer = input("请输入 y 或 n：").strip().lower()

        result = workflow.invoke(
            Command(resume=(answer == "y")),
            config=config
        )

    print("最终状态", result)


if __name__ == "__main__":
    main()
