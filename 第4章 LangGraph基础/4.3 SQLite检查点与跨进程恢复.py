from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import argparse
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith.resources.sandboxes import snapshots

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

CHECKPOINT_DB = Path(__file__).with_name("checkpoint.sqlite3")


def main() -> None:
    parser = argparse.ArgumentParser(description="练习跨进程恢复论文确认流程")
    commands = parser.add_subparsers(dest="action", required=True)

    start = commands.add_parser("start", help="新建任务")
    start.add_argument("keyword", help="论文标题关键词")

    resume = commands.add_parser("resume", help="恢复等待确认的任务")
    resume.add_argument("thread_id", help="start 打印的任务 ID")
    resume.add_argument("decision", choices=("y", "n"))

    args = parser.parse_args()
    thread_id = str(uuid4()) if args.action == "start" else args.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        workflow = builder.compile(checkpointer=saver)

        if args.action == "start":
            result = workflow.invoke(
                {"title_hint": args.keyword},
                config=config,
            )
            pending = result.get("__interrupt__", ())
            if pending:
                print(pending[0].value["question"])
                print("任务 ID:", thread_id)
                return

        else:
            snapshot = workflow.get_state(config)
            if not any(task.interrupts for task in snapshot.tasks):
                parser.error("该任务 ID 没有等待确认的中断。")

            result = workflow.invoke(
                Command(resume=(args.decision == "y")),
                config=config,
            )

        print("最终状态", result)


if __name__ == "__main__":
    main()
