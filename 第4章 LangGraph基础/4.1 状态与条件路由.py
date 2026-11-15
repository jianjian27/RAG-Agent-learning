from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


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


def read_evidence(state: PaperState) -> PaperState:
    paper_id = state["matched_ids"][0]
    return {"evidence": PAPERS[paper_id]["evidence"]}


def explain_no_unique_match(state: PaperState) -> PaperState:
    if not state["matched_ids"]:
        return {"message": "未找到匹配的论文。"}
    return {"message": "找到多篇论文，请提供更具体的标题关键词。"}


builder = StateGraph(PaperState)
builder.add_node("find_paper", find_paper)
builder.add_node("read_evidence", read_evidence)
builder.add_node("explain_no_unique_match", explain_no_unique_match)

builder.add_edge(START, "find_paper")
builder.add_conditional_edges(
    "find_paper",
    route_after_search,
    {
        "read": "read_evidence",
        "stop": "explain_no_unique_match",
    },
)
builder.add_edge("read_evidence", END)
builder.add_edge("explain_no_unique_match", END)

workflow = builder.compile()


if __name__ == "__main__":
    for keyword in ("忆阻器", "Jerk", "洛伦兹", "动力学", ""):
        result = workflow.invoke({"title_hint": keyword})
        print(keyword, result)
