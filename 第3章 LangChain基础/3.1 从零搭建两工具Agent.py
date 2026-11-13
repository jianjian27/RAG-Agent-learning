import os

from langchain.tools import tool
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI


PAPERS = {
    "P-101": {
        "title": "忆阻器神经元的动力学分析",
        "page": 5,
        "evidence": "本文利用分岔图和最大 Lyapunov 指数分析混沌行为。",
    },
    "P-202": {
        "title": "分数阶 Jerk 系统的吸引子研究",
        "page": 8,
        "evidence": "本文利用 Poincaré 截面观察吸引子结构。",
    },
}

@tool
def find_paper(title_hint: str) -> list[dict[str, str]]:
    """按标题关键词查找论文，返回匹配论文的标题和 ID。"""
    keyword = title_hint.strip().casefold()
    if not keyword:
        return []

    return [
        {"paper_id": paper_id, "title": paper["title"]}
        for paper_id, paper in PAPERS.items()
        if keyword in paper["title"].casefold()
    ]


@tool
def read_evidence(paper_id: str) -> dict[str, str | int]:
    """根据 find_paper 返回的论文 ID，读取该论文的原文证据和页码。"""
    paper = PAPERS.get(paper_id)
    if paper is None:
        return {"error": "没有找到这个论文 ID。"}

    return {
        "title": paper["title"],
        "page": paper["page"],
        "evidence": paper["evidence"],
    }


def inspect_tools() -> None:
    print("查询工具的参数结构")
    print(find_paper.get_input_schema().model_json_schema())

    matches = find_paper.invoke({"title_hint": "忆阻器"})
    print("查询结果", matches)

    paper_id = matches[0]["paper_id"]
    evidence = read_evidence.invoke({"paper_id": paper_id})
    print("证据结果", evidence)


def run_agent(question: str) -> None:
    load_dotenv()

    model = ChatOpenAI(
        model=os.environ["DASHSCOPE_MODEL"],
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ["DASHSCOPE_BASE_URL"],
        use_responses_api=False,
        extra_body={"enable_thinking": False},
        timeout=30,
        max_retries=0,
    )

    agent = create_agent(
        model=model,
        tools=[find_paper, read_evidence],
        system_prompt=(
            "你是论文证据助手。先用标题关键词查找论文。"
            "只有找到唯一匹配的论文时，才用查询结果里的 ID 读取证据。"
            "找不到时说明未找到，匹配到多篇时请用户明确要查询哪篇论文。"
            "回答只使用工具返回的证据，并注明页码。"
        ),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 12},
    )

    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print("模型请求", call["name"], call["args"], call["id"])
        elif isinstance(message, ToolMessage):
            print("工具结果", message.tool_call_id, message.content)

    print("最终回答", result["messages"][-1].text)


if __name__ == "__main__":
    run_agent("Jerk 那篇论文用了什么分析方法？请注明页码。")

