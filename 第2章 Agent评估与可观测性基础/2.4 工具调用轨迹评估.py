"""
2.4 工具调用轨迹评估

学习目标：
1. 理解Agent轨迹与最终答案的区别。
2. 检查工具选择、参数、顺序和调用数量。
3. 为轨迹生成分数和具体失败原因。
4. 区分正确、错误、遗漏和多余调用。

本文件使用手动构造的模拟轨迹，不调用真实模型。
"""

from pydantic import BaseModel, ConfigDict, Field


class EvaluationModel(BaseModel):
    """评估模型共用配置。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class ToolCall(EvaluationModel):
    """轨迹中的一次工具调用。"""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, object]


class TraceEvalCase(EvaluationModel):
    """一条轨迹评估案例。"""

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    expected_trace: list[ToolCall]
    actual_trace: list[ToolCall]

    # 只用于检查本节评分器是否符合人工预期。
    expected_passed: bool


class TraceEvaluationResult(EvaluationModel):
    """一条案例的轨迹评分结果。"""

    case_id: str
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    matched_steps: int = Field(ge=0)
    compared_steps: int = Field(ge=0)
    reasons: list[str]


def evaluate_trace(
    case: TraceEvalCase,
) -> TraceEvaluationResult:
    """
    按位置比较参考轨迹和实际轨迹。

    一个步骤只有在工具名称和参数都相同时，
    才算完全匹配。
    """
    expected_count = len(case.expected_trace)
    actual_count = len(case.actual_trace)

    compared_steps = max(
        expected_count,
        actual_count,
    )

    matched_steps = 0
    reasons: list[str] = []

    for index in range(compared_steps):
        step_number = index + 1

        # 实际轨迹比参考轨迹更长。
        if index >= expected_count:
            actual_call = case.actual_trace[index]

            reasons.append(
                f"多余第{step_number}步工具调用："
                f"{actual_call.tool_name}"
            )
            continue

        # 实际轨迹比参考轨迹更短。
        if index >= actual_count:
            expected_call = case.expected_trace[index]

            reasons.append(
                f"缺少第{step_number}步工具调用："
                f"{expected_call.tool_name}"
            )
            continue

        expected_call = case.expected_trace[index]
        actual_call = case.actual_trace[index]

        if actual_call.tool_name != expected_call.tool_name:
            reasons.append(
                f"第{step_number}步工具错误："
                f"期望{expected_call.tool_name}，"
                f"实际{actual_call.tool_name}"
            )
            continue

        if actual_call.arguments != expected_call.arguments:
            reasons.append(
                f"第{step_number}步参数错误："
                f"期望{expected_call.arguments}，"
                f"实际{actual_call.arguments}"
            )
            continue

        matched_steps += 1

    # 如果参考和实际都没有调用工具，
    # 说明空轨迹完全匹配。
    if compared_steps == 0:
        score = 1.0
    else:
        score = matched_steps / compared_steps

    return TraceEvaluationResult(
        case_id=case.case_id,
        passed=len(reasons) == 0,
        score=score,
        matched_steps=matched_steps,
        compared_steps=compared_steps,
        reasons=reasons,
    )


def build_cases() -> list[TraceEvalCase]:
    """建立正常与失败轨迹案例。"""

    expected_trace = [
        {
            "tool_name": "search_chunks",
            "arguments": {
                "query": "同步控制",
                "top_k": 3,
            },
        },
        {
            "tool_name": "get_chunk",
            "arguments": {
                "chunk_id": "chunk_001",
            },
        },
    ]

    return [
        TraceEvalCase(
            case_id="correct_trace",
            question="查找同步控制内容并读取最相关文本块。",
            expected_trace=expected_trace,
            actual_trace=[
                {
                    "tool_name": "search_chunks",
                    "arguments": {
                        "query": "同步控制",
                        "top_k": 3,
                    },
                },
                {
                    "tool_name": "get_chunk",
                    "arguments": {
                        "chunk_id": "chunk_001",
                    },
                },
            ],
            expected_passed=True,
        ),
        TraceEvalCase(
            case_id="wrong_tool",
            question="查找同步控制内容并读取最相关文本块。",
            expected_trace=expected_trace,
            actual_trace=[
                {
                    "tool_name": "list_papers",
                    "arguments": {},
                },
                {
                    "tool_name": "get_chunk",
                    "arguments": {
                        "chunk_id": "chunk_001",
                    },
                },
            ],
            expected_passed=False,
        ),
        TraceEvalCase(
            case_id="wrong_arguments",
            question="查找同步控制内容并读取最相关文本块。",
            expected_trace=expected_trace,
            actual_trace=[
                {
                    "tool_name": "search_chunks",
                    "arguments": {
                        "query": "同步控制",
                        "top_k": 3,
                    },
                },
                {
                    "tool_name": "get_chunk",
                    "arguments": {
                        "chunk_id": "chunk_999",
                    },
                },
            ],
            expected_passed=False,
        ),
        TraceEvalCase(
            case_id="missing_call",
            question="查找同步控制内容并读取最相关文本块。",
            expected_trace=expected_trace,
            actual_trace=[
                {
                    "tool_name": "search_chunks",
                    "arguments": {
                        "query": "同步控制",
                        "top_k": 3,
                    },
                },
            ],
            expected_passed=False,
        ),
        TraceEvalCase(
            case_id="extra_call",
            question="查找同步控制内容并读取最相关文本块。",
            expected_trace=expected_trace,
            actual_trace=[
                {
                    "tool_name": "search_chunks",
                    "arguments": {
                        "query": "同步控制",
                        "top_k": 3,
                    },
                },
                {
                    "tool_name": "get_chunk",
                    "arguments": {
                        "chunk_id": "chunk_001",
                    },
                },
                {
                    "tool_name": "save_note",
                    "arguments": {
                        "text": "同步控制资料",
                    },
                },
            ],
            expected_passed=False,
        ),
        TraceEvalCase(
            case_id="direct_answer_without_tools",
            question="你好。",
            expected_trace=[],
            actual_trace=[],
            expected_passed=True,
        ),
    ]


def run_evaluation(
    cases: list[TraceEvalCase],
) -> list[TraceEvaluationResult]:
    """运行所有轨迹评估案例。"""
    if not cases:
        raise ValueError("轨迹评估案例不能为空。")

    results: list[TraceEvaluationResult] = []

    for case in cases:
        result = evaluate_trace(case)
        results.append(result)

    return results


def print_report(
    results: list[TraceEvaluationResult],
) -> None:
    """打印每条轨迹结果及整体统计。"""
    if not results:
        raise ValueError("轨迹评估结果不能为空。")

    passed_count = 0
    total_score = 0.0

    for result in results:
        total_score += result.score

        if result.passed:
            passed_count += 1
            status = "通过"
        else:
            status = "失败"

        print(
            f"{result.case_id}：{status}，"
            f"轨迹分数{result.score:.1%}，"
            f"匹配{result.matched_steps}/"
            f"{result.compared_steps}步"
        )

        for reason in result.reasons:
            print(f"  原因：{reason}")

    total_count = len(results)
    pass_rate = passed_count / total_count
    average_score = total_score / total_count

    print()
    print(f"案例总数：{total_count}")
    print(f"完全通过数：{passed_count}")
    print(f"完全通过率：{pass_rate:.1%}")
    print(f"平均轨迹分数：{average_score:.1%}")


def check_evaluator(
    cases: list[TraceEvalCase],
    results: list[TraceEvaluationResult],
) -> None:
    """检查评分器判断是否符合人工标注。"""
    if len(cases) != len(results):
        raise AssertionError("案例数和结果数不一致。")

    for index in range(len(cases)):
        case = cases[index]
        result = results[index]

        assert case.case_id == result.case_id, (
            "案例与评分结果的顺序不一致。"
        )

        assert result.passed == case.expected_passed, (
            f"评分器判断不符合人工预期：{case.case_id}"
        )

    print("\n评分器检查通过：所有判断符合人工标注。")


def main() -> None:
    cases = build_cases()

    results = run_evaluation(cases)

    print_report(results)

    check_evaluator(cases, results)


if __name__ == "__main__":
    main()