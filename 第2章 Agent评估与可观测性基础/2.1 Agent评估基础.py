# 学习目标：
# 1. 区分“程序正常运行”和“Agent完成任务”。
# 2. 使用参考结果检查Agent实际结果。
# 3. 记录失败原因并统计通过率。
# 4. 验证评分器是否符合我们制定的规则。


def build_cases() -> list[dict]:
    """准备评估案例。"""
    return [
        {
            "case_id": "location_correct",
            "question": "论文paper_001中的同步条件位于哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
            "actual": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
            # 人工标注：正确
            "expected_passed": True,
        },
        {
            "case_id": "location_wrong",
            "question": "论文paper_001中的同步条件位于哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
            "actual": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 18,
            },
            "expected_passed": False,
        },
        {
            "case_id": "unsupported_answer",
            "question": "现有材料是否证明该方法适用于所有非线性系统？",
            "expected": {
                "status": "insufficient_evidence",
            },
            "actual": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
            "expected_passed": False,
        },
        {
            "case_id": "insufficient_evidence_correct",
            "question": "现有材料是否包含该方法在所有系统上的证明？",
            "expected": {
                "status": "insufficient_evidence",
            },
            "actual": {
                "status": "insufficient_evidence",
            },
            "expected_passed": True,
        },
        {
            "case_id": "paper_wrong",
            "question": "论文paper_001中的同步条件位于哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
            "actual": {
                "status": "answered",
                "paper_id": "paper_002",
                "page": 12,
            },
            "expected_passed": False,
        },
        {
            "case_id": "page_missing",
            "question": "论文paper_001中的同步条件位于哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
            "actual": {
                "status": "answered",
                "paper_id": "paper_001",
                # 故意缺少page，用于检查失败分支。
            },
            "expected_passed": False,
        },
        {
            "case_id": "paper_page_wrong",
            "question": "论文paper_001中的同步条件位于哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
            "actual": {
                "status": "answered",
                "paper_id": "paper_002",
                "page": 17,
            },
            "expected_passed": False,
        },
    ]


def evaluate_case(case: dict) -> dict:
    """按照规则，评估一个案例。"""
    expected = case["expected"]
    actual = case["actual"]

    reasons: list[str] = []

    if actual.get("status") != expected.get("status"):
        reasons.append("回答状态不符合预期")

    # 只有参考结果要求提供定位时，才检查论文和页码。
    if expected["status"] == "answered":
        if actual.get("paper_id") != expected.get("paper_id"):
            reasons.append("论文编号错误或缺失")

        if actual.get("page") != expected["page"]:
            reasons.append("页码错误或缺失")

    return {
        "case_id": case["case_id"],
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }


def run_evaluation(cases: list[dict]) -> list[dict]:
    """评估所有案例，返回每个案例的评分效果。"""
    if not cases:
        raise ValueError("评估案例不能为空。")

    results = []

    for case in cases:
        result = evaluate_case(case)
        results.append(result)

    return results


def print_report(results: list[dict]) -> None:
    """打印逐条结果和整体通过率。"""
    if not results:
        raise ValueError("评估结果不能为空。")

    passed_count = 0

    for result in results:
        if result["passed"]:
            passed_count += 1
            print(f"{result['case_id']}: 通过")
        else:
            print(f"{result['case_id']}: 失败")

            for reason in result["reasons"]:
                print(f"  原因：{reason}")

    total_count = len(results)
    pass_rate = passed_count / total_count

    print()
    print(f"总案例数：{total_count}")
    print(f"通过数：{passed_count}")
    print(f"失败数：{total_count - passed_count}")
    print(f"通过率：{pass_rate:.1%}")


def check_evaluator(cases: list[dict]) -> None:
    """检查评分器的判断是否符合人工标注。"""
    for case in cases:
        result = evaluate_case(case)

        assert result["passed"] == case["expected_passed"], (
            f"评分器判断不符合预期：{case['case_id']}"
        )

    print("\n评分器检查通过：所有判断均符合人工标注。")


def main() -> None:
    cases = build_cases()

    results = run_evaluation(cases)

    print_report(results)

    check_evaluator(cases)


if __name__ == "__main__":
    main()


