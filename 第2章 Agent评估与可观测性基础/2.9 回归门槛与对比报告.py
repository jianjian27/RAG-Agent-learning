# 前面已经产生了很多指标：
# 工具轨迹通过率
# 检索 Recall@K
# 引用正确率
# 声明覆盖率
# Judge 与人工一致率
# Agent 成功率
# 平均延迟
# Token 消耗
# 但只有指标还不够。开发新版本后，我们最终需要回答：
# 新版本能不能替换当前版本？
#
# 因此这一节要实现：
# 旧版本基线指标
#         +
# 新版本候选指标
#         +
# 发布门槛
#         ↓
# 自动生成对比报告
#         ↓
# PASS：允许继续发布
# FAIL：发现质量回归



from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class MetricRule(BaseModel):
    """
    一个指标的发布规则。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    name: str = Field(min_length=1)
    display_name: str = Field(min_length=1)

    direction: Literal[
        "higher_is_better",
        "lower_is_better",
    ]

    unit: Literal[
        "ratio",
        "milliseconds",
        "number",
    ]

    # higher_is_better：
    #   候选值必须 >= absolute_limit
    #
    # lower_is_better：
    #   候选值必须 <= absolute_limit
    absolute_limit: float

    # 相比基线最多允许退步多少。
    max_allowed_regression: float = Field(ge=0)


class EvaluationSnapshot(BaseModel):
    """
    某个Agent版本的一组评估指标。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    version: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)

    metrics: dict[str, float] = Field(
        min_length=1
    )


class MetricComparison(BaseModel):
    name: str
    display_name: str
    direction: Literal[
        "higher_is_better",
        "lower_is_better",
    ]
    unit: Literal[
        "ratio",
        "milliseconds",
        "number",
    ]

    baseline_value: float | None
    candidate_value: float | None
    delta: float | None

    absolute_passed: bool
    regression_passed: bool
    passed: bool

    reasons: list[str]


class ComparisonReport(BaseModel):
    baseline_version: str
    candidate_version: str
    dataset_version: str

    passed: bool
    comparisons: list[MetricComparison]


def compare_metric(
    rule: MetricRule,
    baseline: EvaluationSnapshot,
    candidate: EvaluationSnapshot,
) -> MetricComparison:
    baseline_value = baseline.metrics.get(
        rule.name
    )

    candidate_value = candidate.metrics.get(
        rule.name
    )

    reasons: list[str] = []

    if baseline_value is None:
        reasons.append(
            "基线版本缺少该指标。"
        )

    if candidate_value is None:
        reasons.append(
            "候选版本缺少该指标。"
        )

    if (
        baseline_value is None
        or candidate_value is None
    ):
        return MetricComparison(
            name=rule.name,
            display_name=rule.display_name,
            direction=rule.direction,
            unit=rule.unit,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            delta=None,
            absolute_passed=False,
            regression_passed=False,
            passed=False,
            reasons=reasons,
        )

    delta = candidate_value - baseline_value

    if rule.direction == "higher_is_better":
        absolute_passed = (
            candidate_value
            >= rule.absolute_limit
        )

        regression_limit = (
            baseline_value
            - rule.max_allowed_regression
        )

        regression_passed = (
            candidate_value
            >= regression_limit
        )

        if not absolute_passed:
            reasons.append(
                f"候选值 {candidate_value} "
                f"低于绝对门槛 "
                f"{rule.absolute_limit}。"
            )

        if not regression_passed:
            reasons.append(
                f"候选值相比基线下降过多，"
                f"最多允许下降 "
                f"{rule.max_allowed_regression}。"
            )

    else:
        absolute_passed = (
            candidate_value
            <= rule.absolute_limit
        )

        regression_limit = (
            baseline_value
            + rule.max_allowed_regression
        )

        regression_passed = (
            candidate_value
            <= regression_limit
        )

        if not absolute_passed:
            reasons.append(
                f"候选值 {candidate_value} "
                f"高于绝对上限 "
                f"{rule.absolute_limit}。"
            )

        if not regression_passed:
            reasons.append(
                f"候选值相比基线增加过多，"
                f"最多允许增加 "
                f"{rule.max_allowed_regression}。"
            )

    passed = (
        absolute_passed
        and regression_passed
    )

    if passed:
        reasons.append(
            "绝对门槛和回归门槛均通过。"
        )

    return MetricComparison(
        name=rule.name,
        display_name=rule.display_name,
        direction=rule.direction,
        unit=rule.unit,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        absolute_passed=absolute_passed,
        regression_passed=regression_passed,
        passed=passed,
        reasons=reasons,
    )


def compare_versions(
    *,
    rules: list[MetricRule],
    baseline: EvaluationSnapshot,
    candidate: EvaluationSnapshot,
) -> ComparisonReport:
    if not rules:
        raise ValueError(
            "至少需要提供一条指标规则。"
        )

    rule_names = [
        rule.name
        for rule in rules
    ]

    if len(rule_names) != len(set(rule_names)):
        raise ValueError(
            "指标规则中存在重复的name。"
        )

    if (
        baseline.dataset_version
        != candidate.dataset_version
    ):
        raise ValueError(
            "基线版本和候选版本必须使用"
            "相同的评估数据集。"
        )

    comparisons = [
        compare_metric(
            rule,
            baseline,
            candidate,
        )
        for rule in rules
    ]

    passed = all(
        comparison.passed
        for comparison in comparisons
    )

    return ComparisonReport(
        baseline_version=baseline.version,
        candidate_version=candidate.version,
        dataset_version=baseline.dataset_version,
        passed=passed,
        comparisons=comparisons,
    )


def format_value(
    value: float | None,
    unit: Literal[
        "ratio",
        "milliseconds",
        "number",
    ],
) -> str:
    if value is None:
        return "缺失"

    if unit == "ratio":
        return f"{value:.1%}"

    if unit == "milliseconds":
        return f"{value:.1f} ms"

    return f"{value:.2f}"


def format_delta(
    delta: float | None,
    unit: Literal[
        "ratio",
        "milliseconds",
        "number",
    ],
) -> str:
    if delta is None:
        return "无法计算"

    if unit == "ratio":
        return f"{delta:+.1%}"

    if unit == "milliseconds":
        return f"{delta:+.1f} ms"

    return f"{delta:+.2f}"


def print_report(
    report: ComparisonReport,
) -> None:
    print("\n" + "=" * 60)
    print("Agent版本对比报告")
    print("=" * 60)

    print(
        f"基线版本：{report.baseline_version}"
    )
    print(
        f"候选版本：{report.candidate_version}"
    )
    print(
        f"数据集版本：{report.dataset_version}"
    )
    print(
        f"最终结果："
        f"{'PASS' if report.passed else 'FAIL'}"
    )

    for comparison in report.comparisons:
        print("\n" + "-" * 60)
        print(
            f"指标：{comparison.display_name}"
        )
        print(
            "基线值："
            + format_value(
                comparison.baseline_value,
                comparison.unit,
            )
        )
        print(
            "候选值："
            + format_value(
                comparison.candidate_value,
                comparison.unit,
            )
        )
        print(
            "变化量："
            + format_delta(
                comparison.delta,
                comparison.unit,
            )
        )
        print(
            f"绝对门槛："
            f"{comparison.absolute_passed}"
        )
        print(
            f"回归门槛："
            f"{comparison.regression_passed}"
        )
        print(
            f"指标结果："
            f"{'PASS' if comparison.passed else 'FAIL'}"
        )

        for reason in comparison.reasons:
            print(f"- {reason}")


rules = [
    MetricRule(
        name="retrieval_recall",
        display_name="检索召回率",
        direction="higher_is_better",
        unit="ratio",
        absolute_limit=0.82,
        max_allowed_regression=0.03,
    ),
    MetricRule(
        name="citation_correctness",
        display_name="引用正确率",
        direction="higher_is_better",
        unit="ratio",
        absolute_limit=0.90,
        max_allowed_regression=0.02,
    ),
    MetricRule(
        name="judge_agreement",
        display_name="Judge与人工一致率",
        direction="higher_is_better",
        unit="ratio",
        absolute_limit=0.85,
        max_allowed_regression=0.03,
    ),
    MetricRule(
        name="success_rate",
        display_name="Agent运行成功率",
        direction="higher_is_better",
        unit="ratio",
        absolute_limit=0.93,
        max_allowed_regression=0.02,
    ),
    MetricRule(
        name="average_latency_ms",
        display_name="平均运行延迟",
        direction="lower_is_better",
        unit="milliseconds",
        absolute_limit=2200.0,
        max_allowed_regression=250.0,
    ),
]


baseline = EvaluationSnapshot(
    version="agent-v1.0",
    dataset_version="eval-dataset-v1",
    metrics={
        "retrieval_recall": 0.86,
        "citation_correctness": 0.94,
        "judge_agreement": 0.90,
        "success_rate": 0.96,
        "average_latency_ms": 1800.0,
    },
)


# 正常案例：质量维持或提高，
# 延迟虽然增加，但仍处于允许范围。
good_candidate = EvaluationSnapshot(
    version="agent-v1.1-good",
    dataset_version="eval-dataset-v1",
    metrics={
        "retrieval_recall": 0.88,
        "citation_correctness": 0.95,
        "judge_agreement": 0.91,
        "success_rate": 0.96,
        "average_latency_ms": 1950.0,
    },
)


# 失败案例：
#
# 检索召回率为90%，高于82%的绝对门槛，
# 但相比95%的基线下降太多——这里基线实际为86%，
# 所以该项本身会通过。Judge一致率与延迟则展示
# “绝对门槛通过，但相对回归门槛失败”。
#
# 成功率同时违反绝对门槛和回归门槛。
bad_candidate = EvaluationSnapshot(
    version="agent-v1.1-bad",
    dataset_version="eval-dataset-v1",
    metrics={
        "retrieval_recall": 0.84,
        "citation_correctness": 0.93,
        "judge_agreement": 0.86,
        "success_rate": 0.92,
        "average_latency_ms": 2100.0,
    },
)


# 失败案例：候选版本漏掉了一个必须检查的指标。
incomplete_candidate = EvaluationSnapshot(
    version="agent-v1.1-incomplete",
    dataset_version="eval-dataset-v1",
    metrics={
        "retrieval_recall": 0.88,
        # 故意缺少 citation_correctness
        "judge_agreement": 0.91,
        "success_rate": 0.96,
        "average_latency_ms": 1900.0,
    },
)


good_report = compare_versions(
    rules=rules,
    baseline=baseline,
    candidate=good_candidate,
)

bad_report = compare_versions(
    rules=rules,
    baseline=baseline,
    candidate=bad_candidate,
)

incomplete_report = compare_versions(
    rules=rules,
    baseline=baseline,
    candidate=incomplete_candidate,
)


print_report(good_report)
print_report(bad_report)
print_report(incomplete_report)


# 失败案例：不能比较不同数据集产生的结果。
different_dataset_candidate = EvaluationSnapshot(
    version="agent-v1.2",
    dataset_version="eval-dataset-v2",
    metrics={
        "retrieval_recall": 0.95,
        "citation_correctness": 0.98,
        "judge_agreement": 0.93,
        "success_rate": 0.98,
        "average_latency_ms": 1700.0,
    },
)

print("\n测试不同数据集版本：")

try:
    compare_versions(
        rules=rules,
        baseline=baseline,
        candidate=different_dataset_candidate,
    )
except ValueError as error:
    print(f"无法比较：{error}")