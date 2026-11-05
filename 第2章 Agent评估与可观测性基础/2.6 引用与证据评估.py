# 上一节评估的是：
# 检索器有没有找回正确的 Chunk？
#
# 这一节评估的是：
# Agent 写出答案后，引用是否真实存在？引用的证据是否真的支持这句话？需要引用的结论是否都给出了引用？
#
# 官方 OpenAI 文档也强调，文档问答评估不能只看一个总分，需要针对实际任务定义指标，并结合人工标注校准自动评估结果。OpenAI Evaluation best practices
# 一、引用和证据不是一回事
# 假设 Agent 回答：
# 该论文使用龙格—库塔方法求解系统。[E1]
# 其中：
# - “该论文使用龙格—库塔方法求解系统”是一个声明（claim）；
# - [E1] 是引用（citation），只是一个指向证据的标识；
# - E1 对应的论文 Chunk 才是证据（evidence）。
# 引用存在，不代表它一定正确。例如：
# E1：实验在 NVIDIA GPU 上运行。
# E1 确实存在，但它不支持“使用龙格—库塔方法”这个声明。
#
# 因此，本节定义三个指标：
# 1. 引用有效率：引用指向的证据 ID 是否真实存在。
# 2. 引用正确率：证据是否真的支持对应声明。
# 3. 声明覆盖率：应该引用的声明是否至少有一个正确引用。
# 假设有两个声明：
# 声明1 → 正确引用 E1
# 声明2 → 没有引用
# 那么：
# 引用有效率 = 1 / 1 = 100%
# 引用正确率 = 1 / 1 = 100%
# 声明覆盖率 = 1 / 2 = 50%
# 前两个指标很高，但答案依然漏掉了一半声明的证据。


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


# Evidence 表示检索系统提供给 Agent 的一条证据。
class Evidence(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    evidence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str = Field(min_length=1)


# AnswerClaim 表示 Agent 最终答案中的一条可验证声明。
class AnswerClaim(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    # Agent 为这条声明实际给出的引用。
    citation_ids: list[str]


class CitationEvalCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)

    # 本次回答能够使用的真实证据。
    available_evidence: list[Evidence]

    # Agent 最终答案中包含的声明和实际引用。
    claims: list[AnswerClaim]

    # 人工标注的黄金支持关系：
    # 声明 ID -> 能够正确支持该声明的证据 ID。
    support_map: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_gold_data(self) -> "CitationEvalCase":
        evidence_ids = [
            evidence.evidence_id for evidence in self.available_evidence
        ]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("available_evidence 中存在重复的 evidence_id。")

        claim_ids = [
            claim.claim_id
            for claim in self.claims
        ]

        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claims 中存在重复的 claim_id。")

        if set(self.support_map) != set(claim_ids):
            raise ValueError(
                "support_map 必须为每一条 claim 提供黄金支持关系。"
            )

        valid_evidence_ids = set(evidence_ids)

        for claim_id, supported_ids in self.support_map.items():
            for evidence_id in supported_ids:
                if evidence_id not in valid_evidence_ids:
                    raise ValueError(
                        f"{claim_id} 的黄金证据 {evidence_id} 不存在。"
                    )

        return self


class CitationMetrics(BaseModel):
    case_id: str

    citation_count: int
    valid_citation_count: int
    correct_citation_count: int

    claim_count: int
    covered_claim_count: int

    citation_validity: float
    citation_correctness: float
    claim_coverage: float

    passed: bool
    issues: list[str]


def calculate_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_citations(
    case: CitationEvalCase,
) -> CitationMetrics:
    available_ids = {
        evidence.evidence_id
        for evidence in case.available_evidence
    }

    citation_count = 0
    valid_citation_count = 0
    correct_citation_count = 0
    covered_claim_count = 0

    issues: list[str] = []

    for claim in case.claims:
        correct_ids = set(
            case.support_map[claim.claim_id]
        )

        claim_has_correct_citation = False
        seen_citation_ids: set[str] = set()

        if not claim.citation_ids:
            issues.append(
                f"{claim.claim_id} 没有提供任何引用。"
            )

        for citation_id in claim.citation_ids:
            citation_count += 1

            if citation_id in seen_citation_ids:
                issues.append(
                    f"{claim.claim_id} 重复引用了 {citation_id}。"
                )

            seen_citation_ids.add(citation_id)

            # 第一层检查：引用指向的证据是否存在。
            if citation_id not in available_ids:
                issues.append(
                    f"{claim.claim_id} 引用了不存在的证据 "
                    f"{citation_id}。"
                )
                continue

            valid_citation_count += 1

            # 第二层检查：该证据是否真的支持当前声明。
            if citation_id in correct_ids:
                correct_citation_count += 1
                claim_has_correct_citation = True
            else:
                issues.append(
                    f"{claim.claim_id} 的引用 {citation_id} "
                    f"存在，但不能支持该声明。"
                )

        if claim_has_correct_citation:
            covered_claim_count += 1
        else:
            issues.append(
                f"{claim.claim_id} 没有得到正确证据支持。"
            )

    claim_count = len(case.claims)

    citation_validity = calculate_ratio(
        valid_citation_count,
        citation_count,
    )

    citation_correctness = calculate_ratio(
        correct_citation_count,
        citation_count,
    )

    claim_coverage = calculate_ratio(
        covered_claim_count,
        claim_count,
    )

    passed = (
        citation_validity == 1.0
        and citation_correctness == 1.0
        and claim_coverage == 1.0
        and not issues
    )

    return CitationMetrics(
        case_id=case.case_id,
        citation_count=citation_count,
        valid_citation_count=valid_citation_count,
        correct_citation_count=correct_citation_count,
        claim_count=claim_count,
        covered_claim_count=covered_claim_count,
        citation_validity=citation_validity,
        citation_correctness=citation_correctness,
        claim_coverage=claim_coverage,
        passed=passed,
        issues=issues,
    )


def print_report(result: CitationMetrics) -> None:
    print(f"\n案例：{result.case_id}")
    print(f"是否通过：{result.passed}")
    print(
        f"引用有效率：{result.citation_validity:.1%}"
    )
    print(
        f"引用正确率：{result.citation_correctness:.1%}"
    )
    print(
        f"声明覆盖率：{result.claim_coverage:.1%}"
    )

    if result.issues:
        print("发现的问题：")

        for issue in result.issues:
            print(f"- {issue}")
    else:
        print("没有发现引用问题。")


evidence = [
    Evidence(
        evidence_id="e1",
        source="paper.pdf",
        page=3,
        text="本文采用四阶龙格—库塔方法求解非线性动力系统。",
    ),
    Evidence(
        evidence_id="e2",
        source="paper.pdf",
        page=8,
        text="数值结果表明，在参数增大后系统进入混沌状态。",
    ),
    Evidence(
        evidence_id="e3",
        source="paper.pdf",
        page=12,
        text="所有实验均在配备GPU的工作站上完成。",
    ),
    Evidence(
        evidence_id="e4",
        source="paper.pdf",
        page=10,
        text="在加入噪声后，所提出的方法仍保持稳定。",
    ),
]


# 正常案例：每条声明都引用了正确证据。
normal_case = CitationEvalCase(
    case_id="normal_case",
    question="论文使用了什么数值方法？得出了什么结论？",
    available_evidence=evidence,
    claims=[
        AnswerClaim(
            claim_id="claim_1",
            text="论文使用四阶龙格—库塔方法。",
            citation_ids=["e1"],
        ),
        AnswerClaim(
            claim_id="claim_2",
            text="参数增大后系统进入混沌状态。",
            citation_ids=["e2"],
        ),
    ],
    support_map={
        "claim_1": ["e1"],
        "claim_2": ["e2"],
    },
)


# 失败案例1：引用了根本不存在的证据。
missing_evidence_case = CitationEvalCase(
    case_id="missing_evidence_case",
    question="论文使用了什么数值方法？",
    available_evidence=evidence,
    claims=[
        AnswerClaim(
            claim_id="claim_1",
            text="论文使用四阶龙格—库塔方法。",
            citation_ids=["e99"],
        ),
    ],
    support_map={
        "claim_1": ["e1"],
    },
)


# 失败案例2：证据存在，但与声明内容无关。
wrong_evidence_case = CitationEvalCase(
    case_id="wrong_evidence_case",
    question="论文使用了什么数值方法？",
    available_evidence=evidence,
    claims=[
        AnswerClaim(
            claim_id="claim_1",
            text="论文使用四阶龙格—库塔方法。",
            citation_ids=["e3"],
        ),
    ],
    support_map={
        "claim_1": ["e1"],
    },
)


# 失败案例3：一条声明引用正确，另一条没有引用。
partial_coverage_case = CitationEvalCase(
    case_id="partial_coverage_case",
    question="论文使用了什么数值方法？得出了什么结论？",
    available_evidence=evidence,
    claims=[
        AnswerClaim(
            claim_id="claim_1",
            text="论文使用四阶龙格—库塔方法。",
            citation_ids=["e1"],
        ),
        AnswerClaim(
            claim_id="claim_2",
            text="参数增大后系统进入混沌状态。",
            citation_ids=[],
        ),
    ],
    support_map={
        "claim_1": ["e1"],
        "claim_2": ["e2"],
    },
)


exercise_case = CitationEvalCase(
    case_id="exercise_case",
    question="论文所提出的方法在噪声环境下是否稳定？",
    available_evidence=evidence,
    claims=[
        AnswerClaim(
            claim_id="claim_3",
            text="所提出的方法在噪声环境下仍然稳定。",

            # 第一次先故意引用错误证据 e3
            citation_ids=["e3"],
        ),
    ],
    support_map={
        # 黄金标注表示 claim_3 实际应该由 e4 支持
        "claim_3": ["e4"],
    },
)


cases = [
    normal_case,
    missing_evidence_case,
    wrong_evidence_case,
    partial_coverage_case,
    exercise_case,
]

for evaluation_case in cases:
    result = evaluate_citations(evaluation_case)
    print_report(result)


# 数据集自身错误：
# 黄金标注引用了不存在的证据，这不是 Agent 的错误，
# 而是评估数据集本身写错了。
print("\n测试错误的黄金标注：")

try:
    CitationEvalCase(
        case_id="invalid_gold_case",
        question="论文使用了什么方法？",
        available_evidence=evidence,
        claims=[
            AnswerClaim(
                claim_id="claim_1",
                text="论文使用四阶龙格—库塔方法。",
                citation_ids=["e1"],
            ),
        ],
        support_map={
            "claim_1": ["e100"],
        },
    )
except ValidationError as error:
    print("评估数据集验证失败：")

    for item in error.errors():
        print(item["msg"])



# 这套评估能分别定位三类问题：
# 引用有效率低
# → Agent 编造了证据 ID
# → 检查引用格式、证据编号传递和提示词
#
# 引用正确率低
# → Agent 引用了不相关证据
# → 检查回答生成、证据理解或上下文噪声
#
# 声明覆盖率低
# → Agent 给出了没有来源的结论
# → 要求每个关键事实都附带证据
# 上一节和这一节组合起来，便形成了完整诊断：
# 检索评估
# → 系统有没有找到正确证据？
#
# 引用与证据评估
# → Agent 有没有正确使用找到的证据？