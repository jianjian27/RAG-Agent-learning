# 学习目标：
# 1. 理解Precision@K、Recall@K、Hit Rate@K和MRR@K。
# 2. 使用参考相关Chunk评估检索结果。
# 3. 区分检索覆盖不足、噪声过多和排序靠后。
# 4. 计算多条查询的宏平均指标。

"""
2.5 检索评估指标

学习目标：
1. 理解Precision@K、Recall@K、Hit Rate@K和MRR@K。
2. 使用参考相关Chunk评估检索结果。
3. 区分检索覆盖不足、噪声过多和排序靠后。
4. 计算多条查询的宏平均指标。

本程序使用模拟Chunk ID，不调用真实检索服务。
"""

from math import isclose
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class EvaluationModel(BaseModel):
    """本节所有评估模型共用的配置。"""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
    )


class RetrievalEvalCase(EvaluationModel):
    """一条检索评估案例。"""

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)

    # 人工标注的、回答问题需要用到的Chunk。
    relevant_ids: list[str]

    # 检索器按照相关性从高到低返回的Chunk。
    retrieved_ids: list[str]

    # 只评估前K条。
    k: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        if not self.relevant_ids:
            raise ValueError(
                "本节的检索案例必须至少有一个相关Chunk。"
            )

        if len(set(self.relevant_ids)) != len(self.relevant_ids):
            raise ValueError(
                "relevant_ids中不允许出现重复Chunk。"
            )

        if len(set(self.retrieved_ids)) != len(self.retrieved_ids):
            raise ValueError(
                "retrieved_ids中不允许出现重复Chunk。"
            )

        return self


class RetrievalMetrics(EvaluationModel):
    """一条案例的检索评估结果。"""

    case_id: str
    k: int

    relevant_count: int = Field(ge=1)
    retrieved_at_k: int = Field(ge=0)
    hit_count: int = Field(ge=0)

    precision_at_k: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    hit_rate_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank_at_k: float = Field(ge=0.0, le=1.0)


def evaluate_retrieval(
    case: RetrievalEvalCase,
) -> RetrievalMetrics:
    """计算一条查询的四个检索指标。"""

    top_k_ids = case.retrieved_ids[:case.k]
    relevant_set = set(case.relevant_ids)

    hit_count = 0

    for chunk_id in top_k_ids:
        if chunk_id in relevant_set:
            hit_count += 1

    # 本节采用标准Precision@K：
    # 即使实际返回数量少于K，分母仍然是K。
    precision_at_k = hit_count / case.k

    recall_at_k = (
        hit_count / len(relevant_set)
    )

    if hit_count > 0:
        hit_rate_at_k = 1.0
    else:
        hit_rate_at_k = 0.0

    reciprocal_rank_at_k = 0.0

    for rank, chunk_id in enumerate(
        top_k_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            reciprocal_rank_at_k = 1 / rank
            break

    return RetrievalMetrics(
        case_id=case.case_id,
        k=case.k,
        relevant_count=len(relevant_set),
        retrieved_at_k=len(top_k_ids),
        hit_count=hit_count,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        hit_rate_at_k=hit_rate_at_k,
        reciprocal_rank_at_k=reciprocal_rank_at_k,
    )


def build_cases() -> list[RetrievalEvalCase]:
    """建立正常与失败检索案例。"""

    return [
        RetrievalEvalCase(
            case_id="excellent_retrieval",
            query="同步控制的稳定条件是什么？",
            relevant_ids=[
                "chunk_001",
                "chunk_004",
            ],
            retrieved_ids=[
                "chunk_001",
                "chunk_002",
                "chunk_004",
                "chunk_009",
                "chunk_010",
            ],
            k=3,
        ),
        RetrievalEvalCase(
            case_id="no_hit_at_k",
            query="同步控制的稳定条件是什么？",
            relevant_ids=[
                "chunk_001",
            ],
            retrieved_ids=[
                "chunk_002",
                "chunk_003",
                "chunk_004",
                "chunk_005",
                "chunk_001",
            ],
            k=3,
        ),
        RetrievalEvalCase(
            case_id="late_hit",
            query="同步控制的稳定条件是什么？",
            relevant_ids=[
                "chunk_001",
            ],
            retrieved_ids=[
                "chunk_002",
                "chunk_003",
                "chunk_001",
                "chunk_008",
            ],
            k=3,
        ),
        RetrievalEvalCase(
            case_id="partial_recall",
            query="论文给出了哪些同步条件？",
            relevant_ids=[
                "chunk_001",
                "chunk_004",
                "chunk_007",
            ],
            retrieved_ids=[
                "chunk_002",
                "chunk_004",
                "chunk_003",
                "chunk_008",
                "chunk_009",
            ],
            k=3,
        ),
        RetrievalEvalCase(
            case_id="empty_retrieval",
            query="同步控制的稳定条件是什么？",
            relevant_ids=[
                "chunk_001",
            ],
            retrieved_ids=[],
            k=3,
        ),
    ]


def run_evaluation(
    cases: list[RetrievalEvalCase],
) -> list[RetrievalMetrics]:
    """评估所有检索案例。"""

    if not cases:
        raise ValueError("检索评估案例不能为空。")

    results: list[RetrievalMetrics] = []

    for case in cases:
        result = evaluate_retrieval(case)
        results.append(result)

    return results


def print_result(result: RetrievalMetrics) -> None:
    """打印一条案例的指标。"""

    print(
        f"{result.case_id}，K={result.k}："
    )
    print(
        f"  命中数量："
        f"{result.hit_count}/"
        f"{result.relevant_count}"
    )
    print(
        f"  实际返回数量："
        f"{result.retrieved_at_k}"
    )
    print(
        f"  Precision@{result.k}："
        f"{result.precision_at_k:.1%}"
    )
    print(
        f"  Recall@{result.k}："
        f"{result.recall_at_k:.1%}"
    )
    print(
        f"  Hit Rate@{result.k}："
        f"{result.hit_rate_at_k:.1%}"
    )
    print(
        f"  MRR@{result.k}："
        f"{result.reciprocal_rank_at_k:.3f}"
    )


def print_report(
    results: list[RetrievalMetrics],
) -> None:
    """打印逐条结果和宏平均指标。"""

    if not results:
        raise ValueError("检索评估结果不能为空。")

    precision_total = 0.0
    recall_total = 0.0
    hit_rate_total = 0.0
    reciprocal_rank_total = 0.0

    for result in results:
        print_result(result)
        print()

        precision_total += result.precision_at_k
        recall_total += result.recall_at_k
        hit_rate_total += result.hit_rate_at_k
        reciprocal_rank_total += (
            result.reciprocal_rank_at_k
        )

    case_count = len(results)

    print("宏平均指标：")
    print(
        f"  平均Precision@K："
        f"{precision_total / case_count:.1%}"
    )
    print(
        f"  平均Recall@K："
        f"{recall_total / case_count:.1%}"
    )
    print(
        f"  平均Hit Rate@K："
        f"{hit_rate_total / case_count:.1%}"
    )
    print(
        f"  平均MRR@K："
        f"{reciprocal_rank_total / case_count:.3f}"
    )


def check_metric(
    actual: float,
    expected: float,
    *,
    name: str,
) -> None:
    """检查一个浮点指标是否符合人工计算结果。"""

    assert isclose(
        actual,
        expected,
        abs_tol=1e-9,
    ), (
        f"{name}计算错误："
        f"期望{expected}，实际{actual}"
    )


def check_evaluator(
    results: list[RetrievalMetrics],
) -> None:
    """用人工计算的结果检查指标实现。"""

    result_map = {
        result.case_id: result
        for result in results
    }

    excellent = result_map["excellent_retrieval"]

    check_metric(
        excellent.precision_at_k,
        2 / 3,
        name="excellent Precision@3",
    )
    check_metric(
        excellent.recall_at_k,
        1.0,
        name="excellent Recall@3",
    )
    check_metric(
        excellent.hit_rate_at_k,
        1.0,
        name="excellent Hit Rate@3",
    )
    check_metric(
        excellent.reciprocal_rank_at_k,
        1.0,
        name="excellent MRR@3",
    )

    no_hit = result_map["no_hit_at_k"]

    check_metric(
        no_hit.precision_at_k,
        0.0,
        name="no_hit Precision@3",
    )
    check_metric(
        no_hit.recall_at_k,
        0.0,
        name="no_hit Recall@3",
    )
    check_metric(
        no_hit.hit_rate_at_k,
        0.0,
        name="no_hit Hit Rate@3",
    )
    check_metric(
        no_hit.reciprocal_rank_at_k,
        0.0,
        name="no_hit MRR@3",
    )

    late_hit = result_map["late_hit"]

    check_metric(
        late_hit.precision_at_k,
        1 / 3,
        name="late_hit Precision@3",
    )
    check_metric(
        late_hit.recall_at_k,
        1.0,
        name="late_hit Recall@3",
    )
    check_metric(
        late_hit.hit_rate_at_k,
        1.0,
        name="late_hit Hit Rate@3",
    )
    check_metric(
        late_hit.reciprocal_rank_at_k,
        1 / 3,
        name="late_hit MRR@3",
    )

    print("\n指标检查通过：计算结果符合人工预期。")


def show_invalid_case() -> None:
    """确认重复检索结果会被拒绝。"""

    print("\n失败案例：检索结果包含重复Chunk。")

    try:
        RetrievalEvalCase(
            case_id="duplicate_result",
            query="同步条件是什么？",
            relevant_ids=[
                "chunk_001",
            ],
            retrieved_ids=[
                "chunk_001",
                "chunk_001",
            ],
            k=2,
        )
    except ValueError as error:
        print(f"已识别错误：{error}")
    else:
        raise AssertionError(
            "重复的检索结果应该被拒绝。"
        )


def main() -> None:
    cases = build_cases()

    results = run_evaluation(cases)

    print_report(results)

    check_evaluator(results)

    show_invalid_case()


if __name__ == "__main__":
    main()


# 召回率是满分，只说明需要的内容找到了；精确率和 MRR 告诉我们同时存在较多噪声，而且正确结果排得偏后。
#
# 不同任务应关注不同指标：
# - 一个证据就足以回答时，Hit Rate 和 MRR 很重要。
# - 必须综合多个章节时，Recall 更重要。
# - 上下文窗口有限时，Precision 很重要，因为无关 Chunk 会消耗 Token，并可能干扰模型。
# - 希望用户尽快看到相关结果时，MRR 很重要。
# K 也会影响指标。增大 K 通常更容易提高召回率和命中率，但也可能降低精确率、增加上下文噪声和 Token 成本。
# 评估时必须记录具体的 K，不能比较一个系统的 Recall@3 和另一个系统的 Recall@10 后直接下结论。



# RAG 检索评估指标，用于评价检索器能不能找回正确证据、能不能把证据排在前面。
#
# 它们只能评价 Agent 的“检索环节”，不能完整评价整个 Agent。完整的 Agent 还要评估工具调用、答案正确性、引用可靠性、任务完成率、成本和延迟等。
#
# 先统一一个例子。假设某个问题共有 3 个相关文本块：
# relevant_ids = {"A", "C", "E"}
# 检索器返回的前 5 名是：
# 第1名：A  相关
# 第2名：B  无关
# 第3名：D  无关
# 第4名：C  相关
# 第5名：F  无关

# 1. Precision@K：前 K 个结果有多“准”
# Precision@K 计算：
# 前 K 个结果中的相关结果数
# ──────────────────────
#             K
# 只看前 3 名：
# A：相关
# B：无关
# D：无关
# 所以：
# Precision@3 = 1 / 3 ≈ 0.333
# 它回答的问题是：
# 检索器交给 Agent 的这些材料里，有多少是真正有用的？
#
# Precision 高，意味着无关内容较少，能减少上下文污染和 Token 浪费。
#
# 但它不关心是否找全。即使只找到了一个相关证据，只要返回的结果很少且都正确，Precision 也可能很高。

# 2. Recall@K：相关结果找回了多少
# Recall@K 计算：
# 前 K 个结果中的相关结果数
# ──────────────────────
#     所有相关结果的数量
# 这个例子中一共有 3 个相关结果：
# A、C、E
# 前 3 名只找到了 A，所以：
# Recall@3 = 1 / 3 ≈ 0.333
# 注意：你选中的：
# Recall@3 = 1 / 1
# 成立的前提是该案例总共只有一个相关结果，而它确实出现在了前 3 名中。
#
# Recall 回答的是：
# 所有应该找到的证据中，检索器找回了多少？
#
# 在论文 RAG 中，如果回答一个问题必须综合多段证据，那么 Recall 非常重要。漏掉关键公式、实验结果或限制条件，都可能导致答案不完整。
#
# Precision 和 Recall 往往需要平衡：
# - 只返回少量最确定的结果：Precision 可能较高，Recall 可能较低。
# - 返回大量结果：Recall 可能提高，Precision 可能下降。

# 3. Hit Rate@K：前 K 名有没有命中
# 单个查询的 Hit@K 只有两个结果：
# 前 K 名至少有一个相关结果 → 1
# 前 K 名没有相关结果       → 0
# 在这个例子中，`A` 出现在前 3 名，因此：
# Hit@3 = 1
# 它回答的是：
# 检索器至少找到了一条有用证据吗？
#
# 多个查询的 Hit Rate@K 是所有查询命中值的平均：
# 命中的查询数量
# ──────────────
#   查询总数量
# 例如 10 个查询中有 8 个在前 3 名命中：
# Hit Rate@3 = 8 / 10 = 0.8
# Hit Rate 比较宽松。无论找到 1 个还是 10 个相关结果，它都只记为 1，也不关心相关结果排在第 1 还是第 K。

# 4. MRR@K：第一个相关结果来得有多早
# 对于单个查询，先计算 Reciprocal Rank：
# RR@K = 1 / 第一个相关结果的排名
# 如果前 K 名没有相关结果：
# RR@K = 0
# 这个例子中，第一个相关结果 A 排第 1：
# RR@3 = 1 / 1 = 1
# 如果结果是：
# 第1名：无关
# 第2名：无关
# 第3名：相关
# 那么：
# RR@3 = 1 / 3 ≈ 0.333
# 对多个查询的 RR 求平均，才得到 MRR：
# MRR@K = 所有查询的 RR@K 之和 / 查询数量
# 它回答的是：
# 用户通常要看到第几个结果，才能遇到第一条正确证据？
#
# MRR只关心第一个相关结果。第一名已经相关后，第二、第三名是否相关都不会改变分数。

# 5. MAP@K：多个相关结果整体排得怎么样
# MAP 是 Mean Average Precision，中文叫“平均准确率均值”。
#
# 它比 MRR 更全面：
# - MRR只看第一个相关结果；
# - MAP会检查多个相关结果分别排在什么位置。
# 先对一个查询计算 AP@K。每遇到一个相关结果，就计算当前位置的 Precision，最后对这些值求平均。
#
# 仍用前面的前 5 名：
# 第1名：A  相关
# 第2名：B  无关
# 第3名：D  无关
# 第4名：C  相关
# 第5名：F  无关
# 相关结果出现于第 1 和第 4 名：
# 第1名处的 Precision@1 = 1 / 1 = 1
# 第4名处的 Precision@4 = 2 / 4 = 0.5
# 如果使用前 5 名中找到的相关结果数量作为简化分母：
# AP@5 = (1 + 0.5) / 2 = 0.75
# 不过更标准的截断版本通常会考虑真实相关结果总数：
# AP@K =
# 各相关结果所在位置的 Precision 之和
# ────────────────────────────────
#        min(相关结果总数, K)
# 这里共有 3 个相关结果，但只找到 2 个：
# AP@5 = (1 + 0.5) / 3 = 0.5
# 然后对所有查询的 AP 求平均：
# MAP@K = 所有查询的 AP@K 之和 / 查询数量
# 它回答的是：
# 检索器是否把多个相关证据都尽量排列在前面？
#
# 在需要综合论文定义、方法、公式和实验结论的 RAG 中，MAP 通常比 MRR包含更多信息。
#
# 需要注意，不同评估库对 AP@K 的截断分母可能有细微差别，因此在报告中最好明确公式。

# 6. nDCG@K：考虑相关程度的分级排名指标
# 前五种指标通常把相关性看成二元结果：
# 相关 = 1
# 无关 = 0
# 但现实中，检索结果的相关程度可能不同：
# 3：高度相关，直接回答问题
# 2：较为相关，提供重要辅助证据
# 1：部分相关，只有少量帮助
# 0：无关
# nDCG 是 Normalized Discounted Cumulative Gain，中文通常叫“归一化折损累计增益”。
#
# 它包含三个思想：
# 1. 相关程度越高，价值越大。
# 2. 相同相关程度的结果，排得越前，价值越大。
# 3. 将实际排序与理想排序比较，将结果归一化到 0～1。
# 例如实际排名的相关等级是：
# [3, 0, 2]
# 理想排名应该是：
# [3, 2, 0]
# nDCG 会对排在后面的结果进行位置折损，常见 DCG 公式是：
# DCG@K = Σ ((2^rel - 1) / log₂(rank + 1))
# 其中：
# - rel 是相关性等级；
# - rank 是排名；
# - log₂(rank + 1) 让靠后的结果获得更低权重。
# 然后计算理想排名的 IDCG@K：
# nDCG@K = DCG@K / IDCG@K
# 结果范围一般是：
# 1：当前排序就是理想排序
# 越接近 0：排序质量越差
# 它回答的是：
# 检索结果是否按照“最有价值的证据在最前面”的顺序排列？
#
# nDCG 很适合存在相关性等级的数据，但代价是数据标注更困难：评估人员不能只标记“相关/无关”，还要判断相关程度。
# 六个指标的区别
# 指标	主要评价什么	是否考虑位置	是否考虑多个相关结果	是否支持分级相关性
# Precision@K	返回结果有多准确	否	是	否
# Recall@K	相关结果找回了多少	否	是	否
# Hit Rate@K	是否至少命中一个	否	否	否
# MRR@K	第一个相关结果是否靠前	是	否	否
# MAP@K	多个相关结果是否都靠前	是	是	通常否
# nDCG@K	高相关结果是否排列在前	是	是	是
#
#
# 不存在一个指标能够代替其他所有指标。
#
# 对于你的论文 RAG 项目，可以优先使用：
# Recall@K
# 检查关键证据是否被找回；
# Precision@K
# 检查返回内容中是否混入太多噪声；
# MRR@K
# 检查第一条有用证据是否足够靠前；
# nDCG@K
# 等以后有了“高度相关、部分相关、无关”的人工标注，再加入评估。
#
# MAP也很有价值，但初期可以先做了解，等评估集中出现“一道问题对应多条相关证据”时再实现。
#
# 所以，这些指标确实是 Agent 评估中的常用指标，但准确表述是：
# 它们是 Agent 系统中检索器或 RAG 检索阶段的常用离线评估指标，而不是对整个 Agent 能力的完整评估。
