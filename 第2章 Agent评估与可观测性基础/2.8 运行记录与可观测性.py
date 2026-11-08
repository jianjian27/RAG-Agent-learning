# 前面的评估代码能告诉我们：
# Agent 最终得分是多少？
# 但如果结果失败，还需要回答：
# 检索失败了吗？
# 模型调用失败了吗？
# 调用顺序是什么？
# 哪一步最慢？
# 消耗了多少 Token？
# 错误发生在哪一步？
# 这就是可观测性。
#
# OpenAI 官方文档把 Trace 描述为 Agent 一次运行中的端到端记录，包括模型调用、工具调用和决策步骤。
# 它的价值不只是判断成功或失败，更是定位 Agent 为什么失败。
#
# 没有 Trace 时，我们只能看到：
# Agent回答失败
# 有 Trace 后，我们可以看到：
# Agent回答失败
# → 检索成功
# → 返回3个Chunk
# → 模型调用失败
# → 错误类型：TimeoutError
# → 模型调用耗时30秒
# 它把“程序坏了”变成了一个可以定位的问题。


import time

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TraceSpan(BaseModel):
    """
    Agent运行中的一个步骤。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    span_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)

    sequence: int = Field(ge=1)

    span_type: Literal[
        "retrieval",
        "tool",
        "model",
        "validation",
    ]

    name: str = Field(min_length=1)

    status: Literal[
        "success",
        "error",
    ]

    started_at: str
    duration_ms: float = Field(ge=0)

    input_summary: dict[str, object]
    output_summary: dict[str, object]

    error_type: str | None = None
    error_message: str | None = None


class AgentRun(BaseModel):
    """
    一次完整的Agent运行记录。
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    run_id: str = Field(min_length=1)

    status: Literal[
        "success",
        "error",
    ]

    started_at: str
    duration_ms: float = Field(ge=0)

    query_summary: dict[str, object]

    spans: list[TraceSpan]

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    final_answer: str | None = None
    error_message: str | None = None


class ModelCallOutput(BaseModel):
    answer: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class RunSummary(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int

    success_rate: float
    average_duration_ms: float

    total_input_tokens: int
    total_output_tokens: int

    failed_spans: list[str]


# 这个类型表示：
# 不接收参数、返回任意object的函数。
ObservedOperation = Callable[[], object]


def utc_now() -> str:
    """
    生成带UTC时区的时间。
    """

    return datetime.now(timezone.utc).isoformat()


def summarize_output(value: object) -> dict[str, object]:
    """
    只记录输出的摘要，不直接记录全部内容。
    """

    if isinstance(value, list):
        return {
            "type": "list",
            "item_count": len(value),
        }

    if isinstance(value, str):
        return {
            "type": "str",
            "character_count": len(value),
        }

    if isinstance(value, BaseModel):
        return {
            "type": type(value).__name__,
            "fields": list(value.model_fields_set),
        }

    return {
        "type": type(value).__name__,
    }


def execute_observed_step(
    *,
    run_id: str,
    sequence: int,
    span_type: Literal[
        "retrieval",
        "tool",
        "model",
        "validation",
    ],
    name: str,
    input_summary: dict[str, object],
    operation: ObservedOperation,
) -> tuple[TraceSpan, object | None]:
    """
    执行一个操作，同时记录耗时、状态和异常。
    """

    span_id = uuid4().hex
    started_at = utc_now()
    start_clock = time.perf_counter()

    try:
        output = operation()

    except Exception as error:
        duration_ms = (
            time.perf_counter() - start_clock
        ) * 1000

        span = TraceSpan(
            span_id=span_id,
            run_id=run_id,
            sequence=sequence,
            span_type=span_type,
            name=name,
            status="error",
            started_at=started_at,
            duration_ms=duration_ms,
            input_summary=input_summary,
            output_summary={},
            error_type=type(error).__name__,
            error_message=str(error),
        )

        return span, None

    duration_ms = (
        time.perf_counter() - start_clock
    ) * 1000

    span = TraceSpan(
        span_id=span_id,
        run_id=run_id,
        sequence=sequence,
        span_type=span_type,
        name=name,
        status="success",
        started_at=started_at,
        duration_ms=duration_ms,
        input_summary=input_summary,
        output_summary=summarize_output(output),
        error_type=None,
        error_message=None,
    )

    return span, output


def search_papers(
    query: str,
    *,
    should_fail: bool,
) -> list[str]:
    """
    模拟论文检索。
    """

    time.sleep(0.02)

    if should_fail:
        raise ConnectionError("向量数据库连接失败。")

    return [
        "Chunk A：论文采用四阶龙格—库塔方法。",
        "Chunk B：参数增大后系统进入混沌状态。",
    ]


def generate_answer(
    query: str,
    chunks: list[str],
) -> ModelCallOutput:
    """
    模拟模型生成回答。
    """

    time.sleep(0.03)

    answer = (
        "论文采用四阶龙格—库塔方法进行数值求解，"
        "并发现参数增大后系统进入混沌状态。"
    )

    return ModelCallOutput(
        answer=answer,
        input_tokens=320,
        output_tokens=48,
    )


def run_agent(
    query: str,
    *,
    fail_retrieval: bool = False,
) -> AgentRun:
    run_id = uuid4().hex
    started_at = utc_now()
    run_start_clock = time.perf_counter()

    spans: list[TraceSpan] = []

    retrieval_span, retrieval_output = (
        execute_observed_step(
            run_id=run_id,
            sequence=1,
            span_type="retrieval",
            name="search_papers",
            input_summary={
                # 不保存完整问题，只保存必要摘要。
                "query_length": len(query),
                "top_k": 2,
            },
            operation=lambda: search_papers(
                query,
                should_fail=fail_retrieval,
            ),
        )
    )

    spans.append(retrieval_span)

    if retrieval_span.status == "error":
        duration_ms = (
            time.perf_counter() - run_start_clock
        ) * 1000

        return AgentRun(
            run_id=run_id,
            status="error",
            started_at=started_at,
            duration_ms=duration_ms,
            query_summary={
                "character_count": len(query),
            },
            spans=spans,
            input_tokens=0,
            output_tokens=0,
            final_answer=None,
            error_message=retrieval_span.error_message,
        )

    if not isinstance(retrieval_output, list):
        raise TypeError("检索结果必须是列表。")

    chunks = [
        str(item)
        for item in retrieval_output
    ]

    model_span, model_output = execute_observed_step(
        run_id=run_id,
        sequence=2,
        span_type="model",
        name="generate_answer",
        input_summary={
            "query_length": len(query),
            "chunk_count": len(chunks),
        },
        operation=lambda: generate_answer(
            query,
            chunks,
        ),
    )

    spans.append(model_span)

    if model_span.status == "error":
        duration_ms = (
            time.perf_counter() - run_start_clock
        ) * 1000

        return AgentRun(
            run_id=run_id,
            status="error",
            started_at=started_at,
            duration_ms=duration_ms,
            query_summary={
                "character_count": len(query),
            },
            spans=spans,
            input_tokens=0,
            output_tokens=0,
            final_answer=None,
            error_message=model_span.error_message,
        )

    if not isinstance(model_output, ModelCallOutput):
        raise TypeError(
            "模型调用必须返回ModelCallOutput。"
        )

    duration_ms = (
        time.perf_counter() - run_start_clock
    ) * 1000

    return AgentRun(
        run_id=run_id,
        status="success",
        started_at=started_at,
        duration_ms=duration_ms,
        query_summary={
            "character_count": len(query),
        },
        spans=spans,
        input_tokens=model_output.input_tokens,
        output_tokens=model_output.output_tokens,
        final_answer=model_output.answer,
        error_message=None,
    )


def save_runs_jsonl(
    path: Path,
    runs: list[AgentRun],
) -> None:
    lines = [
        run.model_dump_json()
        for run in runs
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def load_runs_jsonl(
    path: Path,
) -> list[AgentRun]:
    runs: list[AgentRun] = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        run = AgentRun.model_validate_json(line)
        runs.append(run)

    return runs


def summarize_runs(
    runs: list[AgentRun],
) -> RunSummary:
    total_runs = len(runs)

    successful_runs = sum(
        run.status == "success"
        for run in runs
    )

    failed_runs = total_runs - successful_runs

    if total_runs == 0:
        success_rate = 0.0
        average_duration_ms = 0.0
    else:
        success_rate = (
            successful_runs / total_runs
        )

        average_duration_ms = (
            sum(run.duration_ms for run in runs)
            / total_runs
        )

    failed_spans = [
        f"{run.run_id}:{span.name}:{span.error_type}"
        for run in runs
        for span in run.spans
        if span.status == "error"
    ]

    return RunSummary(
        total_runs=total_runs,
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        success_rate=success_rate,
        average_duration_ms=average_duration_ms,
        total_input_tokens=sum(
            run.input_tokens
            for run in runs
        ),
        total_output_tokens=sum(
            run.output_tokens
            for run in runs
        ),
        failed_spans=failed_spans,
    )


def print_run(run: AgentRun) -> None:
    print(f"\nRun ID：{run.run_id}")
    print(f"运行状态：{run.status}")
    print(f"总耗时：{run.duration_ms:.2f} ms")
    print(
        f"Token：输入 {run.input_tokens}，"
        f"输出 {run.output_tokens}"
    )

    for span in run.spans:
        print(
            f"  {span.sequence}. "
            f"{span.name} "
            f"[{span.status}] "
            f"{span.duration_ms:.2f} ms"
        )

        if span.error_message is not None:
            print(
                f"     错误："
                f"{span.error_type}: "
                f"{span.error_message}"
            )


normal_run = run_agent(
    "论文使用了什么数值方法？"
)

failed_run = run_agent(
    "论文使用了什么数值方法？",
    fail_retrieval=True,
)

runs = [
    normal_run,
    failed_run,
]

for current_run in runs:
    print_run(current_run)


# 使用临时目录演示持久化，不污染学习仓库。
with TemporaryDirectory() as temporary_directory:
    trace_path = (
        Path(temporary_directory)
        / "agent_runs.jsonl"
    )

    save_runs_jsonl(trace_path, runs)

    loaded_runs = load_runs_jsonl(trace_path)

    print(
        f"\n从JSONL重新读取："
        f"{len(loaded_runs)} 条运行记录"
    )

    summary = summarize_runs(loaded_runs)


print("\n可观测性汇总")
print(f"总运行数：{summary.total_runs}")
print(f"成功运行数：{summary.successful_runs}")
print(f"失败运行数：{summary.failed_runs}")
print(f"成功率：{summary.success_rate:.1%}")
print(
    f"平均耗时："
    f"{summary.average_duration_ms:.2f} ms"
)
print(
    f"总输入Token："
    f"{summary.total_input_tokens}"
)
print(
    f"总输出Token："
    f"{summary.total_output_tokens}"
)
print(f"失败Span：{summary.failed_spans}")

