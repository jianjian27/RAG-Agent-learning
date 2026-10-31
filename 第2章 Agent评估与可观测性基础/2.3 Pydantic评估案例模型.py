# JSONL 读取器能够识别 JSON 语法错误，但下面这些记录都是合法 JSON：
# {"case_id": 123}
# {"case_id": "case_001", "question": "   "}
# {"case_id": "case_001", "question": "同步条件在哪一页？", "expected": {"status": "answered"}}
# 它们却不能用于可靠评估：编号类型错误、问题为空，或者要求 Agent 定位页码，却没有提供参考页码。
#
# 因此，现在的数据流要增加一层：
# JSONL中的一行文本
# → json.loads()
# → 普通Python字典
# → Pydantic验证
# → EvalCase对象
# → 交给评分器

# 学习目标：
# 1. 使用嵌套模型描述评估案例。
# 2. 检查字段类型、必填项和取值范围。
# 3. 使用模型校验器检查跨字段规则。
# 4. 在加载JSONL时报告案例错误所在的行号。


import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


class EvaluationModel(BaseModel):
    """本节所有数据模型共用的配置。"""

    model_config = ConfigDict(
        extra="forbid",  # 禁止额外字段
        strict=True ,
        str_strip_whitespace=True,  # 字符串字段自动去掉首尾空白
    )


class ExpectedResult(EvaluationModel):
    """人工核对后，为案例提供的参考结果。"""

    status: Literal["answered", "insufficient_evidence"]

    paper_id: str | None = Field(
        default=None,
        min_length=1,
    )

    page: int | None = Field(
        default=None,
        ge=1,
    )

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.status == "answered":
            if self.paper_id is None or self.page is None:
                raise ValueError(
                    "answered状态必须提供paper_id和page。"
                )

        if self.status == "insufficient_evidence":
            if self.paper_id is not None or self.page is not None:
                raise ValueError(
                    "本节约定：证据不足时不填写paper_id和page。"
                )

        return self


class EvalCase(EvaluationModel):
    """一条评估题目，不包含Agent实际输出。"""

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected: ExpectedResult


def load_cases(path: Path) -> list[EvalCase]:
    """读取JSONL，并将每条记录验证为EvalCase对象。"""
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path.name}第{line_number}行：不允许空白行。"
                )

            try:
                raw_case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}第{line_number}行：JSON语法错误。"
                ) from error

            try:
                case = EvalCase.model_validate(raw_case)
            except ValidationError as error:
                raise ValueError(
                    f"{path.name}第{line_number}行："
                    f"案例结构验证失败。\n{error}"
                ) from error

            if case.case_id in seen_ids:
                raise ValueError(
                    f"{path.name}第{line_number}行："
                    f"case_id重复：{case.case_id}"
                )

            seen_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise ValueError("评估数据集不能为空。")

    return cases


def write_demo_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    """将演示字典写成每行一条记录的JSONL。"""
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def show_invalid_case(
    title: str,
    raw_case: dict,
) -> None:
    """验证错误案例，并显示字段位置与错误说明。"""
    print(f"\n{title}")

    try:
        EvalCase.model_validate(raw_case)
    except ValidationError as error:
        for item in error.errors():
            print(f"  位置：{item['loc']}")
            print(f"  原因：{item['msg']}")
    else:
        raise AssertionError("该案例应该验证失败。")


def main() -> None:
    valid_record = {
        "case_id": "case_001",
        "question": "  同步条件位于哪一页？  ",
        "expected": {
            "status": "answered",
            "paper_id": "paper_001",
            "page": 12,
        },
    }

    insufficient_record = {
        "case_id": "case_002",
        "question": "现有材料是否证明该方法适用于所有系统？",
        "expected": {
            "status": "insufficient_evidence",
        },
    }

    print("正常案例：字典转换成嵌套模型。")

    case = EvalCase.model_validate(valid_record)

    print(f"外层类型：{type(case).__name__}")
    print(f"内层类型：{type(case.expected).__name__}")
    print(f"清理后的问题：{case.question}")
    print(f"参考论文：{case.expected.paper_id}")
    print(f"参考页码：{case.expected.page}")

    print("\n转换回普通字典：")
    print(case.model_dump(exclude_none=True))

    show_invalid_case(
        "失败案例一：严格模式拒绝字符串页码。",
        {
            "case_id": "bad_type",
            "question": "同步条件在哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": "12",
            },
        },
    )

    show_invalid_case(
        "失败案例二：answered状态缺少页码。",
        {
            "case_id": "missing_page",
            "question": "同步条件在哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
            },
        },
    )

    show_invalid_case(
        "失败案例三：问题只有空格。",
        {
            "case_id": "empty_question",
            "question": "   ",
            "expected": {
                "status": "insufficient_evidence",
            },
        },
    )

    show_invalid_case(
        "失败案例四：字段名拼错。",
        {
            "case_id": "wrong_field",
            "question": "能否回答这个问题？",
            "expected": {
                "status": "insufficient_evidence",
            },
            "questoin": "这是拼错的字段名。",
        },
    )

    with TemporaryDirectory() as directory:
        data_directory = Path(directory)

        valid_path = data_directory / "cases.jsonl"

        write_demo_jsonl(
            valid_path,
            [valid_record, insufficient_record],
        )

        loaded_cases = load_cases(valid_path)

        print(f"\n正常JSONL：成功加载{len(loaded_cases)}条案例。")

        for loaded_case in loaded_cases:
            print(
                f"{loaded_case.case_id}："
                f"{loaded_case.expected.status}"
            )

        duplicate_path = data_directory / "duplicate.jsonl"

        write_demo_jsonl(
            duplicate_path,
            [valid_record, valid_record],
        )

        print("\n失败案例五：JSONL中案例编号重复。")

        try:
            load_cases(duplicate_path)
        except ValueError as error:
            print(error)
        else:
            raise AssertionError("重复编号应该被拒绝。")

    print("\n所有正常与预期失败案例检查完成。")


if __name__ == "__main__":
    main()









