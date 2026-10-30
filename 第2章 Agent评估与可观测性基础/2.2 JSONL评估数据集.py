# 学习目标：
# 1. 理解JSON与JSONL的区别。
# 2. 把评估案例写入JSONL文件。
# 3. 逐行读取，并报告格式错误所在的行号。
# 4. 将题目、参考结果与Agent实际输出分开保存。


# 把案例直接写在函数里：
# cases = [
#     {...},
#     {...},
# ]
# 案例少时很方便。但当案例增加到几十、几百个时，修改题目也要修改程序，题目还容易和某次 Agent 的回答混在一起。
#
# 我们希望分开保存：
# 评估数据集
# → 用户问题、参考结果
#
# Agent运行结果
# → 本次实际回答
#
# 评分器
# → 根据规则比较二者
# 例如，同一道题可以分别交给旧版、新版 Agent。参考结果保持一致，才能公平比较。

# JSONL，全称 JSON Lines，是一种文本文件格式：每一行都是一个独立、完整的 JSON 值。
# 它使用 UTF-8 编码，通常以 .jsonl 为扩展名，空白行不是合法记录。
# 普通 JSON 可以把全部案例放进一个数组：
# [
#   {"case_id": "case_001", "question": "问题一"},
#   {"case_id": "case_002", "question": "问题二"}
# ]
# JSONL 则写成：
# {"case_id": "case_001", "question": "问题一"}
# {"case_id": "case_002", "question": "问题二"}
# 注意三点：
# - 文件外层没有数组括号。
# - 两条记录之间没有逗号。
# - 每条记录必须在一个物理行内，不能用 indent=2 将它展开为多行。
# 如果问题中包含换行，JSON 会将它编码为 \n，整条记录仍然可以保存在同一行。
#
# JSONL 适合评估数据，因为每行可以独立读取和检查；
# 新增一道题通常只增加一行；格式错误也能定位到具体行。它同样常用于保存运行记录。


import json
from pathlib import Path
from tempfile import TemporaryDirectory


def build_cases() -> list[dict]:
    """准备两道评估题，不包含某次运行的actual结果。"""
    return [
        {
            "case_id": "case_001",
            "question": "论文paper_001中的同步条件位于哪一页？",
            "expected": {
                "status": "answered",
                "paper_id": "paper_001",
                "page": 12,
            },
        },
        {
            "case_id": "case_002",
            "question": "现有材料是否证明该方法适用于所有非线性系统？",
            "expected": {
                "status": "insufficient_evidence",
            },
        },
    ]


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    """将字典列表写入JSONL文件，每行一条记录。"""
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            line = json.dumps(
                record,
                ensure_ascii=False,
            )

            file.write(line + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """
    逐行读取JSONL文件。

    JSONL本身允许一行是数组、数字等JSON值。
    这里额外规定：每条评估记录必须是JSON对象，
    解析到Python后应为dict。
    """
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(
                    f"{path.name}第{line_number}行：不允许空白行。"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name}第{line_number}行："
                    f"不是合法JSON，{error.msg}。"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"{path.name}第{line_number}行："
                    "评估记录必须是JSON对象。"
                )

                records.append(record)

        return records


def show_failure(path: Path) -> None:
    """尝试读取错误文件，并打印预期的失败信息。"""
    try:
        read_jsonl(path)
    except ValueError as error:
        print(f"已识别错误：{error}")
    else:
        raise AssertionError(
            f"{path.name}本应读取失败，但没有失败。"
        )


def main() -> None:
    cases = build_cases()

    with TemporaryDirectory() as directory:
        data_directory = Path(directory)

        # 正常案例：写入后再读取。
        valid_path = data_directory / "cases.jsonl"

        write_jsonl(valid_path, cases)

        print("正常文件内容：")
        print(valid_path.read_text(encoding="utf-8"))

        loaded_cases = read_jsonl(valid_path)

        assert loaded_cases == cases, "读写后的数据不一致。"

        print(f"成功读取{len(loaded_cases)}条案例。")

        for case in loaded_cases:
            print(
                f"{case['case_id']}："
                f"{case['question']}"
            )

        # 准备一行合法记录，供下面的失败案例使用。
        valid_line = json.dumps(
            cases[0],
            ensure_ascii=False,
        )

        print("\n失败案例一：第二行JSON不完整。")

        invalid_json_path = data_directory / "invalid_json.jsonl"

        invalid_json_path.write_text(
            valid_line + "\n"
            + '{"case_id": "broken"\n',
            encoding="utf-8",
        )

        show_failure(invalid_json_path)

        print("\n失败案例二：一行是数组，而不是案例对象。")

        wrong_type_path = data_directory / "wrong_type.jsonl"

        wrong_type_path.write_text(
            '["不是", "案例对象"]\n',
            encoding="utf-8",
        )

        show_failure(wrong_type_path)

        print("\n失败案例三：第二行为空白行。")

        blank_line_path = data_directory / "blank_line.jsonl"

        blank_line_path.write_text(
            valid_line + "\n\n",
            encoding="utf-8",
        )

        show_failure(blank_line_path)

        print("\n所有正常案例与预期失败案例检查完成。")


if __name__ == "__main__":
    main()
