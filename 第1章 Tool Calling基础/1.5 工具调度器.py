# 工具调用消息 -> 工具名称和JSON参数
# 工具注册表 -> 工具名称对应的函数和参数模型

# 工具调度器负责将这些内容连接起来
# 模型返回工具调用
# → 查找工具
# → 解析参数
# → 验证参数
# → 执行函数
# → 包装工具结果消息

# 如果没有调度器，工具调用代码可能写成：
# if tool_name == "get_weather":
#     arguments = WeatherArguments.model_validate(raw_arguments)
#     result = get_weather(**arguments.model_dump())
#
# elif tool_name == "search_papers":
#     arguments = SearchPapersArguments.model_validate(raw_arguments)
#     result = search_papers(**arguments.model_dump())
#
# elif tool_name == "read_paper":
#     ...
#
# elif tool_name == "save_note":
#     ...
# 工具越来越多以后，这个 if/elif 会不断变长，而且每个分支都在重复：
# - 查找工具；
# - 解析 JSON；
# - 验证参数；
# - 调用函数；
# - 序列化结果；
# - 构造 tool 消息；
# - 处理异常。

# 调度器把这些共同流程集中到一个地方。
# 以后新增工具时，只需要注册：
# tool_registry["new_tool"] = ToolDefinition(...)
# 调度器本身不需要为每个工具增加新的 elif。

# 工具注册表
# 负责保存工具
#
# 工具调度器
# 负责执行一次工具调用流程

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


ToolFunction = Callable[..., object]  # 创建了一个类型别名，后续不用反复写 Callable[..., object]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    function: ToolFunction
    arguments_model: type[BaseModel]  # 表示保存的不是Pydantic模型实例，而是一个继承自 BaseModel 的模型类

# 天气参数：
class WeatherArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    city: str = Field(min_length=1)
    days: int = Field(default=1, ge=1, le=7)
    unit: Literal["celsius", "fahrenheit"] = "celsius"

# 论文搜索参数：
class SearchPapersArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

# 两个工具函数：
def get_weather(
    city: str,
    *,
    days: int = 1,
    unit: str = "celsius",
) -> dict[str, object]:
    return {
        "city": city,
        "days": days,
        "unit": unit,
        "temperature": 26,
        "condition": "晴",
    }


def search_papers(
    query: str,
    *,
    top_k: int = 5,
) -> dict[str, object]:
    papers = [
        "Synchronization of coupled oscillators",
        "Nonlinear dynamics and chaos",
        "Stability analysis of complex networks",
    ]

    return {
        "query": query,
        "top_k": top_k,
        "results": papers[:top_k],
    }

# 完整注册表：
tool_registry: dict[str, ToolDefinition] = {
    "get_weather": ToolDefinition(
        name="get_weather",
        description="查询指定城市未来几天的天气。",
        function=get_weather,
        arguments_model=WeatherArguments,
    ),
    "search_papers": ToolDefinition(
        name="search_papers",
        description="根据查询文本搜索相关论文。",
        function=search_papers,
        arguments_model=SearchPapersArguments,
    ),
}

# 模型返回的单词工具调用类似：
tool_call = {
    "id": "call_weather_001",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": (
            '{"city":"北京",'
            '"days":3,'
            '"unit":"celsius"}'
        ),
    },
}

# 调度器从中获得：
# call_id = tool_call["id"] # 用于把结果关联回这次工具调用
# tool_name = tool_call["function"]["name"] # 用于从注册表查找工具
# arguments_json = tool_call["function"]["arguments"] # 模型生成的JSON参数字符串


# 实现仅处理正常情况的调度器：
def dispatch_tool_call(tool_call: dict) -> dict[str, str]:
    call_id = tool_call["id"]
    tool_name = tool_call["function"]["name"]
    arguments_json = tool_call["function"]["arguments"]

    try:
        tool = tool_registry[tool_name]
    except KeyError:
        raise ValueError(
            f"未知工具：{tool_name}"
        ) from None

    raw_arguments = json.loads(arguments_json)

    validated_arguments = tool.arguments_model.model_validate(
        raw_arguments,
        strict=True,
    )

    result = tool.function(
        **validated_arguments.model_dump()
    )

    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(
            result,
            ensure_ascii=False,
        ),
    }

# 测试：
weather_call = {
    "id": "call_weather_001",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": (
            '{"city":"北京",'
            '"days":3,'
            '"unit":"celsius"}' # json字符串
        ),
    },
}

weather_message = dispatch_tool_call(weather_call)
print(weather_message)
# {'role': 'tool', 'tool_call_id': 'call_weather_001', 'content': '{"city": "北京", "days": 3, "unit": "celsius", "temperature": 26, "condition": "晴"}'}

# 调度器返回的不是工具函数原始结果，而是一条能够加入messages的完整tool消息
# 外部代码只需要：
# messages.append(
#     dispatch_tool_call(tool_call)
# )
# 调度器承担“协议数据”和“Python函数”之间的翻译工作

# 真实 Agent 不应该因为模型生成了一次错误参数就直接退出整个进程。
# 可以建立辅助函数：
def build_tool_message(
        call_id: str,
        payload: dict[str, object],
) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(
            payload,
            ensure_ascii=False,
        ),
    }

def dispatch_tool_call(tool_call: dict) -> dict[str, str]:
    call_id = tool_call["id"]
    tool_name = tool_call["function"]["name"]
    arguments_json = tool_call["function"]["arguments"]

    tool = tool_registry.get(tool_name)

    if tool is None:
        return build_tool_message(
            call_id,
            {
                "ok": False,
                "error": {
                    "type": "unknown_tool",
                    "message": f"未知工具：{tool_name}",
                },
            },
        )

    try:
        raw_arguments = json.loads(arguments_json)
    except json.JSONDecodeError as error:
        return build_tool_message(
            call_id,
            {
                "ok": False,
                "error": {
                    "type": "invalid_json",
                    "message": str(error),
                },
            },
        )

    try:
        validated_arguments = (
            tool.arguments_model.model_validate(
                raw_arguments,
                strict=True,
            )
        )
    except ValidationError as error:
        return build_tool_message(
            call_id,
            {
                "ok": False,
                "error": {
                    "type": "validation_error",
                    "details": error.errors(
                        include_url=False
                    ),
                },
            },
        )

    result = tool.function(
        **validated_arguments.model_dump()
    )

    return build_tool_message(
        call_id,
        {
            "ok": True,
            "result": result,
        },
    )
# 现在调度器不会因为常见的模型输入错误而终止，而是返回一条正常的 tool 消息。
# 模型可以看到错误内容，然后决定：
# - 修改参数后重新调用；
# - 改用另一个工具；
# - 向用户询问缺失信息；
# - 告诉用户工具无法执行。

# 不建议写：
# except Exception as error:
#     return ...
# 因为这样可能把程序自身的缺陷也隐藏起来。
# 例如工具内部写错变量名：
# return unknow_variable
# 会产生 NameError。如果捕获所有 Exception，程序可能把真正的代码错误伪装成普通工具失败，使调试非常困难。
# 当前只处理明确预计的错误：
# KeyError
# → 工具没有注册
#
# JSONDecodeError
# → 模型参数不是合法JSON
#
# ValidationError
# → 参数不符合Pydantic模型

# 正常天气调用：
valid_weather_call = {
    "id": "call_001",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": (
            '{"city":"北京",'
            '"days":3,'
            '"unit":"celsius"}'
        ),
    },
}

# 正常论文搜索：
valid_search_call = {
    "id": "call_002",
    "type": "function",
    "function": {
        "name": "search_papers",
        "arguments": (
            '{"query":"nonlinear dynamics",'
            '"top_k":2}'
        ),
    },
}

# 未知工具：
unknown_tool_call = {
    "id": "call_003",
    "type": "function",
    "function": {
        "name": "delete_database",
        "arguments": "{}",
    },
}

# 非法参数：
invalid_arguments_call = {
    "id": "call_004",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": (
            '{"city":"北京",'
            '"days":30,'
            '"unit":"kelvin"}'
        ),
    },
}

# 非法JSON：
invalid_json_call = {
    "id": "call_005",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"city":"北京"',
    },
}

# 依次调度：
test_calls = [
    valid_weather_call,
    valid_search_call,
    unknown_tool_call,
    invalid_arguments_call,
    invalid_json_call,
]

for test_call in test_calls:
    tool_message = dispatch_tool_call(test_call)

    print(tool_message["tool_call_id"])

    content = json.loads(tool_message["content"])

    print(
        json.dumps(
            content,
            ensure_ascii=False,
            indent=2,
        )
    )
