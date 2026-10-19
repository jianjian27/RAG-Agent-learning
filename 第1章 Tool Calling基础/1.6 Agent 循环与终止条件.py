# Agent 循环,是重复执行以下决策：
# 调用模型
#    ↓
# 模型请求工具？
#    ├── 是 → 调度工具 → 记录结果 → 再次调用模型
#    └── 否 → 取得最终回答 → 结束
# 需要循环是因为程序事先不知道解决任务需要调用几次工具。

# 三个组件的分工：
#  组件	                  职责
# 模型函数	   根据对话历史，生成下一条回复
# 工具调度器	   处理一次工具调用，返回 tool 消息
# Agent 循环	   组织消息历史、调用模型、执行工具、决定结束

# for step in range(max_steps):
#     assistant_message = model(messages)
#     messages.append(assistant_message)
#
#     tool_calls = assistant_message.get("tool_calls") or [] # 外层循环，模型思考一轮，可能提出新的工具调用也可能没有
#
#     if not tool_calls:
#         return assistant_message["content"]
#
#     for tool_call in tool_calls: # 内层循环，执行当前这一轮提出的所有工具调用
#         tool_message = dispatch_tool_call(tool_call)
#         messages.append(tool_message)

import json
from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError


ToolFunction = Callable[..., object]

ModelFunction = Callable[[list[dict]], dict]
ToolDispatcher = Callable[[dict], dict[str, str]]


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

tool_registry: dict[str, ToolDefinition] = {
    "get_weather": ToolDefinition(
        name="get_weather",
        description="查询指定城市未来几天的天气。",
        function=get_weather,
        arguments_model=WeatherArguments,
    ),
}

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


def run_agent(
    user_message: str,
    *,
    model: ModelFunction,
    dispatch: ToolDispatcher,
    max_steps: int = 5,
) -> str:
    if not user_message.strip():
        raise ValueError("用户消息不能为空")

    if max_steps <= 0:
        raise ValueError("max_steps必须大于0")

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个工具助手。"
                "需要外部信息时使用工具，"
                "根据工具结果回答用户。"
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    for step in range(1, max_steps + 1):
        print(f"\n第{step}轮： 调用模型")

        assistant_message = model(messages)

        # 先保留模型的原始回复，包括工具调用请求
        messages.append(assistant_message)

        tool_calls = (
            assistant_message.get("tool_calls") or []
        )

        if not tool_calls:
            content = assistant_message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(
                    "模型既没有请求工具，也没有提供有效回答。"
                )

            print("模型给出了最终回答，结束循环。")
            return content

        # 最后一轮也没有得到最终答案时，停止
        # 不再执行无法继续交还模型处理的新工具调用
        if step == max_steps:
            raise RuntimeError(
                f"已达到最大模型调用轮数：{max_steps}，"
                "模型仍在请求工具。"
            )

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            print(f"执行工具：{tool_name}")

            tool_message = dispatch(tool_call)
            messages.append(tool_message)

            print(
                "已记录工具结果："
                f"{tool_message['tool_call_id']}"
            )

    raise RuntimeError("Agent未能正常结束。")

def mock_model(messages: list[dict]) -> dict:
    last_message = messages[-1]

    if last_message["role"] == "user":
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather_001",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps(
                            {
                                "city": "北京",
                                "days": 3,
                                "unit": "celsius",
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        }

    if last_message["role"] == "tool":
        payload = json.loads(last_message["content"])

        if not payload["ok"]:
            return {
                "role": "assistant",
                "content": (
                    "天气查询失败，"
                    "请检查输入或稍后重试。"
                ),
            }

        weather = payload["result"]

        return {
            "role": "assistant",
            "content": (
                f"已查询{weather['city']}"
                f"未来{weather['days']}天的天气。"
                f"模拟数据显示："
                f"{weather['condition']}，"
                f"温度为{weather['temperature']}摄氏度。"
            ),
        }

    raise RuntimeError("模拟模型收到未预期的消息。")

answer = run_agent(
    "北京未来三天天气怎么样？",
    model=mock_model,
    dispatch=dispatch_tool_call,
    max_steps=5,
)

print("\n最终回答：")
print(answer)

def looping_model(messages: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_repeat_{len(messages)}",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"北京"}',
                },
            }
        ],
    }

try:
    run_agent(
        "北京天气怎么样？",
        model=looping_model,
        dispatch=dispatch_tool_call,
        max_steps=3,
    )
except RuntimeError as error:
    print(f"Agent停止：{error}")

def direct_answer_model(messages: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": "你好，我可以帮助你查询天气。",
    }

answer_direct = run_agent(
    "北京未来三天天气怎么样？",
    model=direct_answer_model,
    dispatch=dispatch_tool_call,
    max_steps=5,
)
print(answer_direct)
