# JSON / JSON Schema 工具参数怎样描述和传输
# Pydantic 工具参数怎样验证
# 工具注册表 怎样根据名称找到Python函数

# 用户提出问题
# → 模型决定调用工具
# → Python执行工具
# → 把工具结果交还模型
# → 模型生成最终回答

# Tool Calling 的本质
# 大模型本身不会执行你本地的 Python 函数。
# 当模型“调用工具”时，它实际上只是生成了一份请求：
# {
#   "name": "get_weather",
#   "arguments": {
#     "city": "北京"
#   }
# }
# 真正执行函数的是我们的 Python 程序：
# result = get_weather(city="北京")
# 然后程序把结果交还给模型：
# {
#   "city": "北京",
#   "temperature": 26
# }
# 模型看到工具结果后，才生成自然语言回答：
# 北京当前气温为26摄氏度。

# 所以所谓 Tool Calling，不是模型直接控制 Python，而是一套协作协议：
# 模型负责决定“想调用什么”
# Python负责决定“是否允许以及怎样执行”
# 模型负责根据执行结果组织最终回答
# OpenAI 官方将它概括为五步：
# 发送工具定义、接收工具调用、应用程序执行工具、提交工具结果、接收最终回答或新的工具调用。

# 消息角色
# Chat Completions 风格的消息主要包含这些角色：
# system
# 规定模型的身份、任务和行为规则
#
# user
# 用户发送的问题
#
# assistant
# 模型的回复，也可能包含工具调用请求
#
# tool
# Python程序执行工具后返回的结果
#
# 一次完整流程通常是：
# system
# → user
# → assistant（请求调用工具）
# → tool（返回工具结果）
# → assistant（最终自然语言回答）

# 创建初始消息：
messages = [
    {
        "role": "system",
        "content": (
            "你是一个天气助手。"
            "需要实时天气信息时可以调用get_weather工具。"
        ),
    },
    {
        "role": "user",
        "content": "北京未来三天天气怎么样？",
    },
]
# messages 是整个对话历史。每次调用模型时，程序会把相关历史发给模型，让模型知道此前发生了什么。

# 第一次模型响应：请求调用工具
# 假设模型读取用户问题和工具说明后，决定调用天气工具。先手动模拟其响应：
assistant_tool_call_message = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_weather_001", # 负责把“工具执行结果”与“模型发出的工具调用请求”准确关联起来。
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": (
                    '{"city":"北京",'
                    '"days":3,'
                    '"unit":"celsius"}' # 注意这里的 arguments 是一个 JSON 字符串，而不是字典
                ),
            },
        }
    ],
}
# 必须把模型的工具调用加入历史
# 将模型响应加入消息历史：
# messages.append(assistant_tool_call_message)
# 不能只保留工具结果，而丢掉前面的工具调用请求。
# 模型第二次读取历史时，需要看到：
# 我请求了call_weather_001
# → 应用程序返回了call_weather_001的结果
# 如果缺少 assistant 工具调用消息，后面的 tool 结果就失去了上下文。

# python解析并执行工具
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WeatherArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # 表示禁止模型提供未定义字段
        str_strip_whitespace=True,  # 清理字符串首尾空格
    )

    city: str = Field(min_length=1)
    days: int = Field(default=1, ge=1, le=7)
    unit: Literal["celsius", "fahrenheit"] = "celsius"


def get_weather(
    city: str,
    *,
    days: int = 1,
    unit: str = "celsius",
) -> dict[str, object]: # object 表示可以是任意python对象，通常是 dict、list、str、int、float、bool 或 None
    return {
        "city": city,
        "days": days,
        "unit": unit,
        "temperature": 26,
        "condition": "晴",
    }

# 读取本次工具调用：
tool_call = assistant_tool_call_message["tool_calls"][0]

tool_name = tool_call["function"]["name"]
arguments_json = tool_call["function"]["arguments"]

print(tool_name)
print(arguments_json)
print(type(arguments_json))

# 解析并验证：
raw_arguments = json.loads(arguments_json)

validated_arguments = WeatherArguments.model_validate(
    raw_arguments,
    strict=True,
)

# 调用函数：
tool_result = get_weather(
    **validated_arguments.model_dump()
) # 目前直接调用函数，后续学习通用工具调度器

print(tool_result)

# 把工具结果加入消息历史：
# 工具执行结果是Python字典，消息的content应当是字符串：
tool_result_json = json.dumps(
    tool_result,
    ensure_ascii=False,
)
# 建立工具结果消息：
tool_result_message = {
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": tool_result_json,
}
# 加入历史：
messages.append(tool_result_message)

# 第二次模型响应
# 程序把更新后的消息再次发给模型：
# system
# user
# assistant：请求调用get_weather
# tool：返回天气结果

# 模型现在获得了真实工具结果，可以形成最终回答：
# 手动模拟：
final_assistant_message = {
    "role": "assistant",
    "content": (
        "北京未来三天预计为晴天，"
        "平均气温约为26摄氏度。"
    ),
}

messages.append(final_assistant_message)

# 打印最终回答：
print(final_assistant_message["content"])

# 完整消息角色顺序：
for message in messages:
    print(message["role"])
# system
# user
# assistant
# tool
# assistant

# 第一次请求模型时，不仅需要传入 messages，还需要提供工具说明：
# weather_tool = {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "查询指定城市未来几天的天气。",
#         "parameters": WeatherArguments.model_json_schema(),
#     },
# }
# 概念上的第一次 API 请求是：
# completion = client.chat.completions.create(
#     model="模型名称",
#     messages=messages,
#     tools=[weather_tool],
# )
# 此时模型收到的是：
# 用户的问题
# +
# 可以使用的工具名称
# +
# 工具描述
# +
# 参数JSON Schema
# 模型返回工具调用后，应用程序执行工具，再发起第二次请求：
# completion = client.chat.completions.create(
#     model="模型名称",
#     messages=messages,
#     tools=[weather_tool],
# )
