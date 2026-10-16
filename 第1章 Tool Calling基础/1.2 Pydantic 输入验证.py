# 我们希望把工具调用改造成：
# 模型生成 JSON
# → 解析为 Python 数据
# → Pydantic 验证
# → 验证成功才调用函数
# → 验证失败则返回结构化错误

# Pydantic 为什么被创造出来
# 只使用 Python 类型注解：
# def get_weather(city: str, days: int) -> dict[str, object]:
#     ...
# 并不能阻止下面的调用：
# get_weather(city=100, days="三天")
# city: str 和 days: int 主要是给程序员、IDE和类型检查工具看的。Python运行时默认不会根据类型注解自动检查参数。
# 我们当然可以手动验证：
# if not isinstance(city, str):
#     raise TypeError("city必须是字符串")
#
# if not isinstance(days, int):
#     raise TypeError("days必须是整数")
#
# if not 1 <= days <= 7:
#     raise ValueError("days必须在1到7之间")
# 但字段一多，这些代码就会非常重复。Pydantic 的作用就是：
# 使用Python类型注解描述数据
# +
# 在程序运行时真正验证数据
# +
# 生成清晰的错误信息
# +
# 生成JSON Schema
# 它特别适用于所有 “外部数据进入程序 ”的边界：
# - 大模型生成的工具参数；
# - HTTP API 请求；
# - 配置文件；
# - 环境变量；
# - 数据库查询结果；
# - 用户提交的表单；
# - RAG 文档元数据。
import json

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# 这个类不是普通的业务类，而是一份 “可执行的数据规范 ”。
class WeatherArguments(BaseModel):
    #继承BaseModel后，Pydantic会读取类中的类型注解，并自动获得验证、转换、序列化和生成Schema的能力。
    model_config = ConfigDict(
        extra="forbid",# 表示禁止模型提供未定义字段
        str_strip_whitespace=True,# 清理字符串首尾空格
    )

    city: str = Field(
        min_length=1,
        description="需要查询天气的城市名称。",
    )
    days: int = Field(
        default=1,
        ge=1,
        le=7,
        description="查询未来多少天，范围为1到7。",
    )
    # 表示：
    # 类型：整数
    # 默认值：1
    # ge=1：greater than or equal，必须大于等于1
    # le=7：less than or equal，必须小于等于7
    unit: Literal["celsius", "fahrenheit"] = "celsius"
    # Literal表示这个字段只能从列出的字面值中选择


valid_arguments = {
    "city": "Beijing",
    "days": 3,
    "unit": "celsius",
}

weather_arguments = WeatherArguments.model_validate(valid_arguments)

print(weather_arguments)
print(type(weather_arguments))
print(weather_arguments.city)
print(weather_arguments.days)
print(weather_arguments.unit)

validated_dict = weather_arguments.model_dump()
print(validated_dict)
print(type(validated_dict))

invalid_arguments = {
    "city": "",
    "days": 30,
    "unit": "kelvin",
    "unknown": "多余参数",
}

try:
    WeatherArguments.model_validate(invalid_arguments)
except ValidationError as error:
    print("参数验证失败：")
    print(error)


# 真实 Agent 可以把这些信息整理后交还给模型，让模型重新生成参数，而不是让整个程序崩溃。
try:
    WeatherArguments.model_validate(invalid_arguments)
except ValidationError as error:
    for item in error.errors():
        print(item)


# Pydantic默认可能进行类型转换
# 默认情况下，Pydantic 通常会把可以合理解析的：
# "3"
# 转换成：
# 3
# 这叫作类型转换或类型强制转换。
# 它的便利是：外部输入经常使用字符串表示数字。
converted = WeatherArguments.model_validate(
    {
        "city": "北京",
        "days": "3",
    }
)

print(converted)
print(type(converted.days))
# 但它也可能掩盖上游数据格式错误。
# 对于工具调用，我们有时希望严格要求模型输出真正的 JSON 整数，可以启用严格模式：
class StrictWeatherArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        strict=True,# 启用严格模式
    )

    city: str = Field(
        min_length=1,
        description="需要查询天气的城市名称。",
    )
    days: int = Field(
        default=1,
        ge=1,
        le=7,
        description="查询未来多少天，范围为1到7。",
    )
    unit: Literal["celsius", "fahrenheit"] = "celsius"
# 宽松验证 适合表单、环境变量等经常以字符串传入的数据
# 严格验证 适合协议边界、工具调用和不希望悄悄转换的数据

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
    }

raw_arguments = {
    "city": "上海",
    "days": 3,
    "unit": "celsius",
}

validated_arguments = WeatherArguments.model_validate(raw_arguments)

result = get_weather(
    **validated_arguments.model_dump()
)# 这里不能直接写get_weather(**validated_arguments)
# validated_arguments是Pydanic模型对象，不是普通字典。

print(result)

# 自动生成 JSON Schema

schema = WeatherArguments.model_json_schema()

print(
    json.dumps(
        schema,
        ensure_ascii=False,
        indent=2,
    )
)

# 工具定义可以写成：
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市未来几天的天气。",
        "parameters": WeatherArguments.model_json_schema(),
    },
}

tool_call_json = """
{
  "name": "get_weather",
  "arguments": {
    "city": "  广州  ",
    "days": 3,
    "unit": "celsius"
  }
}
"""

dict1 = json.loads(tool_call_json)
argument = dict1["arguments"]
validated_argument = WeatherArguments.model_validate(argument)
print(validated_argument.city)
result1 = get_weather(**validated_argument.model_dump())
print(result1)



