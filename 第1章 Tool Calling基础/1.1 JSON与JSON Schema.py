# JSON对于Agent来说是一个非常重要的工具，因为它可以帮助Agent在与外部系统交互时进行数据验证和结构化。
# JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，易于人类阅读和编写，同时也易于机器解析和生成。

# 假设用户说：“查询北京未来三天的天气，使用摄氏度。”
# 大模型理解了这句话，但天气函数不能直接执行自然语言。函数需要明确的参数。
# get_weather(
#     city="北京",
#     days=3,
#     unit="Celsius",
# )

# 因此，Tool Calling中间需要一种机器容易读写、不同语言都能理解的数据格式：
# JSON
# {
#   "city": "北京",
#   "days": 3,
#   "unit": "Celsius"
# }

# JSON 在 Agent 中的核心作用：
# 用户自然语言
#     ↓
# 大模型生成 JSON 参数
#     ↓
# Python 验证并解析参数
#     ↓
# 调用真正的 Python 函数
#     ↓
# 将函数结果转换成 JSON
#     ↓
# 交还给大模型组织答案

# Python dict
#     │ json.dumps()
#     ↓
# JSON str
#     │ json.loads()
#     ↓
# Python dict



# JSON Schema 是一种用于验证 JSON 数据结构的工具。
# 它定义了 JSON 数据的格式、类型和约束条件，确保数据符合预期的结构和规则。
# 用来解决模型生成的参数不一定符合函数要求的问题。
# 例如天气工具需要：
# {
#   "city": "北京",
#   "days": 3,
#   "unit": "celsius"
# }
# 但是模型可能生成：
# {
#   "city": 100,
#   "days": -7,
#   "unit": "华氏度",
#   "unknown": "多余参数"
# }
# 虽然可能是合法的 JSON，但不符合函数要求。JSON Schema 可以帮助我们验证和过滤这些参数。
# weather_parameters_schema = {
#     "type": "object",
#     "properties": {
#         "city": {
#             "type": "string",
#             "description": "需要查询天气的城市名称，例如北京或上海。",
#         },
#         "days": {
#             "type": "integer",
#             "minimum": 1,
#             "maximum": 7,
#             "description": "查询未来多少天，范围为1到7。",
#         },
#         "unit": {
#             "type": "string",
#             "enum": ["celsius", "fahrenheit"],
#             "description": "温度单位。",
#         },
#     },
#     "required": ["city"],
#     "additionalProperties": False,
# }
# 它表达的规则是：
# - 整体必须是对象，也就是 Python 中类似字典的结构；
# - city 必须是字符串；
# - days 必须是整数，而且只能在 1～7 之间；
# - unit 只能是 "celsius" 或 "fahrenheit"；
# - city 必须出现；
# - 不允许出现没有定义的额外参数。
# 三个关键词：
# "properties": {...}
# 表示允许有哪些字段以及字段的规则。
# "required": ["city"]
# 表示哪些字段必须提供。没有放进 required 的字段就是可选字段。
# "additionalProperties": False
# 表示不允许模型擅自添加其他字段。

# Python 类型注解主要服务于程序员、IDE 和类型检查器；
# JSON Schema 服务于跨系统的数据说明和验证；
# Schema 不会自动创建或执行 Python 函数；
# 模型看见 Schema 后也仍有可能生成错误参数；
# 因此执行函数前仍然必须验证参数。
# Pydantic用来完成这个验证工作。它可以根据 JSON Schema 定义的规则，检查模型生成的参数是否符合要求，并在不符合时抛出错误或提供默认值。

# 在 Tool Calling 中，不能只把参数 Schema 发给模型，还要告诉模型工具叫什么、有什么用途。
# weather_tool = {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "查询指定城市未来几天的天气。",
#         "parameters": weather_parameters_schema,
#     },
# }
# 将它打印出来：
# print(
#     json.dumps(
#         weather_tool,
#         ensure_ascii=False,
#         indent=2,
#     )
# )
# 这个定义可以理解为工具的公开说明书：
# name
# 告诉模型应该请求调用哪个工具
#
# description
# 告诉模型什么情况下应该选择这个工具
#
# parameters
# 告诉模型调用工具时需要生成什么参数
# 非常重要的一点是：这只是工具描述，并不是函数本身。
# weather_tool       # 描述工具的字典
# get_weather        # 真正可以执行的Python函数
# 建立 “工具注册表 ”，把名字和函数对应起来：
# tool_registry = {
#     "get_weather": get_weather,
# }
# 这样当模型返回：
# {
#   "name": "get_weather",
#   "arguments": {
#     "city": "北京",
#     "days": 3
#   }
# }
# Agent 才能根据 "get_weather" 找到真正的 get_weather() 函数。

import json

tool_call_json = """
{
    "name": "get_weather",
    "arguments": {
    "city": "北京",
    "days": 3,
    "unit": "celsius"
  }
}
"""

dict1 = json.loads(tool_call_json)
print(type(dict1))

print(dict1["name"])
print(dict1["arguments"]["city"])
print(dict1["arguments"]["days"])

arguments = dict1["arguments"]
# a = get_weather(**arguments)


