# Pydantic解决模型生成的参数能不能安全使用
# 本节解决模型只返回了字符串，Python如何找到真正的对应函数

# Python中的函数不只是一段可执行的代码，它本身也是对象

def get_weather(city: str) -> dict[str, object]:
    return {
        "city": city,
        "temperature": 26,
    }


print(get_weather) # <function get_weather at 0x...>
print(type(get_weather)) # <class 'function'>
print(callable(get_weather)) # True

# get_weather表示函数对象本身，没有执行函数
# get_weather("北京")表示执行函数，返回函数的结果

# 既然函数是对象，它就可以：
# - 赋值给变量；
# - 放进列表；
# - 放进字典；
# - 作为参数传给另一个函数；
# - 作为返回值返回。
# 例如：
# weather_function = get_weather
#
# result = weather_function("上海")
#
# print(result)
# 这里没有复制函数，只是让变量 weather_function 指向同一个函数对象。


# callable() 用于判断一个对象能否使用圆括号调用：
print(callable(get_weather)) # True
print(callable("get_weather")) # False
print(callable(123)) # False

# 字符串“get_weather”只是函数名称的文本，本身不是函数对象，不能像"get_weather"( )这样调用
# 模型生成的工具名通常恰好就是一个字符串，因此我们需要建立一个映射，把工具名称映射到真正的函数对象

# 工具注册表
# 最简单的工具注册表就是一个字典：
tool_registry = {
    "get_weather": get_weather,
} # 工具名称字符串->函数对象

# 现在可以根据字符串找到函数：
tool_name = "get_weather"

tool_function = tool_registry[tool_name]

print(tool_function)
print(callable(tool_function))

result = tool_function("北京")

print(result)

# 不使用eval(),exec()等函数，直接使用字典映射字符串到函数对象，是最安全的做法。
# 在 Agent 项目中，我们使用工具注册表作为白名单，不使用 eval() 查找工具。
# 模型只能调用注册表中明确存在的工具。

# Callable类型注解
# 注册表的值应该是函数，因此可以使用 Callable 类型注解表示“可以被调用的对象”
from collections.abc import Callable

def add(x: int, y: int) -> int:
    return x + y

# Callable类型注解
operation: Callable[[int, int], int] = add # 注解给出了参数类型和返回类型，接收两个int参数，并返回一个int的可调用对象
# 有时很难给注册表写一个统一的精确函数签名，可以使用：
# Callable[..., object] # ...表示任意参数，object表示任意python对象

# 简单注册表可以注解为：
# tool_registry: dict[str, Callable[..., object]] = {
#     "get_weather": get_weather,
#     "search_papers": search_papers,
# }
# Callable只是类型注解，帮助理解代码，不会自动验证模型参数，参数验证仍然由Pydantic完成。



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

def search_papers(
    query: str,
    *,
    top_k: int = 5,
) -> dict[str, object]:
    return {
        "query": query,
        "top_k": top_k,
        "results": [
            "Synchronization of coupled oscillators",
            "Nonlinear dynamics and chaos",
        ][:top_k],
    }

ToolFunction = Callable[..., object] # 创建了一个类型别名，后续不用反复写 Callable[..., object]

tool_registry: dict[str, ToolFunction] = {
    "get_weather": get_weather,
    "search_papers": search_papers,
}

tool_name = "delete_all_files"
# tool_function = tool_registry[tool_name] # 注册表中没有这个工具，直接访问会报错 KeyError

def get_tool(tool_name: str) -> ToolFunction:
    try:
        return tool_registry[tool_name]
    except KeyError:
        raise ValueError(
            f"未知工具：{tool_name}"
        ) from None

tool_call = {
    "name": "get_weather",
    "arguments": {
        "city": "北京",
        "days": 3,
        "unit": "celsius",
    },
}
tool_name = tool_call["name"]
arguments = tool_call["arguments"]

tool_function = get_tool(tool_name)
result = tool_function(**arguments)

print(result)


# 只有 “名称到函数 ”的映射还不够。一个工具通常还需要：
# - 工具名称；
# - 工具描述；
# - Python 函数；
# - Pydantic 参数模型；
# - JSON Schema。
# 可以使用 dataclass 表示完整的工具定义：

from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field # Field 可以在类型注解之外，为字段补充默认值、验证规则、说明文字等信息。

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    function: ToolFunction # 表示保存一个可调用的对象
    arguments_model: type[BaseModel] # 表示保存的不是Pydantic模型实例，而是一个继承自 BaseModel 的模型类

from typing import Literal

class WeatherArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    city: str = Field(min_length=1)
    days: int = Field(default=1, ge=1, le=7)
    unit: Literal["celsius", "fahrenheit"] = "celsius"

# 定义参数模型
class SearchPapersArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

# 注册完整工具
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

tool = tool_registry["get_weather"]

print(tool.name)
print(tool.description)
print(tool.function)
print(tool.arguments_model)

schema = tool.arguments_model.model_json_schema()
print(schema)

# ToolDefinition
# ├── name：模型返回的工具名
# ├── description：告诉模型什么时候使用
# ├── function：真正执行的Python函数
# └── arguments_model：验证参数并生成Schema

weather_call = {
    "name": "get_weather",
    "arguments": {
        "city": "杭州",
        "days": 2,
        "unit": "celsius",
    },
}

tool = tool_registry[weather_call["name"]]

validated_arguments = tool.arguments_model(**weather_call["arguments"])
result = tool.function(**validated_arguments.model_dump())

print(result)


search_call = {
    "name": "search_papers",
    "arguments": {
        "query": "nonlinear dynamics",
        "top_k": 1,
    },
}

search_tool = get_tool(search_call["name"])

validated_search_arguments = (
    search_tool.arguments_model.model_validate(
        search_call["arguments"],
        strict=True,
    )
)

search_result = search_tool.function(
    **validated_search_arguments.model_dump()
)

print(search_result)

unknown_call = {
    "name": "delete_database",
    "arguments": {},
}

try:
    get_tool(unknown_call["name"])
except ValueError as error:
    print(error)