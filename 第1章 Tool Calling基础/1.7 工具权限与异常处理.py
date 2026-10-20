# 工具只有获得授权才能执行
# 执行失败时，程序要能区分“可预期失败”和“代码缺陷”

# 把工具加入注册表，只说明程序认识它，是否允许某次调用，需要结合当前权限判断

# 即使提示词写着“不要删除文件”，模型仍有可能生成删除请求。
# 真正决定操作能否发生的是执行工具的程序，因此检查必须位于函数调用之前：
# 模型提出调用请求
# → 注册表查找
# → 检查权限
# → 验证参数
# → 执行工具

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel


ToolFunction = Callable[..., object]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    function: ToolFunction
    arguments_model: type[BaseModel]

    read_only: bool = True # 默认工具只读取资源
    requires_confirmation: bool = False # 默认工具不需要用户确认即可执行

# 当前这次调用获得了哪些权限
@dataclass(frozen=True)
class ExecutionContext:
    allowed_tools: frozenset[str]
    allow_write: bool = False
    user_confirmed: bool = False

# 例如:
context = ExecutionContext(
    allowed_tools=frozenset(
        {"get_weather", "save_note"}
    ), # 本次允许考虑调用天气和保存笔记工具
    allow_write=False, # 当前禁止写入操作
    user_confirmed=False # 用户尚未确认本次操作
)

# 建立专门的异常类型
class ToolPermissionError(Exception):
    """当前调用没有执行权限。"""

class ToolUnavailableError(Exception):
    """外部服务暂时不可用。"""

# 程序需要根据错误含义选择处理方式：
# ToolPermissionError
# → 说明调用被权限规则拒绝
#
# ToolUnavailableError
# → 说明外部服务发生可预期故障
#
# NameError、意外的KeyError等
# → 可能是代码写错，需要排查

# 权限检查函数
def check_permission(
    tool: ToolDefinition,
    context: ExecutionContext,
) -> None:
    if tool.name not in context.allowed_tools:
        raise ToolPermissionError(
            "当前任务不允许使用该工具。"
        )

    if not tool.read_only and not context.allow_write: # 如果这个工具会修改数据，并且当前没有写入权限，就拒绝执行
        raise ToolPermissionError(
            "当前任务不允许修改数据。"
        )

    if tool.requires_confirmation and not context.user_confirmed: # 如果这个工具需要用户确认，并且用户尚未确认，就拒绝执行
        raise ToolPermissionError(
            "该操作需要用户确认。"
        )

# 处理工具执行失败
# 假设服务超时。底层异常可能包含内部地址、请求细节或其他不应该直接展示的信息。
# 工具层可以把底层异常转换成业务异常：
def get_weather(city: str) -> dict[str, object]:
    try:
        # 模拟外部服务超时
        raise TimeoutError("请求天气服务超时。")
    except TimeoutError as error:
        raise ToolUnavailableError(
            "天气服务暂时不可用，请稍后再试。"
        ) from error # 对外提供清楚的业务错误,保留底层异常链，便于日志排查

def save_note(text: str) -> dict[str, object]:
    print("已进入保存笔记函数")

    return {
        "saved": True,
        "text": text,
        "mode": "模拟保存，未写入文件",
    }

# 参数模型
from pydantic import ConfigDict, Field

class WeatherArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    city: str = Field(min_length=1)

class NoteArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: str = Field(min_length=1, max_length=1000)

# 建立注册表
tool_registry = {
    "get_weather": ToolDefinition(
        name="get_weather",
        description="查询指定城市的天气。",
        function=get_weather,
        arguments_model=WeatherArguments,
    ),
    "save_note": ToolDefinition(
        name="save_note",
        description="保存笔记。",
        function=save_note,
        arguments_model=NoteArguments,
        read_only=False, # 这个工具会修改数据
        requires_confirmation=True, # 这个工具需要用户确认才能执行
    ),
}

# 把权限检查和错误处理放到同一个执行函数里
import logging
from pydantic import ValidationError
logger = logging.getLogger(__name__)
def execute_tool(
        tool_name: str,
        raw_arguments: object,
        context: ExecutionContext,
) -> dict[str, object]:
    tool = tool_registry.get(tool_name)

    if tool is None:
        return {
            "ok": False,
            "error": {
                "type": "unknown_tool",
                "message": "请求的工具不存在。",
            },
        }

    try:
        check_permission(tool, context)
    except ToolPermissionError as error:
        return {
            "ok": False,
            "error": {
                "type": "permission_denied",
                "message": str(error),
            },
        }

    try:
        arguments = tool.arguments_model.model_validate(raw_arguments, strict =True,)
    except ValidationError as error:
        return {
            "ok": False,
            "error": {
                "type": "validation_error",
                "details": [
                    {
                        "field": list(item["loc"]),
                        "type": item["type"],
                    }
                    for item in error.errors()
                ],
            },
        }

    try:
        result = tool.function(
            **arguments.model_dump()
        )
    except ToolUnavailableError:
        logger.exception(
            "工具外部服务调用失败：%s",
            tool.name,
        )

        return {
            "ok": False,
            "error": {
                "type": "tool_unavailable",
                "message": "工具服务暂时不可用，请稍后重试。",
            },
        }

    return {
        "ok": True,
        "result": result,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # 1. 天气工具获得授权，但外部服务超时。
    weather_context = ExecutionContext(
        allowed_tools=frozenset({"get_weather"}),
    )

    print(
        execute_tool(
            "get_weather",
            {"city": "北京"},
            weather_context,
        )
    )

    # 2. 笔记工具在白名单中，但没有写入权限。
    read_only_context = ExecutionContext(
        allowed_tools=frozenset({"save_note"}),
    )

    print(
        execute_tool(
            "save_note",
            {"text": "记录论文中的同步条件。"},
            read_only_context,
        )
    )

    # 3. 允许写入，但用户尚未确认。
    unconfirmed_context = ExecutionContext(
        allowed_tools=frozenset({"save_note"}),
        allow_write=True,
    )

    print(
        execute_tool(
            "save_note",
            {"text": "记录论文中的同步条件。"},
            unconfirmed_context,
        )
    )

    # 4. 本次调用获得写入权限和用户确认。
    confirmed_context = ExecutionContext(
        allowed_tools=frozenset({"save_note"}),
        allow_write=True,
        user_confirmed=True,
    )

    print(
        execute_tool(
            "save_note",
            {"text": "记录论文中的同步条件。"},
            confirmed_context,
        )
    )

    # 5. 即使允许写入并已确认，仍不能调用白名单之外的工具。
    excluded_context = ExecutionContext(
        allowed_tools=frozenset({"get_weather"}),
        allow_write=True,
        user_confirmed=True,
    )

    print(
        execute_tool(
            "save_note",
            {"text": "这条笔记不应该被保存。"},
            excluded_context,
        )
    )