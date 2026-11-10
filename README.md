# RAG Agent Learning

这是一个面向实践的 RAG、Tool Calling 与 AI Agent 学习仓库。

仓库不依赖大型 Agent 框架，而是从 JSON、参数验证和工具调度开始，逐步实现 Agent 循环、评估体系与可观测性。每个小节都是可以独立运行的 Python 示例，重点解释工程组件为什么需要、怎样工作以及怎样验证。

## 当前进度

| 章节 | 状态 | 内容 |
| --- | --- | --- |
| 第 1 章 Tool Calling 基础 | 已完成 | JSON Schema、Pydantic、工具注册、消息协议、调度器、Agent 循环、权限和异常 |
| 第 2 章 Agent 评估与可观测性基础 | 已完成 | 评估数据集、轨迹评估、检索指标、引用评估、LLM-as-Judge、Trace、回归门槛 |
| 第 3 章 RAG 检索基础 | 计划中 | 文档切分、Embedding、向量索引、Top-K 检索和上下文构建 |
| 第 4 章 RAG 质量优化 | 计划中 | 混合检索、查询改写、Reranker、评估集和错误分析 |
| 第 5 章 RAG Agent | 计划中 | 自主检索、多轮工具调用、证据缺口判断和端到端评估 |

目前仓库已经从“Agent 能运行”推进到“Agent 能被评估和诊断”。下一阶段将正式进入 RAG 检索实现。

## 学习目标

完成当前内容后，可以理解并实现：

- JSON 与 JSON Schema 在 Tool Calling 中的作用
- 使用 Pydantic 验证模型生成的工具参数
- 使用注册表管理工具定义和 Python 函数
- Tool Calling 的 assistant、tool 消息配对流程
- 带终止条件和调用上限的基础 Agent 循环
- 工具权限控制与结构化异常处理
- 使用 JSONL 建立可复用评估数据集
- 评估工具选择、参数、顺序和调用数量
- 计算 Precision@K、Recall@K、Hit Rate@K 和 MRR@K
- 评估引用有效率、引用正确率和声明覆盖率
- 使用 LLM-as-Judge 评估正确性、证据性、相关性和完整性
- 使用 Trace 记录延迟、Token、步骤和错误
- 对比基线版本与候选版本，并设置质量回归门槛

## 章节目录

### 第 1 章 Tool Calling 基础

这一章从数据协议开始，逐步搭建一个不依赖框架的轻量 Agent 运行循环。

| 小节 | 核心内容 |
| --- | --- |
| [1.1 JSON 与 JSON Schema](第1章%20Tool%20Calling基础/1.1%20JSON与JSON%20Schema.py) | JSON 序列化、工具参数结构和 Schema 约束 |
| [1.2 Pydantic 输入验证](第1章%20Tool%20Calling基础/1.2%20Pydantic%20输入验证.py) | 字段约束、默认值、额外字段和验证错误 |
| [1.3 Callable 与工具注册表](第1章%20Tool%20Calling基础/1.3%20Callable与工具注册表.py) | 工具定义、函数引用、注册和查找 |
| [1.4 Tool Calling 消息流程](第1章%20Tool%20Calling基础/1.4%20Tool%20Calling%20消息流程.py) | system、user、assistant、tool 消息与 call_id |
| [1.5 工具调度器](第1章%20Tool%20Calling基础/1.5%20工具调度器.py) | 参数解析、验证、函数执行和结果封装 |
| [1.6 Agent 循环与终止条件](第1章%20Tool%20Calling基础/1.6%20Agent%20循环与终止条件.py) | 多轮调用、并行工具请求、步数限制和退出条件 |
| [1.7 工具权限与异常处理](第1章%20Tool%20Calling基础/1.7%20工具权限与异常处理.py) | 权限检查、未知工具、执行异常和安全边界 |

第 1 章形成的基本运行链路如下：

~~~text
用户问题
  ↓
模型生成工具调用请求
  ↓
JSON 参数解析
  ↓
Pydantic 参数验证
  ↓
工具注册表查找
  ↓
权限检查与工具执行
  ↓
结构化 tool 消息
  ↓
模型继续调用工具或生成最终回答
~~~

### 第 2 章 Agent 评估与可观测性基础

这一章关注 Agent 是否完成任务、为什么失败，以及新版本是否发生质量回归。示例主要使用可重复的模拟数据，便于先验证评分逻辑，再接入真实模型和真实检索系统。

| 小节 | 核心内容 |
| --- | --- |
| [2.1 Agent 评估基础](第2章%20Agent评估与可观测性基础/2.1%20Agent评估基础.py) | 参考结果、实际结果、失败原因和通过率 |
| [2.2 JSONL 评估数据集](第2章%20Agent评估与可观测性基础/2.2%20JSONL评估数据集.py) | 数据集读写、逐行校验和错误定位 |
| [2.3 Pydantic 评估案例模型](第2章%20Agent评估与可观测性基础/2.3%20Pydantic评估案例模型.py) | 严格案例模型、条件约束和重复编号检查 |
| [2.4 工具调用轨迹评估](第2章%20Agent评估与可观测性基础/2.4%20工具调用轨迹评估.py) | 工具名称、参数、顺序、遗漏和多余调用 |
| [2.5 检索评估指标](第2章%20Agent评估与可观测性基础/2.5%20检索评估指标.py) | Precision@K、Recall@K、Hit Rate@K、MRR@K |
| [2.6 引用与证据评估](第2章%20Agent评估与可观测性基础/2.6%20引用与证据评估.py) | 引用有效率、引用正确率和声明覆盖率 |
| [2.7 LLM-as-Judge](第2章%20Agent评估与可观测性基础/2.7%20LLM-as-Judge.py) | 结构化裁判结果、失败维度和人工一致性 |
| [2.8 运行记录与可观测性](第2章%20Agent评估与可观测性基础/2.8%20运行记录与可观测性.py) | Run、Span、延迟、Token、错误和 JSONL Trace |
| [2.9 回归门槛与对比报告](第2章%20Agent评估与可观测性基础/2.9%20回归门槛与对比报告.py) | 基线对比、绝对门槛、允许退步幅度和发布判断 |

第 2 章建立的质量闭环如下：

~~~text
评估数据集
  ↓
运行 Agent
  ↓
记录最终答案、检索结果和工具轨迹
  ↓
计算任务、检索、引用和 Judge 指标
  ↓
记录运行耗时、Token 和异常
  ↓
与基线版本比较
  ↓
通过或阻止候选版本
~~~

## 环境要求

- Python 3.10 或更高版本
- Pydantic 2.x

创建虚拟环境并安装依赖：

~~~powershell
git clone https://github.com/jianjian27/RAG-Agent-learning.git
Set-Location RAG-Agent-learning

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

Linux 或 macOS 激活方式：

~~~bash
source .venv/bin/activate
~~~

## 运行示例

每个文件都可以从仓库根目录单独运行。

运行 Tool Calling 消息流程：

~~~powershell
python ".\第1章 Tool Calling基础\1.4 Tool Calling 消息流程.py"
~~~

运行基础 Agent 循环：

~~~powershell
python ".\第1章 Tool Calling基础\1.6 Agent 循环与终止条件.py"
~~~

运行检索评估指标：

~~~powershell
python ".\第2章 Agent评估与可观测性基础\2.5 检索评估指标.py"
~~~

运行引用与证据评估：

~~~powershell
python ".\第2章 Agent评估与可观测性基础\2.6 引用与证据评估.py"
~~~

运行回归门槛对比：

~~~powershell
python ".\第2章 Agent评估与可观测性基础\2.9 回归门槛与对比报告.py"
~~~

运行某一章全部示例：

~~~powershell
Get-ChildItem ".\第2章 Agent评估与可观测性基础" -Filter "*.py" |
    Sort-Object Name |
    ForEach-Object { python $_.FullName }
~~~

## 模型配置

仓库保留了兼容 OpenAI 风格接口的环境变量模板：

~~~powershell
Copy-Item .env.example .env
~~~

~~~dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
~~~

当前示例以本地、确定性数据为主，其中 LLM-as-Judge 小节使用模拟模型输出验证评分流程，因此运行现有章节不需要 API Key。后续接入真实模型时再填写这些变量。

请勿将真实密钥提交到 Git 仓库。

## 项目结构

~~~text
.
├── README.md
├── requirements.txt
├── .env.example
├── 第1章 Tool Calling基础
│   ├── 1.1 JSON与JSON Schema.py
│   ├── 1.2 Pydantic 输入验证.py
│   ├── 1.3 Callable与工具注册表.py
│   ├── 1.4 Tool Calling 消息流程.py
│   ├── 1.5 工具调度器.py
│   ├── 1.6 Agent 循环与终止条件.py
│   └── 1.7 工具权限与异常处理.py
└── 第2章 Agent评估与可观测性基础
    ├── 2.1 Agent评估基础.py
    ├── 2.2 JSONL评估数据集.py
    ├── 2.3 Pydantic评估案例模型.py
    ├── 2.4 工具调用轨迹评估.py
    ├── 2.5 检索评估指标.py
    ├── 2.6 引用与证据评估.py
    ├── 2.7 LLM-as-Judge.py
    ├── 2.8 运行记录与可观测性.py
    └── 2.9 回归门槛与对比报告.py
~~~

## 推荐学习顺序

建议按照文件编号依次运行，并在每一节完成三个动作：

1. 先阅读文件顶部的概念说明和学习目标。
2. 运行示例，观察正常案例与故意构造的失败案例。
3. 修改参数、轨迹或参考答案，确认评分和异常分支符合预期。

第 2 章的重点是建立评估意识。Agent 能返回自然语言答案，只能说明流程已经执行。工具是否选对、证据是否找全、引用是否支持结论以及新版本是否退步，都需要独立指标验证。

## 与实践项目的关系

本仓库用于拆解和练习 Agent 与 RAG 的基础组件。完整的论文知识库、语义检索、RAG 问答和科研 Agent 实践位于：

[nonlinear-dynamics-paper-assistant](https://github.com/jianjian27/nonlinear-dynamics-paper-assistant)

学习仓库中的工具协议、运行循环、评估指标和可观测性方法，会逐步应用到该实践项目中。

## 后续计划

下一阶段计划进入 RAG 检索基础：

- 文档、章节与 Chunk 的数据模型
- 文本切分与元数据保留
- Embedding 编码接口
- 向量索引与相似度检索
- Top-K 结果和来源过滤
- 带引用的上下文构建
- 基于标准相关 Chunk 的检索评估
- 将检索能力封装为 Agent 工具

之后继续学习混合检索、查询改写、Reranker、多轮自主检索和端到端 RAG Agent 评估。
