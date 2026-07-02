---
title: "LangChain和LangGraph"
date: 2026-07-01T16:00:00+08:00
draft: false
description: "介绍LangChain和LangGraph"

# 分类（建议：大方向，1个即可）
categories:
  - "AI Agent"

# 标签（建议：具体技术点/工具，可以写多个）
tags:
  - "LangChain"
  - "LangGraph"
---
## **LangChain和LangGraph**

|特性|LangChain|LangGraph|
|---|---|---|
|**诞生背景**|解决LLM应用开发的零散、重复问题|解决LangChain复杂智能体自定义难、扩展难、生产部署难的问题[](https://www.langchain.com/blog/building-langgraph)|
|**核心理念**|模块化、可组合的链|“图即工作流”，低抽象、强控制的状态机[](https://www.langchain.com/blog/building-langgraph)|
|**核心组件**|Components, Chains, LCEL|Nodes, Edges, StateGraph, Checkpointer|
|**执行模式**|线性串联为主，LLM自主决策为辅|有向图驱动，支持循环、分支与并行处理|
|**生态系统**|丰富的社区集成，庞大的工具库|原生LangChain集成，可独立运行|
|**设计目标**|快速开发、快速原型|生产可靠、可观测、易扩展[](https://www.langchain.com/blog/building-langgraph)|
Deep Agents是一个基于LangGraph的代理工具：规划、子代理、文件系统工具和上下文管理。  
LangChain是代理框架：模型、工具和代理循环的抽象和集成。  
LangGraph是编排运行时：持久执行、流、人在循环和持久性。  
LangSmith是用于跨框架跟踪、评估、提示和部署的平台。

**Function Calling（函数调用/工具调用）** 是大语言模型（LLM）的一项核心能力——它允许模型在生成回复时，不直接输出文本，而是输出一个 **结构化的函数调用请求**（通常是 JSON），由开发者实际执行该函数，然后将执行结果返回给模型，模型再据此生成最终回复。

### 注：本文根据吴恩达AI Agent课程学习整理，模型限制未使用chatgpt，用Deekseek替换。

基于react模式的智能代理
代码  
```py
import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI

# ================= 改动开始 =================
# 使用 DeepSeek 的 API 地址和你的 API Key
# 建议在 .env 文件中设置 DEEPSEEK_API_KEY
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 如果没有设置，会读取 OPENAI_API_KEY 兼容
if not DEEPSEEK_API_KEY:
    raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"      # DeepSeek 官方 API 地址
)
# ================= 改动结束 =================

# 测试调用（可选）
chat_completion = client.chat.completions.create(
    model="deepseek-chat",          # 使用 DeepSeek 模型
    messages=[{"role": "user", "content": "Hello world"}]
)
print(chat_completion.choices[0].message.content)


class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        completion = client.chat.completions.create(
            model="deepseek-chat",          # 统一使用 DeepSeek 模型
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content


prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()


def calculate(what):
    return eval(what)


def average_dog_weight(name):
    if name in "Scottish Terrier":
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")


known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

abot = Agent(prompt)
result = abot("How much does a toy poodle weigh?")
print(result)
result = average_dog_weight("Toy Poodle")
result
next_prompt = "Observation: {}".format(result)
abot(next_prompt)
abot.messages

abot = Agent(prompt)
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
abot(question)
next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
print(next_prompt)
abot(next_prompt)
next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
print(next_prompt)
abot(next_prompt)
next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)
next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)
abot(next_prompt)


action_re = re.compile('^Action: (\w+): (.*)$')   # python regular expression to selection action


def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a)
            for a in result.split('\n')
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)
```
这段代码实现了一个**基于大语言模型的 ReAct 风格智能代理**，能够通过“思考-行动-观察”循环来解决需要多步推理或工具调用的问题。它使用了 DeepSeek 模型（兼容 OpenAI API），并内置了两个简单工具：数学计算和查询狗的平均体重。

下面分段详细解释代码的功能和工作原理。

---

### 1. 环境准备与 API 客户端初始化

python
```py
from dotenv import load_dotenv
_ = load_dotenv()
from openai import OpenAI
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)
```
- 从 `.env` 文件加载环境变量，获取 DeepSeek 的 API 密钥。
    
- 创建 OpenAI 客户端，并指定 `base_url` 为 DeepSeek 官方 API 地址。
    
- 后续所有对大模型的请求都通过该客户端发送。
    

---

### 2. Agent 类：对话管理与模型调用

python
```py
class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})
    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    def execute(self):
        completion = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content
```
- **`Agent`** 维护一个对话历史 `self.messages`。
    
- **初始化时**可以设置系统提示（system prompt）。
    
- **调用实例**（`__call__`）时：
    
    1. 将用户消息加入 `messages`。
        
    2. 调用 `execute()` 发送当前完整消息列表给模型。
        
    3. 将模型的回复加入 `messages`。
        
    4. 返回回复内容。
        
- `temperature=0` 使输出尽可能确定，利于解析 Action 格式。
    

---

### 3. 系统 Prompt：定义 ReAct 行为模式

python
```py
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation...
...（详细内容）
""".strip()
```
这个 prompt 告诉模型：

- 使用 **Thought** 描述推理过程。
    
- 使用 **Action** 选择一个可用工具并给出参数，然后输出 **PAUSE**。
    
- 之后用户会提供一个 **Observation**（工具执行结果）。
    
- 模型根据 Observation 继续思考，最终输出 **Answer**。
    

**可用工具**：

- `calculate`：计算数学表达式（用 Python 的 `eval`）。
    
- `average_dog_weight`：根据狗品种返回平均体重。
    

**示例会话**展示了输入输出格式，保证模型按规则生成。

---

### 4. 工具函数与动作映射

python
```py
def calculate(what):
    return eval(what)
def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")
known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}
```
- **`calculate`**：直接 `eval` 字符串（有安全风险，示例可接受）。
    
- **`average_dog_weight`**：基于品种名返回写死的体重字符串。
    
- **`known_actions`**：字典，模型输出 `Action: 函数名: 参数` 时，可映射到实际函数。
    

---

### 5. 手动演示交互（非自动循环）

在代码中间部分，用户手动演示了如何一步步调用：

python
```py
abot = Agent(prompt)
result = abot("How much does a toy poodle weigh?")
print(result)                     # 模型输出 Action 和 PAUSE
result = average_dog_weight("Toy Poodle")   # 手工执行工具
next_prompt = "Observation: {}".format(result)
abot(next_prompt)                 # 将 Observation 喂给 Agent
```
类似地，后面的代码又手动处理了两只狗的体重计算问题：

- 先分别调用 `average_dog_weight` 获取两只狗的体重。
    
- 再手工计算 `37 + 20` 得到总和。
    
- 每次将 Observation 作为新消息传给 Agent，最终让模型输出 Answer。
    

**这种手动方式繁琐且容易被中断，因此代码最后提供了一个自动循环函数 `query`**。

---

### 6. 自动循环函数 `query`

python
```py
action_re = re.compile('^Action: (\w+): (.*)$')
def query(question, max_turns=5):
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return
```
#### 工作流程

1. **初始化**：创建新的 `Agent` 实例（传入系统 prompt），`next_prompt` 初始为用户问题。
    
2. **循环**（最多 `max_turns` 次）：
    
    - 调用 `bot(next_prompt)` 得到模型回复 `result`。
        
    - 打印回复内容。
        
    - 用正则表达式 `^Action: (\w+): (.*)$` 逐行匹配，找到所有形如 `Action: 动作名: 参数` 的行。
        
    - **如果有 Action**：
        
        - 取第一个匹配的 Action。
            
        - 验证动作是否在 `known_actions` 中。
            
        - 执行对应的工具函数，得到 `observation`。
            
        - 打印执行信息。
            
        - 构造新的 `next_prompt = "Observation: " + observation`。
            
        - 继续循环（模型会看到 Observation 并继续推理）。
            
    - **如果没有 Action**（例如模型直接输出 Answer）：循环结束，函数返回。
        
3. **结束条件**：模型不再输出 Action（即输出了 Answer），或达到最大轮数。
    

#### 示例调用

python
```py
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)
```
运行时会自动：

1. 模型输出：`Action: average_dog_weight: Border Collie` → 工具返回 `"a Border Collies average weight is 37 lbs"` → 作为 Observation 喂回。
    
2. 模型继续输出：`Action: average_dog_weight: Scottish Terrier` → 工具返回 `"Scottish Terriers average 20 lbs"` → 再喂回。
    
3. 模型输出：`Action: calculate: 37 + 20` → 工具返回 `57` → 喂回。
    
4. 模型最终输出：`Answer: The combined weight is 57 lbs`（没有 Action），循环结束。
    

---

### 7. 总结：整体设计思想

- **ReAct 模式**：将推理（Thought）与行动（Action）交错，借助外部工具增强 LLM 的能力。
    
- **模型无关**：通过 OpenAI 兼容接口，可替换任意支持该 API 的模型（此处用 DeepSeek）。
    
- **工具可扩展**：只需在 `known_actions` 中添加新函数，并更新系统 prompt 中的描述即可。
    
- **自动循环**：`query` 函数实现了完整的“解析 Action → 执行工具 → 反馈 Observation”循环，直到模型给出最终答案。
    

这段代码是一个简洁的教学示例，展示了如何用很少的代码构建一个能计算、查表并进行多步推理的智能代理。在实际应用中，可以替换工具函数为更复杂的 API 调用（如搜索引擎、数据库查询等），从而构建出功能更强大的 Agent。

————————————————————————————————————————

## **基于 LangGraph 的 ReAct Agent**
```py
from dotenv import load_dotenv
_ = load_dotenv()
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import os
 

# ================= DeepSeek 配置开始 =================
# 从环境变量获取 DeepSeek API Key（建议在 .env 中设置 DEEPSEEK_API_KEY）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")  
# 创建 ChatOpenAI 实例，指向 DeepSeek 的 API 地址
# 注意：模型名称使用 deepseek-chat 或 deepseek-coder
deepseek_model = ChatOpenAI(
    model="deepseek-chat",                # DeepSeek 对话模型
    openai_api_key=DEEPSEEK_API_KEY,      # 注意参数名是 openai_api_key（兼容 OpenAI）
    openai_api_base="https://api.deepseek.com/v1",  # DeepSeek 官方地址
    temperature=0                         # 可选：降低随机性
)

# ================= DeepSeek 配置结束 =================

# 搜索工具（与模型无关）
tool = TavilySearch(max_results=4)
print(type(tool))
print(tool.name)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}

        # 模型绑定工具（同样兼容 DeepSeek，因为 DeepSeek 支持 function calling）

        self.model = model.bind_tools(tools)
    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

# 系统提示（可根据 DeepSeek 调整，但原 prompt 也适用）
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
  
# ================= 使用 DeepSeek 模型创建 Agent =================

# 对于简单任务（如天气查询），使用 deepseek-chat 即可
abot = Agent(deepseek_model, [tool], system=prompt)

# 可选：可视化图结构（需要 IPython，如果没有可注释）
# from IPython.display import Image
# Image(abot.graph.get_graph().draw_png())

# 测试1：单个城市天气
messages = [HumanMessage(content="What is the weather in sf?")]
result = abot.graph.invoke({"messages": messages})
print("Final answer:", result['messages'][-1].content)

# 测试2：两个城市天气
messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})
print("Final answer:", result['messages'][-1].content)

# 测试3：多步推理问题（需要更好性能，仍用 deepseek-chat，也可以用 deepseek-coder）
query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question."
messages = [HumanMessage(content=query)]
# 这里如果希望使用更强的模型，DeepSeek 目前主要提供 deepseek-chat，已经足够
abot = Agent(deepseek_model, [tool], system=prompt)
result = abot.graph.invoke({"messages": messages})
print("Multi-step answer:", result['messages'][-1].content)
```
### 1. 代码整体架构

你实现了一个**基于 LangGraph 的 ReAct Agent**，它能够循环地进行“思考→调用工具→观察结果→继续思考”的过程。核心组件包括：

- **StateGraph**：定义 Agent 的状态和流程。
    
- **AgentState**：存储消息列表（使用 `operator.add` 作为 reducer，自动累加新消息）。
    
- **Agent 类**：封装了图构建、模型调用、工具执行逻辑。
    
- **外部工具**：`TavilySearch`（搜索互联网）。
    
- **大模型**：DeepSeek Chat（通过 `ChatOpenAI` 接口调用，支持 function calling）。
    

---

### 2. 流程图与节点

text

llm (调用模型) → exists_action (判断是否有工具调用) → 如果有 → action (执行工具) → 回到 llm
                                                   → 如果没有 → END (结束)

- **`call_openai`**：调用模型，传入系统提示和对话历史。模型可以选择返回普通文本或请求工具调用。
    
- **`exists_action`**：检查模型返回的消息中是否包含 `tool_calls`。
    
- **`take_action`**：遍历所有工具调用，执行对应的工具函数（`TavilySearch`），将结果包装成 `ToolMessage` 返回。
    

由于 `graph.add_edge("action", "llm")`，工具执行完后会自动再次调用模型，让模型根据工具结果继续推理。

---

### 3. DeepSeek 配置兼容性

python
```py
deepseek_model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0
)
```
- DeepSeek 完全兼容 OpenAI 的 function calling API，因此 `model.bind_tools(tools)` 可以正常工作。
    
- 使用 `openai_api_base` 而不是 `base_url`，这是 `langchain_openai` 的旧参数名，但依然有效。
    

---

### 4. 运行结果逐项分析

#### 测试1：单个城市天气

**输入**：`"What is the weather in sf?"`

**执行过程**：

1. 模型调用一次，返回 `tool_calls`，请求 `tavily_search`，参数 `query: 'current weather in San Francisco'`。
    
2. Agent 执行搜索工具，得到天气结果（温度55°F，多云等）。
    
3. 模型再次被调用，结合搜索结果生成最终回答。
    

**输出**：

text

Here's the current weather in San Francisco (as of May 7, 2026, ~11:38 PM local time)...
模型没有继续请求工具，因此流程结束。

### 测试2：两个城市天气

**输入**：`"What is the weather in SF and LA?"`

**执行过程**：

- 模型一次返回了两个 `tool_calls`（并行请求）：
    1. 搜索 San Francisco 天气    
    2. 搜索 Los Angeles 天气
- `take_action` 遍历这两个调用，依次执行（串行）。 
- 得到两个观察结果后，模型生成对比回答。
**输出**：

text
Here's the current weather for both cities (as of late evening on May 7, 2026)...
模型正确合并了两个来源的信息。

### 测试3：多步推理问题

**输入**：

text
Who won the super bowl in 2024? In what state is the winning team headquarters located? 
What is the GDP of that state? Answer each question.

**执行过程**（从日志可见多轮交互）：
1. 第一轮：模型请求搜索 `Who won the Super Bowl in 2024?` → 得到结果 “Kansas City Chiefs”。  
2. 第二轮：模型根据上一步结果，请求搜索 `Kansas City Chiefs headquarters location state` → 得到 “Missouri”。    
3. 第三轮：同时请求搜索 `GDP of Missouri 2024` 和再次确认总部位置（避免错误）→ 得到 GDP 数据。
4. 第四轮：模型整理答案，不再请求工具，输出最终答案。

**输出**：
text
1. Who won the Super Bowl in 2024? The Kansas City Chiefs won Super Bowl LVIII...
2. In what state...? Missouri.
3. What is the GDP...? ~$356.65 billion (real GDP) or ~$448.7 billion (nominal).

**值得注意**：模型能够根据中间结果决定下一步需要查询什么，体现了 ReAct 的多步推理能力。

---

## 5. 警告信息解读

### 警告1

text

LangChainPendingDeprecationWarning: The default value of `allowed_objects` will change in a future version...
- 来源：`langgraph.cache.base` 内部使用 `JsonPlusSerializer` 时产生的警告。    
- 影响：目前不影响运行，未来 LangGraph 更新可能需要显式设置 `allowed_objects` 参数。暂时可忽略。

### 警告2 （未出现在本次运行，但你之前遇到过）
你之前遇到的 `TavilySearchResults` 弃用警告已经通过使用 `langchain_tavily.TavilySearch` 解决了。现在没有该警告。

---

## 6. 代码的潜在改进点

尽管代码能正确运行，仍有几个可以优化的地方：

### 6.1 工具调用的并行执行

目前 `take_action` 是串行执行所有工具调用。如果模型一次请求多个工具（如测试2），可以改为并发执行以提高效率：

python
```py
from concurrent.futures import ThreadPoolExecutor
def take_action(self, state):
    tool_calls = state['messages'][-1].tool_calls
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(self._invoke_tool, tool_calls))
    return {'messages': results}
```
### 6.2 错误处理增强

当工具名称不存在时，只返回字符串 `"bad tool name, retry"`。可以更明确地引导模型：

python
```py
result = f"Error: Tool '{t['name']}' not found. Available tools: {list(self.tools.keys())}"
```
### 6.3 最大循环次数保护

目前没有设置最大迭代次数，如果模型陷入循环（反复请求相同工具），理论上会无限运行。可以在 `Agent` 中添加计数器：

python
```py
def __init__(self, ..., max_iterations=10):
    self.max_iterations = max_iterations
```
并在 `call_openai` 或图外部控制。

### 6.4 环境变量检查

你已经检查了 `DEEPSEEK_API_KEY`，但 `TAVILY_API_KEY` 没有显式检查。可以添加：

python
```py
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("请设置 TAVILY_API_KEY 环境变量")
```
### 6.5 模型选择

- `deepseek-chat` 对于大多数任务足够好。如果任务涉及大量代码或逻辑推理，可以尝试 `deepseek-coder`。
- 对于第三步中的 GDP 查询，模型没有进行数值计算而是直接输出了搜索到的结果，符合预期。

---

## 7. 总结

- **成功运行**：你的 Agent 能够正确理解问题、调用 Tavily 搜索工具、整合信息并返回答案。
    
- **DeepSeek 兼容性**：完全支持 OpenAI function calling，与 LangChain 无缝集成。
    
- **LangGraph 工作流**：`llm` → `exists_action` → `action` → `llm` 循环稳定，能够处理单步和多步推理。
    
- **警告处理**：已消除 Tavily 弃用警告，剩余的 LangGraph 内部警告不影响使用。

## AI Search
### Agentic Search

```py
# 导入 dotenv 中的 load_dotenv 函数，用于加载 .env 文件中的环境变量
# pip install python-dotenv tavily-python 提前导入相关python依赖
# 搜索工具初始连接
from dotenv import load_dotenv


# 导入 os 模块，用于读取操作系统的环境变量

import os
  
# 从 tavily 库中导入 TavilyClient 类，用于调用 Tavily 搜索 API
from tavily import TavilyClient
  
# 调用 load_dotenv()，它会自动查找项目根目录下的 .env 文件，并将其中的键值对加载到环境变量中
load_dotenv()
# 从环境变量中获取 TAVILY_API_KEY 的值（即你注册 Tavily 后获得的真实 API Key）
# 这里使用 os.getenv()，它等同于 os.environ.get()，如果环境变量不存在则返回 None

api_key = os.getenv("TAVILY_API_KEY")

# 检查 API Key 是否成功读取，如果为空则主动抛出异常，让程序停止并提示错误信息

if not api_key:

    raise ValueError("未找到 TAVILY_API_KEY，请检查 .env 文件")  

# 使用读取到的 API Key 创建 TavilyClient 对象，该对象负责发起搜索请求
client = TavilyClient(api_key=api_key)
# 调用 client.search() 方法向 Tavily API 发起搜索请求
# 参数 "What is in Nvidia's new Blackwell GPU?" 是查询的问题
# include_answer=True 表示希望 Tavily 在返回结果中直接提供一个总结性的答案（而非只返回原始网页片段）
result = client.search("What is in Nvidia's new Blackwell GPU?",
                       include_answer=True) 
# 从返回的 result 字典中提取 "answer" 字段，并使用 print() 函数将其输出到控制台
# 注意：在 VSCode 中运行 Python 脚本时，必须显式使用 print() 才能看到输出

print(result["answer"])
```
### Regular Search
```py
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import re

city = "Bei Jing"
query = f"""
    what is your current weather in {city}?
    Should I travel there today?
"""

# ② 代理设置（如果需要）—— 使用字符串而不是字典
proxy = "http://127.0.0.1:7897"   # 改成你实际代理地址
ddg = DDGS(proxy=proxy)           # ✅ 传入字符串
 
# 如果没有代理，就用 ddg = DDGS()

# def search(query, max_results=3):
#     # 使用 backend="html" 或 "lite"
#     results = ddg.text(query, max_results=max_results, backend="html")
#     return [item["href"] for item in results]  
# urls = search(query)
# for url in urls:
#     print(url) 

def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(f"returning previous results due to exception reaching ddg.")
        results = [ # cover case where DDG rate limits due to high deeplearning.ai volume
        "https://weather.com/weather/today/l/USCA0987:1:US",         "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results  
  
for i in search(query):
    print(i)
def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
        return "Weather information could not be found."
    # fetch data
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve the webpage."
    # parse result
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup 
url = search(query)[0]  

# scrape first wesbsite
soup = scrape_weather_info(url)
# print(f"Website: {url}\n\n")
# print(str(soup.body)[:50000])  
weather_data = []
for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
    text = tag.get_text(" ", strip=True)
    weather_data.append(text)
  
# combine all elements into a single string
weather_data = "\n".join(weather_data) 
# remove all spaces from the combined text
weather_data = re.sub(r'\s+', ' ', weather_data)
print(f"Website: {url}\n\n")
print(weather_data)
```
*different from agentic and regular search*

```py
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import re
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import json
from pygments import highlight,lexers,formatters

load_dotenv()
api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError("未找到 TAVILY_API_KEY，请检查 .env 文件")
  
client = TavilyClient(api_key=api_key)

city = "Bei Jing"

query = f"""
    what is your current weather in {city}?
    Should I travel there today?
"""
result = client.search(query,max_results=1)  

data =result["results"][0]["content"]
# print(data)
# parse JSON

parsed_json = json.loads(data.replace("'", '"'))

# pretty print JSON with syntax highlighting
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter()) 
print(colorful_json)
```

## Persistence and Streaming

Persistence
```py
"""
基于 DeepSeek 模型 + Tavily 搜索工具的 LangGraph 智能体示例
支持多轮对话和工具调用，使用内存检查点保存线程状态。
"""
# ======================= 1. 导入依赖 =======================

import os
from dotenv import load_dotenv
# 加载 .env 文件中的环境变量（API 密钥等）
_ = load_dotenv()  
# LangGraph 核心组件
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # 内存检查点（简便，无需 SQLite） 
# 类型定义
from typing import TypedDict, Annotated
import operator 
# LangChain 消息类型
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
# DeepSeek 模型（OpenAI 兼容）
from langchain_openai import ChatOpenAI
# Tavily 搜索工具（新版包，需安装 langchain-tavily）
from langchain_tavily import TavilySearch  # 替代已弃用的 TavilySearchResults

# ======================= 2. 定义状态结构 =======================
class AgentState(TypedDict):
    """智能体的状态结构，使用 Annotated 实现消息自动累加"""
    messages: Annotated[list[AnyMessage], operator.add]  
# ======================= 3. 初始化工具 =======================
# 创建 Tavily 搜索工具实例，每次返回最多 2 条结果
tool = TavilySearch(max_results=2) 
# ======================= 4. 定义 Agent 类 =======================
class Agent:
    """
    支持工具调用的智能体封装类
    使用 LangGraph 构建执行图，包含 LLM 节点和动作节点，
    并自动路由：无工具调用则结束，有调用则执行工具后返回 LLM。
    """  
    def __init__(self, model, tools, checkpointer, system=""):
        """
        参数:
            model: 绑定工具后的 LLM 模型实例
            tools: 工具列表（每个工具应有 name, invoke 等方法）
            checkpointer: 检查点保存器（例如 MemorySaver）
            system: 系统提示词（可选）
        """
        self.system = system
        # 构建 StateGraph，状态类型为 AgentState
        graph = StateGraph(AgentState)

        # 添加两个节点
        graph.add_node("llm", self.call_openai)      # LLM 推理节点
        graph.add_node("action", self.take_action)   # 工具执行节点

        # 添加条件边：从 llm 节点出发，根据 exists_action 决定跳转到 action 或 END
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        # 添加普通边：action 执行完毕后回到 llm 节点
        graph.add_edge("action", "llm")
        # 设置入口点为 llm
        graph.set_entry_point("llm")
        # 编译图，传入检查点保存器（支持多线程对话记忆）
        self.graph = graph.compile(checkpointer=checkpointer)
        # 将工具列表转换为 {工具名: 工具对象} 映射，便于快速调用
        self.tools = {t.name: t for t in tools}
        # 模型绑定工具（使模型能够生成 tool_calls）
        self.model = model.bind_tools(tools)
  
    def call_openai(self, state: AgentState):
        """LLM 节点：调用模型生成回复（可能包含工具调用请求）"""
        messages = state['messages']
        # 如果设置了系统提示词，将其插入到消息列表最前面
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        # 调用模型
        response = self.model.invoke(messages)
        # 返回新消息（将被自动添加到状态中的 messages 列表）
        return {'messages': [response]} 

    def exists_action(self, state: AgentState):
        """条件判断函数：检查最后一条消息是否包含工具调用"""
        last_message = state['messages'][-1]
        # 如果有 tool_calls 属性且非空，则返回 True
        return len(last_message.tool_calls) > 0

    def take_action(self, state: AgentState):
        """动作节点：执行工具调用，并将结果封装为 ToolMessage 返回"""
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls 
        results = []
        for tool_call in tool_calls:
            print(f"调用工具: {tool_call}")   # 调试输出
            # 根据工具名获取工具实例，并调用其 invoke 方法
            tool_result = self.tools[tool_call['name']].invoke(tool_call['args'])
            # 构造 ToolMessage，必须包含 tool_call_id 和原工具名
            results.append(
                ToolMessage(
                    tool_call_id=tool_call['id'],
                    name=tool_call['name'],
                    content=str(tool_result)
                )
            )
        print("工具执行完毕，返回模型继续推理...")
        return {'messages': results}

# ======================= 5. 配置模型和检查点 =======================
# 设置系统提示词（指导智能体合理使用搜索工具）

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!

"""
# 创建 DeepSeek 模型实例（使用与 OpenAI 兼容的接口）
model = ChatOpenAI(
    model="deepseek-chat",                     # DeepSeek 对话模型
    base_url="https://api.deepseek.com/v1",    # DeepSeek API 地址
    api_key=os.getenv("DEEPSEEK_API_KEY")      # 从环境变量读取 API Key
)
# 创建内存检查点保存器（程序重启后记忆丢失，但无需 SQLite 依赖）
memory = MemorySaver()
# 实例化智能体
abot = Agent(model, [tool], system=prompt, checkpointer=memory)
# ======================= 6. 多线程对话示例 =======================
# 定义线程配置（每个 thread_id 代表一个独立的对话会话）
thread_1 = {"configurable": {"thread_id": "1"}}
thread_2 = {"configurable": {"thread_id": "2"}}
# ---- 线程1：询问旧金山天气 ----
print("=== 线程1：询问旧金山天气 ===")
messages = [HumanMessage(content="What is the weather in sf?")]
for event in abot.graph.stream({"messages": messages}, thread_1):
    # 遍历流式输出的事件，打印每个节点的消息
    for node_name, node_output in event.items():
        print(f"[{node_name}] 输出: {node_output['messages'][-1].content}")  

# ---- 线程1：继续询问洛杉矶天气（利用记忆） ----
print("\n=== 线程1：询问洛杉矶天气（同一会话） ===")
messages = [HumanMessage(content="What about in la?")]
for event in abot.graph.stream({"messages": messages}, thread_1):
    for node_name, node_output in event.items():
        print(f"[{node_name}] 输出: {node_output['messages'][-1].content}")


# ---- 线程1：比较两个城市温度 ----
print("\n=== 线程1：询问哪个更暖和（同一会话） ===")
messages = [HumanMessage(content="Which one is warmer?")]
for event in abot.graph.stream({"messages": messages}, thread_1):
    for node_name, node_output in event.items():
        print(f"[{node_name}] 输出: {node_output['messages'][-1].content}")

# ---- 线程2：新会话，同样的问题（无上下文记忆） ----
print("\n=== 线程2：新会话询问哪个更暖和（无历史） ===")
messages = [HumanMessage(content="Which one is warmer?")]
for event in abot.graph.stream({"messages": messages}, thread_2):
    for node_name, node_output in event.items():
        print(f"[{node_name}] 输出: {node_output['messages'][-1].content}")
```

Streaming
```py
"""
基于 DeepSeek 模型 + Tavily 搜索工具的 LangGraph 智能体
- 使用异步 SQLite 检查点（支持多线程记忆）
- 使用 astream_events 实现流式输出
- 修正导入路径为 langgraph.checkpoint.sqlite.aio
"""

import os
import asyncio
from dotenv import load_dotenv
_ = load_dotenv()
  

# 正确的异步 SQLite 检查点导入路径
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch  # 需安装 langchain-tavily

# ======================= 状态定义 =======================
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
# ======================= 工具 =======================

tool = TavilySearch(max_results=2)  
# ======================= Agent 类 =======================
class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        # 编译时传入有效的 checkpointer 实例
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        response = self.model.invoke(messages)
        return {'messages': [response]}
  
    def exists_action(self, state: AgentState):
        last_message = state['messages'][-1]
        return len(last_message.tool_calls) > 0

    def take_action(self, state: AgentState):
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls
        results = []
        for t in tool_calls:
            print(f"调用工具: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("工具执行完毕，返回模型继续推理...")
        return {'messages': results}

# ======================= 系统提示词 =======================
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
""" 
# ======================= 模型配置（DeepSeek） =======================
model = ChatOpenAI(
    model="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    api_key=os.getenv("DEEPSEEK_API_KEY")
)


# ======================= 异步主函数 =======================
async def main():
    # 重要：AsyncSqliteSaver.from_conn_string 返回异步上下文管理器，
    # 必须用 async with 才能获得真正的 checkpointer 实例
    async with AsyncSqliteSaver.from_conn_string(":memory:") as memory:
        abot = Agent(model, [tool], system=prompt, checkpointer=memory)
        thread = {"configurable": {"thread_id": "4"}}
        messages = [HumanMessage(content="What is the weather in SF?")]
        # 流式输出
        async for event in abot.graph.astream_events(
            {"messages": messages}, thread, version="v1"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:  # 过滤掉工具调用时的空内容
                    print(content, end="|")
        print("\n")  # 换行结束
        
        # 可选：同一线程继续对话，验证记忆功能

        print("继续询问洛杉矶天气：")

        messages2 = [HumanMessage(content="What about in LA?")]

        async for event in abot.graph.astream_events(
            {"messages": messages2}, thread, version="v1"
        ):
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="|")
        print("\n")

if __name__ == "__main__":
    asyncio.run(main())
```
执行结果

Let| me| look| up| the| current| weather| in| San| Francisco|.|调用工具: {'name': 'tavily_search', 'args': {'query': 'current weather in San Francisco', 'search_depth': 'basic'}, 'id': 'call_00_QaGKAAhxxDmgUExKL9Md7517', 'type': 'tool_call'}
工具执行完毕，返回模型继续推理...
Let| me| try| again|.|调用工具: {'name': 'tavily_search', 'args': {'query': 'San Francisco weather today', 'search_depth': 'basic'}, 'id': 'call_00_0t5rBYaVJrV0Iry7XtIJ8882', 'type': 'tool_call'}
工具执行完毕，返回模型继续推理...
Here|'s| the| current| weather| in| San| Francisco| as| of| late| evening| on| **|May| |8|,| |202|6|**|:

|🌡|️| **|Temperature|**:| |54|°|F| (|12|.|2|°|C|),| but| feels| like| **|52|.|6|°|F| (|11|.|4|°|C|)|**
|⛅| **|Conditions|**:| Part|ly| cloudy|
|💨| **|Wind|**:| WSW| at| |5|.|6| mph| (|gust|ing| up| to| |8|.|6| mph|)
|💧| **|Hum|idity|**:| |86|%
|☁|️| **|Cloud| cover|**:| |50|%
|🌧|️| **|Ch|ance| of| rain|**:| |0|%
|👁|️| **|Visibility|**:| |9| miles|

|It|'s| a| cool|,| partly| cloudy| evening| in| SF| —| typical| for| this| time| of| year|!| If| you|'re| planning| for| tomorrow| (|May| |9|),| it|'s| expected| to| be| sunny| with| a| high| around| **|66|°|F| (|19|°|C|)**| in| the| afternoon|.|

继续询问洛杉矶天气：
调用工具: {'name': 'tavily_search', 'args': {'query': 'current weather in Los Angeles', 'search_depth': 'basic'}, 'id': 'call_00_DU0izSYvzKayq7bI5btZ5735', 'type': 'tool_call'}
工具执行完毕，返回模型继续推理...
Here|'s| the| current| weather| in| **|Los| Angeles|**| as| of| the| same| time| (|even|ing| of| May| |8|,| |202|6|):

|🌡|️| **|Temperature|**:| |61|°|F| (|16|.|1|°|C|),| feels| like| **|61|°|F|**
|☁|️| **|Conditions|**:| Over|cast| (|cloud|y|)
|💨| **|Wind|**:| South| at| |3|.|1| mph|
|💧| **|Hum|idity|**:| |83|%
|☁|️| **|Cloud| cover|**:| |100|%| (|full| over|cast|)
|🌧|️| **|Ch|ance| of| rain|**:| |0|%
|👁|️| **|Visibility|**:| |8| miles|

|**|Quick| comparison| —| SF| vs| LA|:**

||| || San| Francisco| || Los| Angeles| |
||---|---||---||
||| 🌡|️| Temp| || |54|°|F| (|12|°|C|)| || |61|°|F| (|16|°|C|)| |
||| ☁|️| Sky| || Part|ly| cloudy| || Over|cast| |
||| 💨| Wind| || WSW| |5|.|6| mph| || S| |3|.|1| mph| |
||| 💧| Humidity| || |86|%| || |83|%| |

|LA| is| about| **|7|°|F| warmer|**| than| SF| tonight|,| though| it|'s| more| over|cast|.| Both| are| dry| with| no| rain| expected|!|
