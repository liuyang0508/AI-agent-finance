"""
实时数据 Agent

职责：通过 MCP 工具获取最新股价、新闻，必要时调用财务计算器。
模式：标准 ReAct —— 使用 LangGraph 的 create_react_agent 构建，
      工具调用失败时自动重试（由 ReAct 循环处理）。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL
from src.agents.state import AgentState, NODE_SUPERVISOR
from src.mcp.tools import (
    get_stock_quote, get_financial_news, calculate_financial_metric,
)

# ── 将 MCP 工具函数包装为 LangChain @tool ─────────────────────────────────────

@tool
def tool_get_stock_quote(symbol: str) -> str:
    """
    获取实时股价行情。
    参数 symbol：股票代码，A股加后缀如 600519.SH，美股直接填 AAPL。
    返回最新价、涨跌幅、成交量等信息。
    """
    try:
        result = get_stock_quote(symbol)
        return result.model_dump_json(ensure_ascii=False)
    except Exception as e:
        return f"获取行情失败：{e}"


@tool
def tool_get_financial_news(query: str, limit: int = 5) -> str:
    """
    获取金融新闻资讯和情感分析。
    参数 query：公司名称或行业关键词；limit：返回条数（1-20）。
    返回新闻标题、摘要、情感倾向（Bullish/Bearish/Neutral）。
    """
    try:
        result = get_financial_news(query, limit)
        return result.model_dump_json(ensure_ascii=False)
    except Exception as e:
        return f"获取新闻失败：{e}"


@tool
def tool_calculate_financial_metric(metric: str, params: dict) -> str:
    """
    精确计算财务指标，避免 LLM 数学幻觉。
    支持：pe_ratio、pb_ratio、roe、gross_margin、net_margin、
          debt_ratio、cagr、ev_ebitda。
    参数 params：计算所需的数值字典，如 {"price": 1800, "eps": 70}。
    """
    try:
        result = calculate_financial_metric(metric, params)
        return result.model_dump_json(ensure_ascii=False)
    except Exception as e:
        return f"计算失败：{e}"


_REALTIME_TOOLS = [
    tool_get_stock_quote,
    tool_get_financial_news,
    tool_calculate_financial_metric,
]

# ReAct Agent 系统提示词
_REALTIME_SYSTEM_PROMPT = (
    "你是一位专注于实时市场数据的金融助手。\n"
    "你拥有以下工具：获取实时股价、获取最新新闻、精确计算财务指标。\n"
    "请根据用户问题，选择合适工具获取数据，并将结果整理为简洁的摘要。\n"
    "如果工具调用失败，请尝试调整参数后重试，最多重试 2 次。\n"
    "最终输出：用 Markdown 格式呈现行情数据和新闻摘要。"
)


class RealtimeAgent:
    """
    实时数据 Agent

    内部使用 LangGraph create_react_agent 构建完整 ReAct 循环，
    工具调用失败时由 ReAct 自动处理重试逻辑。
    """

    def __init__(self):
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )
        # 创建内置 ReAct Agent（自动处理 Tool Call → Observation → 下一步）
        self._react_agent = create_react_agent(
            model=self._llm,
            tools=_REALTIME_TOOLS,
        )

    def __call__(self, state: AgentState) -> AgentState:
        """实时数据 Agent 节点入口"""
        query = state.get("user_query", "")
        print(f"[实时数据 Agent] 开始获取实时数据：{query[:50]}...")

        # 调用内置 ReAct Agent
        react_input = {
            "messages": [
                HumanMessage(content=f"{_REALTIME_SYSTEM_PROMPT}\n\n用户问题：{query}")
            ]
        }
        result = self._react_agent.invoke(react_input)

        # 提取最终回复（最后一条 AIMessage）
        final_message = result["messages"][-1]
        realtime_data = final_message.content if hasattr(final_message, "content") else str(final_message)

        print(f"[实时数据 Agent] 完成，输出长度：{len(realtime_data)} 字符")

        return {
            **state,
            "realtime_data": realtime_data,
            "next_agent": NODE_SUPERVISOR,
        }
