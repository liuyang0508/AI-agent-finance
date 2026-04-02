"""
LangGraph 共享状态定义

所有 Agent 节点共享同一个 AgentState，通过状态字段传递中间结果。
使用 TypedDict + Annotated 实现消息列表的追加合并语义。
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    多 Agent 系统的全局共享状态

    字段说明：
      messages        — 完整对话历史，使用 add_messages 自动追加（不覆盖）
      user_query      — 用户原始问题
      intent          — Supervisor 识别出的意图标签
      next_agent      — Supervisor 指定的下一个执行节点名称
      rag_context     — 检索 Agent 返回的文档上下文（Markdown 格式）
      financial_analysis — 财务分析 Agent 的分析结论
      risk_assessment — 风控 Agent 的风险评估结论
      realtime_data   — 实时数据 Agent 返回的行情/新闻摘要
      final_report    — 最终生成的研究报告（Markdown）
      iteration       — 当前迭代轮次，防止无限循环
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    intent: Optional[str]
    next_agent: Optional[str]
    rag_context: Optional[str]
    financial_analysis: Optional[str]
    risk_assessment: Optional[str]
    realtime_data: Optional[str]
    final_report: Optional[str]
    iteration: int


# 意图标签常量（与 Semantic Router 路由名称对应）
INTENT_INDUSTRY   = "industry_overview"   # 行业综述
INTENT_STOCK      = "stock_comparison"    # 个股对比
INTENT_FINANCIAL  = "financial_report"    # 财报解读
INTENT_REALTIME   = "realtime_query"      # 实时行情/新闻

# Agent 节点名称常量
NODE_SUPERVISOR   = "supervisor"
NODE_RETRIEVAL    = "retrieval_agent"
NODE_FINANCIAL    = "financial_agent"
NODE_RISK         = "risk_agent"
NODE_REALTIME     = "realtime_agent"
NODE_REPORT       = "report_generator"
NODE_END          = "__end__"

# 最大迭代轮次（防止死循环）
MAX_ITERATIONS = 10
