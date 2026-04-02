"""
LangGraph 多 Agent 编排图

将所有 Agent 节点组装为有向状态图（StateGraph），
定义节点间的边和条件路由，输出可执行的 CompiledGraph。

图结构：
    START
      ↓
  [supervisor] ──条件路由──→ [retrieval_agent]
      ↑                    → [financial_agent]
      │                    → [risk_agent]
      │                    → [realtime_agent]
      └──── 各 Agent 完成后回到 supervisor ────┘
      ↓（所有任务完成）
  [report_generator]
      ↓
    END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import (
    AgentState,
    NODE_SUPERVISOR, NODE_RETRIEVAL, NODE_FINANCIAL,
    NODE_RISK, NODE_REALTIME, NODE_REPORT, NODE_END,
)
from src.agents.supervisor import SupervisorAgent, route_supervisor
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.financial_agent import FinancialAgent
from src.agents.risk_agent import RiskAgent
from src.agents.realtime_agent import RealtimeAgent
from src.report.generator import ReportGenerator
from src.rag.embedder import DocumentEmbedder, EmbedRequest
from src.rag.reranker import RAGPipeline


def build_graph(use_checkpointer: bool = True):
    """
    构建并编译 LangGraph 多 Agent 图

    Args:
        use_checkpointer: 是否启用内存检查点（支持断点续传和流式输出）

    Returns:
        CompiledGraph  可直接调用 .invoke() 的编译图
    """
    # ── 初始化各组件 ──────────────────────────────────────────────────────────
    print("正在初始化各 Agent 组件...")

    # RAG Pipeline（检索 Agent 依赖）
    embedder = DocumentEmbedder()
    # 仅在向量库为空时自动入库（首次运行）
    embedder.embed(EmbedRequest(overwrite=False))
    vectorstore = embedder.get_vectorstore()
    rag_pipeline = RAGPipeline(vectorstore=vectorstore, use_rerank=True)

    # 实例化各 Agent
    supervisor    = SupervisorAgent()
    retrieval     = RetrievalAgent(rag_pipeline=rag_pipeline)
    financial     = FinancialAgent()
    risk          = RiskAgent()
    realtime      = RealtimeAgent()
    report_gen    = ReportGenerator()

    # ── 构建状态图 ────────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)

    # 注册节点
    graph.add_node(NODE_SUPERVISOR, supervisor)
    graph.add_node(NODE_RETRIEVAL,  retrieval)
    graph.add_node(NODE_FINANCIAL,  financial)
    graph.add_node(NODE_RISK,       risk)
    graph.add_node(NODE_REALTIME,   realtime)
    graph.add_node(NODE_REPORT,     report_gen)

    # 起始边：START → supervisor
    graph.add_edge(START, NODE_SUPERVISOR)

    # Supervisor 条件路由（核心边）
    graph.add_conditional_edges(
        NODE_SUPERVISOR,
        route_supervisor,
        {
            NODE_RETRIEVAL: NODE_RETRIEVAL,
            NODE_FINANCIAL: NODE_FINANCIAL,
            NODE_RISK:      NODE_RISK,
            NODE_REALTIME:  NODE_REALTIME,
            NODE_REPORT:    NODE_REPORT,
            NODE_END:       END,
        },
    )

    # 各 Sub-Agent 完成后均回到 Supervisor 审核
    for node in [NODE_RETRIEVAL, NODE_FINANCIAL, NODE_RISK, NODE_REALTIME]:
        graph.add_edge(node, NODE_SUPERVISOR)

    # 报告生成后结束
    graph.add_edge(NODE_REPORT, END)

    # ── 编译图 ────────────────────────────────────────────────────────────────
    checkpointer = MemorySaver() if use_checkpointer else None
    compiled = graph.compile(checkpointer=checkpointer)

    print("LangGraph 多 Agent 图编译完成。")
    return compiled


def run_query(question: str, thread_id: str = "default") -> str:
    """
    便捷函数：执行一次完整的投研查询

    Args:
        question:  用户问题
        thread_id: 会话 ID（用于 Checkpointer 断点续传）

    Returns:
        最终生成的研究报告（Markdown 字符串）
    """
    graph = build_graph(use_checkpointer=True)

    initial_state: AgentState = {
        "messages":           [],
        "user_query":         question,
        "intent":             None,
        "next_agent":         None,
        "rag_context":        None,
        "financial_analysis": None,
        "risk_assessment":    None,
        "realtime_data":      None,
        "final_report":       None,
        "iteration":          0,
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(initial_state, config=config)

    return final_state.get("final_report", "报告生成失败，请检查日志。")
