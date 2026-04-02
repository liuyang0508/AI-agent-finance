"""
风控 Agent

职责：基于财务分析结论和检索上下文，输出独立的风险评估报告。
评估维度：财务风险、行业风险、政策监管风险、估值风险。
设计原则：风控 Agent 与财务分析 Agent 独立运行，避免"回声室效应"。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL
from src.agents.state import AgentState, NODE_SUPERVISOR

# 风控专家系统提示词
_RISK_SYSTEM_PROMPT = """你是一位资深风险控制专家，专注于 A 股投资风险评估。

你的职责是**挑战**财务分析结论，而非认可它。请从以下维度识别潜在风险：

1. **财务风险**：资产负债率过高、商誉减值、应收账款异常、现金流与利润背离
2. **行业与竞争风险**：行业景气度下行、竞争格局恶化、技术替代风险
3. **政策监管风险**：行业政策收紧、反垄断、环保合规压力
4. **估值风险**：当前估值是否充分反映基本面，是否存在高估

输出要求：
- 每个风险点需说明"触发条件"和"潜在影响程度"（高/中/低）
- 给出综合风险等级：高风险 / 中等风险 / 低风险
- 提出 2-3 条风险规避建议
"""


class RiskAgent:
    """
    风控 Agent

    输入：state.rag_context + state.financial_analysis
    输出：state.risk_assessment（风险评估报告）
    """

    def __init__(self):
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0.2,  # 风控分析要求更保守、确定
        )

    def __call__(self, state: AgentState) -> AgentState:
        """风控 Agent 节点入口"""
        query = state.get("user_query", "")
        context = state.get("rag_context", "")
        financial_analysis = state.get("financial_analysis", "")
        print(f"[风控 Agent] 开始风险评估：{query[:50]}...")

        assessment = self._assess(query, context, financial_analysis)
        print(f"[风控 Agent] 评估完成，输出长度：{len(assessment)} 字符")

        return {
            **state,
            "risk_assessment": assessment,
            "next_agent": NODE_SUPERVISOR,
        }

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _assess(self, query: str, context: str, financial_analysis: str) -> str:
        """调用 LLM 进行风险评估"""
        user_prompt = (
            f"## 用户问题\n{query}\n\n"
            f"## 财务分析结论（需独立审视）\n{financial_analysis}\n\n"
            f"## 原始财报参考片段\n{context}\n\n"
            "请基于以上信息，撰写独立的风险评估报告。"
            "你的角色是风控专家，需要挑战财务分析中可能过于乐观的结论。"
        )
        messages = [
            SystemMessage(content=_RISK_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()
