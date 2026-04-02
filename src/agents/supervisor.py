"""
Supervisor Agent

职责：意图识别 → 任务分配 → 结果审核
特点：不直接执行任何分析，只负责"指挥"和"审核"。

意图识别使用 Few-shot Prompt + GLM，无需 semantic-router 依赖。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config.settings import GLM_API_KEY, GLM_BASE_URL, LLM_MODEL
from src.agents.state import (
    AgentState,
    INTENT_INDUSTRY, INTENT_STOCK, INTENT_FINANCIAL, INTENT_REALTIME,
    NODE_RETRIEVAL, NODE_FINANCIAL, NODE_RISK, NODE_REALTIME, NODE_REPORT, NODE_END,
    MAX_ITERATIONS,
)

# Few-shot 示例提示词，帮助 GLM 准确分类四种意图
_INTENT_PROMPT = (
    "你是一个金融问题意图分类器。将用户问题归类为以下四种意图之一，"
    "只输出意图标签，不要任何解释。\n\n"
    f"意图标签：\n"
    f"- {INTENT_INDUSTRY}：询问某个行业的整体情况、趋势、竞争格局\n"
    f"- {INTENT_STOCK}：对比或分析多只个股的财务或估值\n"
    f"- {INTENT_FINANCIAL}：解读某公司的财报、财务指标、经营数据\n"
    f"- {INTENT_REALTIME}：查询实时股价、今日涨跌、最新新闻资讯\n\n"
    "示例：\n"
    f"问题：分析新能源汽车行业的发展趋势 → {INTENT_INDUSTRY}\n"
    f"问题：对比茅台和五粮液的估值水平 → {INTENT_STOCK}\n"
    f"问题：解读贵州茅台2023年年报 → {INTENT_FINANCIAL}\n"
    f"问题：茅台今天股价多少 → {INTENT_REALTIME}\n\n"
    "问题：{query} → "
)


class SupervisorAgent:
    """
    Supervisor Agent

    LangGraph 节点函数：接收 AgentState，识别意图，决定下一步路由目标。
    同时承担"结果审核"角色：当所有子 Agent 完成后，判断是否进入报告生成。
    """

    def __init__(self):
        """初始化 GLM LLM"""
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )

    def __call__(self, state: AgentState) -> AgentState:
        """
        Supervisor 节点入口

        根据当前状态决定：
          1. 首次进入 → 识别意图 → 分配首个子 Agent
          2. 子 Agent 完成后回到 Supervisor → 审核并分配下一个任务
          3. 所有任务完成 → 路由到报告生成节点
        """
        iteration = state.get("iteration", 0)

        if iteration >= MAX_ITERATIONS:
            print(f"[Supervisor] 已达最大迭代次数 {MAX_ITERATIONS}，强制进入报告生成")
            return {**state, "next_agent": NODE_REPORT, "iteration": iteration + 1}

        query = state.get("user_query", "")

        # 首次进入：识别意图
        if state.get("intent") is None:
            intent = self._recognize_intent(query)
            print(f"[Supervisor] 意图识别结果：{intent}")
            next_agent = self._dispatch_first_agent(intent)
            return {
                **state,
                "intent": intent,
                "next_agent": next_agent,
                "iteration": iteration + 1,
            }

        # 后续进入：审核已完成的结果，决定下一步
        next_agent = self._review_and_dispatch(state)
        print(f"[Supervisor] 审核完成，下一节点：{next_agent}")
        return {**state, "next_agent": next_agent, "iteration": iteration + 1}

    # ── 意图识别 ──────────────────────────────────────────────────────────────

    def _recognize_intent(self, query: str) -> str:
        """使用 Few-shot Prompt 调用 GLM 识别用户意图"""
        prompt = _INTENT_PROMPT.format(query=query)
        response = self._llm.invoke([HumanMessage(content=prompt)])
        intent = response.content.strip()

        valid = {INTENT_INDUSTRY, INTENT_STOCK, INTENT_FINANCIAL, INTENT_REALTIME}
        if intent not in valid:
            # 模型返回了多余文字，尝试从回复中提取标签
            for label in valid:
                if label in intent:
                    return label
            return INTENT_FINANCIAL  # 兜底默认财报解读

        return intent

    # ── 首次任务分配 ──────────────────────────────────────────────────────────

    def _dispatch_first_agent(self, intent: str) -> str:
        """根据意图分配第一个执行的子 Agent"""
        mapping = {
            INTENT_REALTIME:  NODE_REALTIME,
            INTENT_FINANCIAL: NODE_RETRIEVAL,
            INTENT_INDUSTRY:  NODE_RETRIEVAL,
            INTENT_STOCK:     NODE_RETRIEVAL,
        }
        return mapping.get(intent, NODE_RETRIEVAL)

    # ── 结果审核与后续分配 ────────────────────────────────────────────────────

    def _review_and_dispatch(self, state: AgentState) -> str:
        """
        审核已完成结果，决定下一个执行节点

        标准流程：检索 → 财务分析 → 风控 → 实时数据 → 报告
        实时查询：直接获取行情后生成报告，跳过 RAG 和分析环节
        """
        intent = state.get("intent", "")

        if intent == INTENT_REALTIME:
            if state.get("realtime_data") and not state.get("final_report"):
                return NODE_REPORT
            if state.get("final_report"):
                return NODE_END
            return NODE_REALTIME

        if not state.get("rag_context"):
            return NODE_RETRIEVAL
        if not state.get("financial_analysis"):
            return NODE_FINANCIAL
        if not state.get("risk_assessment"):
            return NODE_RISK
        if not state.get("realtime_data"):
            return NODE_REALTIME
        if not state.get("final_report"):
            return NODE_REPORT
        return NODE_END


# ── LangGraph 路由函数（边条件）──────────────────────────────────────────────

def route_supervisor(state: AgentState) -> str:
    """读取 state.next_agent，决定下一个节点名称"""
    return state.get("next_agent", NODE_END)
