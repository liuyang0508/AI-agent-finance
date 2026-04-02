"""
HyDE 检索模块（Hypothetical Document Embeddings）

标准检索：直接将用户问题向量化后检索 → 语义偏差大（问题短，文档长）
HyDE 检索：先让 LLM 生成一段"假设性答案文档"，再用该文档去检索真实文档
           → 显著提升财报场景下的召回率
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config.settings import (
    GLM_API_KEY, GLM_BASE_URL, LLM_MODEL, TOP_K_RETRIEVE,
)


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="用户原始查询问题")
    top_k: int = Field(default=TOP_K_RETRIEVE, description="召回文档数量")
    use_hyde: bool = Field(default=True, description="是否启用 HyDE 增强检索")


class RetrieveResult(BaseModel):
    """检索结果"""
    query: str = Field(..., description="原始查询")
    hypothetical_doc: str = Field(default="", description="HyDE 生成的假设性文档（use_hyde=True 时有值）")
    documents: list[dict] = Field(..., description="检索到的文档列表，每项含 content 和 source")


# ── HyDE 检索器 ───────────────────────────────────────────────────────────────

# HyDE 提示词：引导 LLM 生成一段专业的金融分析片段，而非直接回答问题
_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位资深金融分析师。请根据用户的问题，生成一段 150-200 字的"
        "专业财务分析文本片段，就像这段文字出现在真实研究报告或财务报告中一样。"
        "不需要回答问题本身，只需生成风格匹配的文档片段。",
    ),
    ("human", "{query}"),
])


class HyDERetriever:
    """
    HyDE 检索器

    同时支持普通向量检索和 HyDE 增强检索，可通过 use_hyde 参数切换。
    """

    def __init__(self, vectorstore: Chroma):
        """
        Args:
            vectorstore: 已初始化的 ChromaDB 向量库实例
        """
        self._vectorstore = vectorstore
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0.3,  # 适度随机，生成多样化假设文档
        )
        self._hyde_chain = _HYDE_PROMPT | self._llm

    def retrieve(self, request: RetrieveRequest) -> RetrieveResult:
        """
        执行检索

        Args:
            request: 检索请求参数

        Returns:
            RetrieveResult 检索结果（未经 Rerank）
        """
        hypothetical_doc = ""

        if request.use_hyde:
            # 第一步：生成假设性文档
            hypothetical_doc = self._generate_hypothetical_doc(request.query)
            # 第二步：用假设文档向量检索真实文档
            search_text = hypothetical_doc
            print(f"[HyDE] 假设性文档已生成（{len(hypothetical_doc)} 字符），开始检索...")
        else:
            # 普通检索：直接用原始问题
            search_text = request.query

        raw_docs = self._vectorstore.similarity_search(
            search_text, k=request.top_k
        )

        documents = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知来源"),
            }
            for doc in raw_docs
        ]

        print(f"检索完成：召回 {len(documents)} 个文档块")

        return RetrieveResult(
            query=request.query,
            hypothetical_doc=hypothetical_doc,
            documents=documents,
        )

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _generate_hypothetical_doc(self, query: str) -> str:
        """调用 LLM 生成假设性文档"""
        try:
            response = self._hyde_chain.invoke({"query": query})
            return response.content.strip()
        except Exception as e:
            print(f"[HyDE] 假设文档生成失败，降级为普通检索：{e}")
            return query  # 降级：直接使用原始问题
