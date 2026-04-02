"""
Rerank 重排序模块

检索阶段（向量相似度）召回的是"语义相近"的文档，但不一定是"最相关"的。
Rerank 使用交叉编码器（Cross-Encoder）对召回结果精排，显著提升精度。

实现策略（可配置）：
  1. Cross-Encoder（默认）：本地模型，无需 API，速度快
     模型：BAAI/bge-reranker-v2-m3（中英文双语，适合财报场景）
  2. LLM Rerank（备选）：调用 LLM 打分，更准确但慢且有成本
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from config.settings import TOP_K_RERANK


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class RerankRequest(BaseModel):
    """重排序请求"""
    query: str = Field(..., description="用户原始查询问题")
    documents: list[dict] = Field(
        ...,
        description="待重排文档列表，每项需含 'content' 和 'source' 字段"
    )
    top_k: int = Field(default=TOP_K_RERANK, description="重排后返回的文档数量")


class RerankResult(BaseModel):
    """重排序结果"""
    query: str = Field(..., description="原始查询")
    documents: list[dict] = Field(
        ...,
        description="重排后的文档列表，每项含 content、source、score 字段"
    )


# ── Reranker ──────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Cross-Encoder 重排序器

    使用 sentence-transformers 的交叉编码器对召回文档重新打分并排序。
    Cross-Encoder 同时编码「查询 + 文档」，比双编码器更准确。
    """

    # 推荐模型：支持中英文，适合财报场景
    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        初始化 Cross-Encoder 模型

        Args:
            model_name: HuggingFace 模型名称，首次使用会自动下载
        """
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
            print(f"Reranker 模型加载完成：{model_name}")
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers：pip install sentence-transformers"
            )

    def rerank(self, request: RerankRequest) -> RerankResult:
        """
        执行重排序

        Args:
            request: 重排序请求

        Returns:
            RerankResult 精排后的 top-k 文档
        """
        if not request.documents:
            return RerankResult(query=request.query, documents=[])

        # 构建 (query, document) 对，Cross-Encoder 同时编码两段文本
        pairs = [(request.query, doc["content"]) for doc in request.documents]

        # 批量打分
        scores = self._model.predict(pairs)

        # 附加分数并排序
        scored_docs = [
            {**doc, "score": float(score)}
            for doc, score in zip(request.documents, scores)
        ]
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        top_docs = scored_docs[: request.top_k]

        print(
            f"重排序完成：{len(request.documents)} → {len(top_docs)} 个文档，"
            f"最高分 {top_docs[0]['score']:.4f}"
        )

        return RerankResult(query=request.query, documents=top_docs)


# ── 完整 RAG Pipeline 封装 ────────────────────────────────────────────────────

class RAGPipeline:
    """
    完整 RAG 流水线

    统一封装「检索 → HyDE → Rerank」三个阶段，供 Agent 直接调用。
    """

    def __init__(self, vectorstore, use_rerank: bool = True):
        """
        Args:
            vectorstore: 已初始化的 ChromaDB 向量库
            use_rerank: 是否启用 Rerank（调试时可关闭以加快速度）
        """
        from src.rag.retriever import HyDERetriever, RetrieveRequest
        self._retriever = HyDERetriever(vectorstore)
        self._reranker = CrossEncoderReranker() if use_rerank else None
        self._use_rerank = use_rerank

    def query(self, question: str, use_hyde: bool = True) -> list[dict]:
        """
        执行完整 RAG 查询

        Args:
            question: 用户问题
            use_hyde: 是否启用 HyDE

        Returns:
            重排后的文档列表（含 content、source、score 字段）
        """
        from src.rag.retriever import RetrieveRequest
        from config.settings import TOP_K_RETRIEVE, TOP_K_RERANK

        # 阶段一：HyDE 检索
        retrieve_result = self._retriever.retrieve(
            RetrieveRequest(query=question, use_hyde=use_hyde)
        )

        if not self._use_rerank:
            # 不 Rerank：直接返回，补充默认 score=1.0
            return [
                {**doc, "score": 1.0}
                for doc in retrieve_result.documents[:TOP_K_RERANK]
            ]

        # 阶段二：Cross-Encoder Rerank
        rerank_result = self._reranker.rerank(
            RerankRequest(
                query=question,
                documents=retrieve_result.documents,
                top_k=TOP_K_RERANK,
            )
        )

        return rerank_result.documents
