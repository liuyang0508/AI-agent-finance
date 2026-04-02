"""
向量化与存储模块

负责将 Markdown 文档切块、向量化，并持久化存储到 ChromaDB。
提供增量入库（跳过已存在文档）与清空重建两种模式。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config.settings import (
    MARKDOWN_DIR, VECTORSTORE_DIR,
    GLM_API_KEY, GLM_BASE_URL, EMBEDDING_MODEL,
    CHROMA_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
)


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    """向量化入库请求"""
    markdown_dir: str = Field(
        default=str(MARKDOWN_DIR),
        description="Markdown 文件所在目录，默认为 data/markdown/"
    )
    overwrite: bool = Field(
        default=False,
        description="True：清空已有向量库重建；False：增量追加（跳过已入库文件）"
    )


class EmbedResult(BaseModel):
    """向量化入库结果"""
    total_files: int = Field(..., description="扫描到的 Markdown 文件总数")
    embedded_files: int = Field(..., description="本次实际入库的文件数")
    skipped_files: int = Field(..., description="跳过的文件数（已存在）")
    total_chunks: int = Field(..., description="本次入库的文本块总数")
    vectorstore_path: str = Field(..., description="ChromaDB 持久化路径")


# ── 向量化器 ──────────────────────────────────────────────────────────────────

class DocumentEmbedder:
    """
    文档向量化器

    流程：Markdown 文件 → 文本切块 → GLM Embeddings → ChromaDB 持久化
    """

    def __init__(self):
        """初始化 GLM Embedding 模型与 ChromaDB 客户端"""
        self._embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # 财报场景：优先在段落、换行、表格行边界处切分
            separators=["\n\n", "\n", "。", "；", " ", ""],
        )
        self._vectorstore: Optional[Chroma] = None

    def embed(self, request: EmbedRequest) -> EmbedResult:
        """
        执行向量化入库

        Args:
            request: 入库请求参数

        Returns:
            EmbedResult 入库统计结果
        """
        md_dir = Path(request.markdown_dir)
        md_files = list(md_dir.glob("*.md"))

        if not md_files:
            print(f"未在 {md_dir} 中找到 Markdown 文件，请先运行 PDF 解析模块。")
            return EmbedResult(
                total_files=0, embedded_files=0,
                skipped_files=0, total_chunks=0,
                vectorstore_path=str(VECTORSTORE_DIR),
            )

        # 获取或创建向量库
        vectorstore = self._get_vectorstore(reset=request.overwrite)

        # 查询已入库的文件名，用于增量跳过
        existing_sources = self._get_existing_sources(vectorstore)

        embedded_files = 0
        skipped_files = 0
        total_chunks = 0

        for md_file in md_files:
            if md_file.name in existing_sources and not request.overwrite:
                print(f"跳过（已入库）：{md_file.name}")
                skipped_files += 1
                continue

            docs = self._load_and_split(md_file)
            if docs:
                vectorstore.add_documents(docs)
                total_chunks += len(docs)
                embedded_files += 1
                print(f"已入库：{md_file.name}（{len(docs)} 个文本块）")

        # 持久化（新版 ChromaDB 自动持久化，此处保留显式调用兼容旧版）
        try:
            vectorstore.persist()
        except AttributeError:
            pass  # 新版 ChromaDB 不需要手动 persist

        self._vectorstore = vectorstore
        print(f"\n入库完成：{embedded_files} 个文件，{total_chunks} 个文本块")

        return EmbedResult(
            total_files=len(md_files),
            embedded_files=embedded_files,
            skipped_files=skipped_files,
            total_chunks=total_chunks,
            vectorstore_path=str(VECTORSTORE_DIR),
        )

    def get_vectorstore(self) -> Chroma:
        """获取已初始化的向量库实例（供检索模块调用）"""
        if self._vectorstore is None:
            self._vectorstore = self._get_vectorstore(reset=False)
        return self._vectorstore

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _get_vectorstore(self, reset: bool) -> Chroma:
        """创建或加载 ChromaDB 向量库"""
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        if reset and VECTORSTORE_DIR.exists():
            import shutil
            shutil.rmtree(VECTORSTORE_DIR)
            VECTORSTORE_DIR.mkdir(parents=True)
            print("已清空旧向量库，重新构建。")

        return Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self._embedding_model,
            persist_directory=str(VECTORSTORE_DIR),
        )

    def _get_existing_sources(self, vectorstore: Chroma) -> set[str]:
        """查询向量库中已存在的文件名集合（metadata.source 字段）"""
        try:
            result = vectorstore.get(include=["metadatas"])
            return {
                Path(m.get("source", "")).name
                for m in result.get("metadatas", [])
                if m
            }
        except Exception:
            return set()

    def _load_and_split(self, md_file: Path) -> list[Document]:
        """读取 Markdown 文件并切块，每块携带来源元数据"""
        text = md_file.read_text(encoding="utf-8")
        chunks = self._splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={"source": md_file.name, "file_path": str(md_file)},
            )
            for chunk in chunks
        ]
