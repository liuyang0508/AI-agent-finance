"""
项目全局配置文件
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
MARKDOWN_DIR = DATA_DIR / "markdown"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# 报告输出目录
REPORTS_DIR = BASE_DIR / "reports"

# GLM API 配置（兼容 OpenAI 接口格式）
GLM_API_KEY = os.getenv("GLM_API_KEY", "")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-plus")

# 嵌入模型（GLM embedding-2，兼容 OpenAI 接口）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-2")

# ChromaDB 配置
CHROMA_COLLECTION_NAME = "financial_reports"

# RAG 配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVE = 10
TOP_K_RERANK = 5
