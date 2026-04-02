# 项目名称
企业级智能投研与研报自动化助手

# 技术栈
- 语言：Python 3.10+
- 核心框架：LangGraph（Multi-Agent 编排）
- RAG：LangChain + ChromaDB 向量数据库
- PDF 解析：Marker
- 意图识别：Semantic Router
- 工具协议：MCP
- 评测：Ragas
- 包管理：pip，所有依赖写入 requirements.txt
- IDE：PyCharm

# 项目目标
构建一个 Multi-Agent 系统，实现从用户提问到自动生成深度研究报告的全流程。

# 系统架构
采用 Hierarchical（层级）架构：
- Supervisor Agent：负责意图识别、任务分配、结果审核，不直接执行
- Sub-Agent（每个都是独立 ReAct 节点）：
  - 财务分析 Agent：负责财报解读
  - 风控 Agent：负责风险评估
  - 检索 Agent：负责 RAG 数据检索
  - 实时数据 Agent：通过 MCP 获取股价/新闻

# 核心功能模块（按顺序开发）
1. PDF 解析模块：用 Marker 将财报 PDF 转为 Markdown，处理跨页表格
2. RAG 模块：向量化存储 + HyDE 检索 + Rerank 重排序
3. MCP 工具模块：接入外部金融 API 获取实时数据
4. LangGraph 编排：Supervisor + Sub-Agent 状态机
5. 报告生成：输出 Markdown 格式研报
6. Ragas 评测：自动化评测报告质量

# 代码规范
- 所有注释和 docstring 用中文
- 每个模块单独一个文件
- 使用 Pydantic 定义所有工具的 schema
- 不要一次性生成所有代码，每完成一个模块告诉我，等我确认后再继续

# 开发顺序
请严格按照以下顺序开发，每步完成后暂停等待我确认：
1. 先搭建项目目录结构
2. 实现 PDF 解析模块
3. 实现 RAG 模块
4. 实现 MCP 工具模块
5. 实现 LangGraph 多 Agent 编排
6. 实现报告生成与 Ragas 评测