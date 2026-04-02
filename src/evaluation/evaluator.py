"""
Ragas 自动化评测模块

对生成的研报进行多维度质量评估：
  - Faithfulness（忠实度）  ：研报结论是否忠实于检索上下文，无幻觉
  - Answer Relevancy（相关性）：研报是否切实回答了用户问题
  - Context Precision（上下文精确率）：检索到的文档是否真正有用
  - Context Recall（上下文召回率）  ：相关信息是否被充分检索到

评测结果保存为 JSON 文件，供后续优化参考。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import (
    GLM_API_KEY, GLM_BASE_URL, LLM_MODEL, EMBEDDING_MODEL, REPORTS_DIR,
)


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    """评测请求"""
    question: str = Field(..., description="用户原始问题")
    answer: str = Field(..., description="生成的研报内容（最终报告）")
    contexts: list[str] = Field(..., description="RAG 检索到的上下文片段列表")
    ground_truth: Optional[str] = Field(
        default=None,
        description="参考答案（可选）；有值时才能计算 context_recall"
    )


class EvalResult(BaseModel):
    """评测结果"""
    question: str = Field(..., description="原始问题")
    faithfulness: Optional[float] = Field(None, description="忠实度（0-1，越高越好）")
    answer_relevancy: Optional[float] = Field(None, description="答案相关性（0-1）")
    context_precision: Optional[float] = Field(None, description="上下文精确率（0-1）")
    context_recall: Optional[float] = Field(None, description="上下文召回率（0-1，需 ground_truth）")
    overall_score: float = Field(..., description="综合得分（有效指标的加权均值）")
    evaluated_at: str = Field(..., description="评测时间")
    saved_path: Optional[str] = Field(None, description="结果保存路径")


# ── 评测器 ────────────────────────────────────────────────────────────────────

class RagasEvaluator:
    """
    Ragas 自动化评测器

    使用 Ragas 框架对 RAG 生成结果进行多维度打分。
    评测本身也需要调用 LLM（作为"裁判"），会产生少量 API 费用。
    """

    def __init__(self):
        """初始化评测所用的 LLM 和 Embedding 模型"""
        # Ragas 需要 LangChain 格式的 LLM 和 Embeddings
        self._llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
            temperature=0,
        )
        self._embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=GLM_API_KEY,
            openai_api_base=GLM_BASE_URL,
        )

    def evaluate(self, request: EvalRequest, save_result: bool = True) -> EvalResult:
        """
        执行 Ragas 评测

        Args:
            request:     评测请求（问题、答案、上下文、参考答案）
            save_result: 是否将结果保存为 JSON 文件

        Returns:
            EvalResult 各维度得分及综合分
        """
        print(f"[Ragas 评测] 开始评测：{request.question[:40]}...")

        # 构建 Ragas 所需的 HuggingFace Dataset 格式
        data = {
            "question":  [request.question],
            "answer":    [request.answer],
            "contexts":  [request.contexts],
        }
        if request.ground_truth:
            data["ground_truth"] = [request.ground_truth]

        dataset = Dataset.from_dict(data)

        # 选择评测指标（有 ground_truth 才能算 context_recall）
        metrics = [faithfulness, answer_relevancy, context_precision]
        if request.ground_truth:
            metrics.append(context_recall)

        # 执行评测（Ragas 内部会多次调用 LLM 打分）
        scores_df = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self._llm,
            embeddings=self._embeddings,
        ).to_pandas()

        # 提取各维度分数
        def _get(col: str) -> Optional[float]:
            if col in scores_df.columns:
                val = scores_df[col].iloc[0]
                return round(float(val), 4) if val == val else None  # NaN 检查
            return None

        faith    = _get("faithfulness")
        relevancy = _get("answer_relevancy")
        precision = _get("context_precision")
        recall    = _get("context_recall")

        # 综合得分：有效指标的均值
        valid_scores = [s for s in [faith, relevancy, precision, recall] if s is not None]
        overall = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0

        evaluated_at = datetime.now().isoformat()
        saved_path = None

        result = EvalResult(
            question=request.question,
            faithfulness=faith,
            answer_relevancy=relevancy,
            context_precision=precision,
            context_recall=recall,
            overall_score=overall,
            evaluated_at=evaluated_at,
            saved_path=saved_path,
        )

        # 打印摘要
        self._print_summary(result)

        # 保存结果
        if save_result:
            saved_path = self._save(result)
            result = result.model_copy(update={"saved_path": saved_path})

        return result

    def batch_evaluate(self, requests: list[EvalRequest]) -> list[EvalResult]:
        """
        批量评测多条问答

        Args:
            requests: 评测请求列表

        Returns:
            EvalResult 列表
        """
        results = []
        for i, req in enumerate(requests, 1):
            print(f"\n[Ragas 评测] 第 {i}/{len(requests)} 条...")
            result = self.evaluate(req, save_result=False)
            results.append(result)

        # 批量保存为单个 JSON 文件
        self._save_batch(results)

        # 打印批量统计
        overall_avg = sum(r.overall_score for r in results) / len(results)
        print(f"\n[Ragas 评测] 批量评测完成，平均综合得分：{overall_avg:.4f}")

        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _print_summary(self, result: EvalResult) -> None:
        """打印评测结果摘要"""
        print(f"\n{'─' * 40}")
        print(f"[Ragas 评测结果]")
        print(f"  忠实度       (Faithfulness)    : {result.faithfulness}")
        print(f"  答案相关性   (Answer Relevancy): {result.answer_relevancy}")
        print(f"  上下文精确率 (Context Precision): {result.context_precision}")
        print(f"  上下文召回率 (Context Recall)  : {result.context_recall}")
        print(f"  综合得分                        : {result.overall_score}")
        print(f"{'─' * 40}")

    def _save(self, result: EvalResult) -> str:
        """保存单条评测结果为 JSON"""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"eval_{timestamp}.json"
        path.write_text(
            result.model_dump_json(indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Ragas 评测] 结果已保存：{path}")
        return str(path)

    def _save_batch(self, results: list[EvalResult]) -> str:
        """保存批量评测结果为 JSON"""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"eval_batch_{timestamp}.json"
        data = [r.model_dump() for r in results]
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Ragas 评测] 批量结果已保存：{path}")
        return str(path)


# ── 便捷函数：评测单次研报输出 ────────────────────────────────────────────────

def evaluate_report(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> EvalResult:
    """
    便捷函数：对单次研报输出执行 Ragas 评测

    Args:
        question:     用户原始问题
        answer:       生成的研报全文
        contexts:     RAG 检索到的上下文片段（字符串列表）
        ground_truth: 参考答案（可选，有值才能计算 context_recall）

    Returns:
        EvalResult 评测得分
    """
    evaluator = RagasEvaluator()
    return evaluator.evaluate(
        EvalRequest(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
    )
