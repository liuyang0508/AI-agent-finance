"""
PDF 解析模块

使用 Marker 将财报 PDF 转换为 Markdown 格式，并对跨页表格进行合并修复。
输出结果保存到 data/markdown/ 目录，供 RAG 模块使用。
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

from config.settings import PDF_DIR, MARKDOWN_DIR


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class ParseRequest(BaseModel):
    """PDF 解析请求参数"""
    pdf_path: str = Field(..., description="PDF 文件路径，可以是绝对路径或相对于 data/pdfs/ 的文件名")
    save_output: bool = Field(True, description="是否将解析结果保存到 data/markdown/ 目录")
    fix_cross_page_tables: bool = Field(True, description="是否启用跨页表格合并修复")


class ParseResult(BaseModel):
    """PDF 解析结果"""
    source_pdf: str = Field(..., description="源 PDF 文件路径")
    markdown_text: str = Field(..., description="转换后的完整 Markdown 文本")
    output_path: Optional[str] = Field(None, description="保存的 Markdown 文件路径（save_output=True 时有值）")
    page_count: int = Field(..., description="PDF 总页数")
    table_count: int = Field(..., description="识别到的表格数量")
    merged_table_count: int = Field(0, description="合并修复的跨页表格数量")


# ── 核心解析器 ────────────────────────────────────────────────────────────────

class FinancialPDFParser:
    """
    财报 PDF 解析器

    封装 Marker 库，专为财务报告场景优化：
    - 保留表格的 Markdown 结构（| 列 | 列 |）
    - 自动检测并合并被分页截断的跨页表格
    - 清理页眉页脚噪声
    """

    def __init__(self):
        """初始化 Marker 模型（首次运行会下载模型权重）"""
        print("正在加载 Marker 模型，首次运行需要下载模型权重，请稍候...")
        self._model_dict = create_model_dict()
        self._converter = PdfConverter(artifact_dict=self._model_dict)
        print("Marker 模型加载完成。")

    def parse(self, request: ParseRequest) -> ParseResult:
        """
        执行 PDF 解析

        Args:
            request: 解析请求参数

        Returns:
            ParseResult 解析结果对象
        """
        # 解析文件路径
        pdf_path = self._resolve_path(request.pdf_path)

        # 调用 Marker 转换
        print(f"开始解析：{pdf_path.name}")
        rendered = self._converter(str(pdf_path))
        markdown_text, _, metadata = text_from_rendered(rendered)

        # 获取页数
        page_count = metadata.get("page_count", 0)

        # 统计原始表格数量
        table_count = self._count_tables(markdown_text)
        merged_count = 0

        # 跨页表格合并
        if request.fix_cross_page_tables:
            markdown_text, merged_count = self._fix_cross_page_tables(markdown_text)

        # 清理噪声（页眉页脚等）
        markdown_text = self._clean_noise(markdown_text)

        # 保存结果
        output_path = None
        if request.save_output:
            output_path = self._save_markdown(pdf_path, markdown_text)

        print(f"解析完成：共 {page_count} 页，{table_count} 个表格，合并跨页表格 {merged_count} 处")

        return ParseResult(
            source_pdf=str(pdf_path),
            markdown_text=markdown_text,
            output_path=output_path,
            page_count=page_count,
            table_count=table_count,
            merged_table_count=merged_count,
        )

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _resolve_path(self, pdf_path: str) -> Path:
        """解析 PDF 文件路径：优先绝对路径，其次在 data/pdfs/ 中查找"""
        path = Path(pdf_path)
        if path.is_absolute() and path.exists():
            return path
        # 在默认目录中查找
        fallback = PDF_DIR / pdf_path
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}（也尝试了 {fallback}）")

    def _count_tables(self, text: str) -> int:
        """统计 Markdown 文本中的表格数量（以表头分隔行 |---|---| 计数）"""
        return len(re.findall(r'^\|[-| :]+\|$', text, flags=re.MULTILINE))

    def _fix_cross_page_tables(self, text: str) -> tuple[str, int]:
        """
        合并跨页表格

        财报中表格经常跨页，Marker 会在分页处断开表格并各自加表头。
        本方法识别"相邻两个列数相同的表格，且后一个表格的表头与前一个相同"
        的模式，将第二个表格的表头行删除，实现无缝合并。

        Returns:
            (修复后的文本, 合并次数)
        """
        # 匹配完整表格块：表头行 + 分隔行 + 若干数据行
        table_pattern = re.compile(
            r'(?P<header>\|.+\|)\n'       # 表头行
            r'(?P<separator>\|[-| :]+\|)\n'  # 分隔行
            r'(?P<rows>(?:\|.+\|\n?)+)',   # 数据行
            re.MULTILINE
        )

        tables = list(table_pattern.finditer(text))
        if len(tables) < 2:
            return text, 0

        merged_count = 0
        # 从后往前处理，避免 offset 错乱
        for i in range(len(tables) - 1, 0, -1):
            prev = tables[i - 1]
            curr = tables[i]

            prev_header = prev.group("header").strip()
            curr_header = curr.group("header").strip()

            # 判断两个表格列数是否相同且表头一致
            if prev_header != curr_header:
                continue

            # 检查两个表格在文本中是否紧邻（中间只允许有空行或分页标记）
            between = text[prev.end():curr.start()]
            if not re.fullmatch(r'[\n\r\s]*', between):
                continue

            # 合并：去掉后一个表格的表头行和分隔行
            merged_rows = curr.group("rows")
            replacement = text[curr.start():curr.end()]
            new_block = merged_rows  # 仅保留数据行
            text = text[:curr.start()] + new_block + text[curr.end():]
            merged_count += 1

        return text, merged_count

    def _clean_noise(self, text: str) -> str:
        """
        清理页眉页脚等噪声

        Marker 有时会把重复的公司名称、页码等识别为正文，此处做简单清理：
        - 去除连续的纯数字行（页码）
        - 去除三行以内的重复短行
        """
        lines = text.split('\n')
        cleaned = []
        seen_short: dict[str, int] = {}

        for line in lines:
            stripped = line.strip()

            # 跳过纯数字行（页码）
            if re.fullmatch(r'\d{1,4}', stripped):
                continue

            # 对短行（≤20字符）去重：同一内容出现超过 3 次则视为页眉页脚
            if 0 < len(stripped) <= 20:
                seen_short[stripped] = seen_short.get(stripped, 0) + 1
                if seen_short[stripped] > 3:
                    continue

            cleaned.append(line)

        return '\n'.join(cleaned)

    def _save_markdown(self, pdf_path: Path, markdown_text: str) -> str:
        """将 Markdown 文本保存到 data/markdown/ 目录"""
        MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
        output_path = MARKDOWN_DIR / (pdf_path.stem + ".md")
        output_path.write_text(markdown_text, encoding="utf-8")
        print(f"Markdown 已保存至：{output_path}")
        return str(output_path)


# ── 批量解析工具函数 ──────────────────────────────────────────────────────────

def parse_all_pdfs(fix_cross_page_tables: bool = True) -> list[ParseResult]:
    """
    批量解析 data/pdfs/ 目录下的所有 PDF 文件

    Args:
        fix_cross_page_tables: 是否启用跨页表格合并

    Returns:
        ParseResult 列表
    """
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"未在 {PDF_DIR} 中找到 PDF 文件，请将财报 PDF 放入该目录后重试。")
        return []

    parser = FinancialPDFParser()
    results = []

    for pdf_file in pdf_files:
        request = ParseRequest(
            pdf_path=str(pdf_file),
            save_output=True,
            fix_cross_page_tables=fix_cross_page_tables,
        )
        result = parser.parse(request)
        results.append(result)

    print(f"\n批量解析完成，共处理 {len(results)} 个文件。")
    return results


# ── 快捷入口（直接运行此文件时使用）────────────────────────────────────────────

if __name__ == "__main__":
    results = parse_all_pdfs()
    for r in results:
        print(json.dumps({
            "source_pdf": r.source_pdf,
            "page_count": r.page_count,
            "table_count": r.table_count,
            "merged_table_count": r.merged_table_count,
            "output_path": r.output_path,
        }, ensure_ascii=False, indent=2))
