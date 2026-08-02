r"""建立 V6 BM25 關鍵字索引的命令列工具。

範例（Windows PowerShell）：

    python build_bm25_index_v6.py `
      --processed-data ..\01_processed_data `
      --output 02_db\bm25_db\index.json

工具會讀取 PDF、DOCX、CSV，切成每塊最多 600 字、前後重疊 150 字。
Chroma 與 BM25 共用這些文字塊。建立索引時會加入病理學、醫事檢驗及
院內自訂詞彙，但不會修改原始文件或 ODS 詞庫。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from bm25_index_v6 import (
    BM25Index,
    JiebaTokenizer,
    build_term_dictionary,
    write_dictionary_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROCESSED_DATA = Path(
    os.getenv("PATHOLOGY_PROCESSED_DATA_PATH", str(PROJECT_ROOT / "01_processed_data"))
)
DEFAULT_OUTPUT = Path(
    os.getenv("PATHOLOGY_BM25_PATH", str(PROJECT_ROOT / "02_db" / "bm25_db" / "index.json"))
)
PATHOLOGY_ODS = PROJECT_ROOT / "BM25" / "病理學名詞壓縮檔_0.ods"
LABORATORY_ODS = PROJECT_ROOT / "BM25" / "醫學名詞-醫事檢驗名詞壓縮檔_0.ods"
HOSPITAL_TEMPLATE = PROJECT_ROOT / "BM25" / "hospital_terms_template.csv"


def _clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n+", "\n\n", text or "")
    return "\n".join(line.lstrip() for line in text.splitlines()).strip()


def _chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 150) -> list[str]:
    """依段落、換行、句號、分號及空白順序切割文字。"""

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size 必須大於 0，chunk_overlap 必須介於 0 與 chunk_size 之間")
    clean = _clean_text(text)
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0
    separators = ("\n\n", "\n", "。", "；", ";", " ")
    while start < len(clean):
        proposed_end = min(start + chunk_size, len(clean))
        end = proposed_end
        if proposed_end < len(clean):
            lower_bound = start + max(chunk_size // 2, 1)
            for separator in separators:
                split_at = clean.rfind(separator, lower_bound, proposed_end)
                if split_at > start:
                    end = split_at + len(separator)
                    break
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        next_start = max(end - chunk_overlap, start + 1)
        start = next_start
    return chunks


def _read_docx_text(path: Path) -> str:
    """以 OOXML 讀取 DOCX，避免 BM25 建置器依賴 docx2txt。"""

    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ET.fromstring(document_xml)
    paragraphs: list[str] = []
    for paragraph in root.findall(f".//{{{namespace}}}p"):
        parts = [node.text or "" for node in paragraph.findall(f".//{{{namespace}}}t")]
        if parts:
            paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


def _read_pdf_pages(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - deployment dependency check
        raise RuntimeError("建立 PDF BM25 索引需要 pypdf") from error
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _read_csv_text(path: Path) -> str:
    rows: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            values = [str(value).strip() for value in row if str(value).strip()]
            if values:
                rows.append("\t".join(values))
    return "\n".join(rows)


def _document_parts(path: Path) -> list[tuple[int | None, str]]:
    extension = path.suffix.casefold()
    if extension == ".pdf":
        return list(enumerate(_read_pdf_pages(path)))
    if extension == ".docx":
        # DOCX 的 OOXML 本身沒有可靠的 PDF 式頁碼；頁面會隨 Word
        # 版本、字型、紙張與版面設定重新流動，因此不要假造第 0 頁。
        return [(None, _read_docx_text(path))]
    if extension == ".csv":
        return [(None, _read_csv_text(path))]
    return []


def processed_data_fingerprint(processed_data: str | Path) -> str:
    """以支援的實體檔案清單、大小與修改時間建立快速指紋。"""

    root = Path(processed_data).expanduser().resolve()
    entries: list[str] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in {".pdf", ".docx", ".csv"}:
                continue
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            entries.append(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def load_processed_documents(
    processed_data: str | Path,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
    paths: Iterable[str | Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """讀取處理後資料並切成可供 BM25 使用的文件片段。"""

    root = Path(processed_data).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"找不到 processed data：{root}")

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    file_count = 0
    if paths is None:
        candidate_paths = sorted(root.rglob("*"))
    else:
        candidate_paths = []
        for raw_path in paths:
            candidate = Path(raw_path).expanduser().resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            candidate_paths.append(candidate)

    for path in candidate_paths:
        if not path.is_file() or path.suffix.casefold() not in {".pdf", ".docx", ".csv"}:
            continue
        file_count += 1
        relative_source = path.relative_to(root).as_posix()
        try:
            parts = _document_parts(path)
            for page_number, page_text in parts:
                for chunk_number, chunk in enumerate(
                    _chunk_text(page_text, chunk_size, chunk_overlap)
                ):
                    # page=None 代表非 PDF 文件沒有可供顯示的頁碼；
                    # ID 使用 NA，避免把 Python 的 None 寫進穩定 ID。
                    page_key = page_number if page_number is not None else "NA"
                    document_id = (
                        f"{relative_source}::page={page_key}::chunk={chunk_number}"
                    )
                    documents.append(
                        {
                            "document_id": document_id,
                            "text": chunk,
                            "metadata": {
                                "source": relative_source,
                                "source_relative": relative_source,
                                "source_name": path.name,
                                "source_type": path.suffix.casefold().lstrip("."),
                                "page": page_number,
                                "chunk_id": document_id,
                                "hybrid_id": document_id,
                            },
                        }
                    )
        except Exception as error:  # keep other files indexable
            errors.append({"source": relative_source, "error": str(error)})

    return documents, {
        "processed_data": str(root),
        "file_count": file_count,
        "document_count": len(documents),
        "error_count": len(errors),
        "errors": errors,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "fingerprint": processed_data_fingerprint(root),
    }


def build_index(
    *,
    processed_data: str | Path,
    pathology_ods: str | Path = PATHOLOGY_ODS,
    medical_laboratory_ods: str | Path = LABORATORY_ODS,
    hospital_csv: str | Path | None = HOSPITAL_TEMPLATE,
    output: str | Path = DEFAULT_OUTPUT,
    dictionary_manifest_output: str | Path | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> dict[str, Any]:
    dictionary = build_term_dictionary(
        pathology_ods,
        medical_laboratory_ods,
        hospital_csv if hospital_csv and Path(hospital_csv).exists() else None,
    )
    documents, document_stats = load_processed_documents(
        processed_data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    index = BM25Index(JiebaTokenizer(dictionary["terms"], hmm=False))
    index.add_documents(documents)
    index.save(
        output,
        dictionary_manifest=dictionary,
        index_metadata={
            "processed_data": str(Path(processed_data).expanduser().resolve()),
            "fingerprint": document_stats["fingerprint"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "document_count": document_stats["document_count"],
        },
    )

    if dictionary_manifest_output:
        write_dictionary_manifest(dictionary, dictionary_manifest_output)

    return {
        "output": str(Path(output).resolve()),
        "index": index.stats(),
        "dictionary": dictionary["stats"],
        "documents": document_stats,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建立 V6 病理 BM25 索引")
    parser.add_argument("--processed-data", default=str(DEFAULT_PROCESSED_DATA))
    parser.add_argument("--pathology-ods", default=str(PATHOLOGY_ODS))
    parser.add_argument("--medical-laboratory-ods", default=str(LABORATORY_ODS))
    parser.add_argument("--hospital-csv", default=str(HOSPITAL_TEMPLATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--dictionary-manifest-output",
        default=str(PROJECT_ROOT / "BM25" / "dictionary_manifest_v6.json"),
    )
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        summary = build_index(
            processed_data=args.processed_data,
            pathology_ods=args.pathology_ods,
            medical_laboratory_ods=args.medical_laboratory_ods,
            hospital_csv=args.hospital_csv,
            output=args.output,
            dictionary_manifest_output=args.dictionary_manifest_output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as error:
        print(f"BM25 索引建立失敗：{error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
