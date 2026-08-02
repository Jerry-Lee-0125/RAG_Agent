"""V6 的 Chroma＋BM25 混合檢索。

Chroma 負責找語意相近的內容，BM25 負責找明確關鍵字，RRF 再依兩邊
的排名合併結果。只有真正執行混合檢索時才載入 EnsembleRetriever，
因此離線建立 BM25 索引時不需要先啟動完整的 LangChain 環境。
"""

from __future__ import annotations

import hashlib
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

from bm25_index_v6 import BM25Index


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BM25_PATH = Path(
    os.getenv(
        "PATHOLOGY_BM25_PATH",
        str(PROJECT_ROOT / "02_db" / "bm25_db" / "index.json"),
    )
)

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3
RRF_C = 60


class HybridRetrieverUnavailable(RuntimeError):
    """BM25 索引或 EnsembleRetriever 不可用時使用的明確例外。"""


def _normalise_path(value: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(value))))


def _relative_source(source: str | Path, processed_data_path: str | Path) -> str:
    try:
        return Path(source).expanduser().resolve().relative_to(
            Path(processed_data_path).expanduser().resolve()
        ).as_posix()
    except ValueError:
        return Path(str(source)).as_posix()


def _stable_hybrid_id(
    metadata: Mapping[str, Any],
    page_content: str,
    processed_data_path: str | Path,
) -> str:
    for key in ("hybrid_id", "chunk_id", "document_id", "id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    source = str(metadata.get("source_relative") or metadata.get("source") or "")
    page = str(metadata.get("page") if metadata.get("page") is not None else 0)
    digest = hashlib.sha1(
        f"{source}|{page}|{page_content}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{source}|page={page}|hash={digest}"


def _metadata_for_document(
    metadata: Mapping[str, Any],
    page_content: str,
    processed_data_path: str | Path,
) -> dict[str, Any]:
    result = dict(metadata or {})
    source = str(result.get("source") or "")
    source_relative = str(result.get("source_relative") or "").strip()
    if not source_relative and source:
        source_relative = _relative_source(source, processed_data_path)
    if source_relative:
        result["source_relative"] = source_relative
        if not source or not os.path.isabs(source):
            result["source"] = str(
                (Path(processed_data_path).expanduser().resolve() / source_relative).resolve()
            )
    hybrid_id = _stable_hybrid_id(result, page_content, processed_data_path)
    result["hybrid_id"] = hybrid_id
    result.setdefault("chunk_id", hybrid_id)
    return result


def _source_matches(
    metadata: Mapping[str, Any],
    source_filter: str | None,
    source_relative_filter: str | None,
    processed_data_path: str | Path,
) -> bool:
    if not source_filter and not source_relative_filter:
        return True
    candidate_relative = str(
        metadata.get("source_relative") or metadata.get("source") or ""
    )
    candidate_source = str(metadata.get("source") or candidate_relative)
    if source_relative_filter and candidate_relative.replace("\\", "/").casefold() == source_relative_filter.replace("\\", "/").casefold():
        return True
    if source_filter:
        if _normalise_path(candidate_source) == _normalise_path(source_filter):
            return True
        if not os.path.isabs(candidate_source):
            candidate_absolute = Path(processed_data_path).expanduser().resolve() / candidate_source
            if _normalise_path(candidate_absolute) == _normalise_path(source_filter):
                return True
    return False


def _load_langchain_classes():
    try:
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
        from pydantic import ConfigDict
    except ImportError as error:  # pragma: no cover - depends on deployment environment
        raise HybridRetrieverUnavailable(
            "缺少 langchain-classic 或 LangChain Core，無法建立 EnsembleRetriever。"
        ) from error
    return EnsembleRetriever, Document, BaseRetriever, ConfigDict


def execute_hybrid_search(
    *,
    query: str,
    vectordb: Any,
    bm25_index: BM25Index,
    final_k: int,
    candidate_k: int,
    score_threshold: float,
    processed_data_path: str | Path,
    source_filter: str | None = None,
    source_relative_filter: str | None = None,
    trace_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """使用官方 EnsembleRetriever 執行加權 RRF，並回傳可觀測紀錄。"""

    if final_k <= 0 or candidate_k <= 0:
        return [], [], {
            "retriever": "ensemble_rrf",
            "weights": [VECTOR_WEIGHT, BM25_WEIGHT],
            "rrf_c": RRF_C,
            "candidate_k": candidate_k,
            "candidate_count": 0,
            "passed_count": 0,
        }

    EnsembleRetriever, Document, BaseRetriever, ConfigDict = _load_langchain_classes()
    collector: dict[str, Any] = {
        "vector": {"raw_candidate_count": 0, "results": []},
        "bm25": {"raw_candidate_count": 0, "results": []},
    }

    class VectorBranchRetriever(BaseRetriever):
        vectordb: Any
        k: int
        score_threshold: float
        source_filter: str | None = None
        collector: Any
        processed_data_path: str

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            del run_manager
            search_kwargs: dict[str, Any] = {"k": int(self.k)}
            if self.source_filter:
                search_kwargs["filter"] = {"source": self.source_filter}
            raw_results = self.vectordb.similarity_search_with_relevance_scores(
                query,
                **search_kwargs,
            )
            branch = self.collector["vector"]
            branch["raw_candidate_count"] = len(raw_results)
            documents = []
            for raw_rank, pair in enumerate(raw_results, start=1):
                doc, score = pair
                score = float(score)
                if score < float(self.score_threshold):
                    continue
                metadata = _metadata_for_document(
                    getattr(doc, "metadata", {}) or {},
                    getattr(doc, "page_content", ""),
                    self.processed_data_path,
                )
                hybrid_id = metadata["hybrid_id"]
                output_doc = Document(
                    page_content=getattr(doc, "page_content", ""),
                    metadata=metadata,
                )
                documents.append(output_doc)
                branch["results"].append(
                    {
                        "hybrid_id": hybrid_id,
                        "rank": len(branch["results"]) + 1,
                        "raw_rank": raw_rank,
                        "score": score,
                        "document": output_doc,
                    }
                )
            return documents

    class BM25BranchRetriever(BaseRetriever):
        bm25_index: Any
        k: int
        source_filter: str | None = None
        source_relative_filter: str | None = None
        collector: Any
        processed_data_path: str

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def _get_relevant_documents(self, query: str, *, run_manager=None):
            del run_manager
            raw_results = self.bm25_index.search(query, top_k=int(self.k))
            branch = self.collector["bm25"]
            branch["raw_candidate_count"] = len(raw_results)
            documents = []
            for raw_result in raw_results:
                metadata = _metadata_for_document(
                    getattr(raw_result, "metadata", {}) or {},
                    getattr(raw_result, "text", ""),
                    self.processed_data_path,
                )
                if not _source_matches(
                    metadata,
                    self.source_filter,
                    self.source_relative_filter,
                    self.processed_data_path,
                ):
                    continue
                hybrid_id = str(
                    getattr(raw_result, "document_id", "")
                    or metadata["hybrid_id"]
                )
                metadata["hybrid_id"] = hybrid_id
                metadata.setdefault("chunk_id", hybrid_id)
                output_doc = Document(
                    page_content=getattr(raw_result, "text", ""),
                    metadata=metadata,
                )
                documents.append(output_doc)
                branch["results"].append(
                    {
                        "hybrid_id": hybrid_id,
                        "rank": len(branch["results"]) + 1,
                        "raw_rank": len(branch["results"]) + 1,
                        "score": float(getattr(raw_result, "score", 0.0)),
                        "document": output_doc,
                    }
                )
            return documents

    vector_retriever = VectorBranchRetriever(
        vectordb=vectordb,
        k=int(candidate_k),
        score_threshold=float(score_threshold),
        source_filter=source_filter,
        collector=collector,
        processed_data_path=str(Path(processed_data_path).expanduser().resolve()),
    )
    bm25_retriever = BM25BranchRetriever(
        bm25_index=bm25_index,
        k=int(candidate_k),
        source_filter=source_filter,
        source_relative_filter=source_relative_filter,
        collector=collector,
        processed_data_path=str(Path(processed_data_path).expanduser().resolve()),
    )
    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[VECTOR_WEIGHT, BM25_WEIGHT],
        c=RRF_C,
        id_key="hybrid_id",
    )
    ensemble_documents = list(ensemble.invoke(query))[: int(final_k)]

    branch_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for branch_name, branch_data in collector.items():
        branch_maps[branch_name] = {
            item["hybrid_id"]: item
            for item in branch_data["results"]
        }

    records: list[dict[str, Any]] = []
    output_documents: list[Any] = []
    weights = {"vector": VECTOR_WEIGHT, "bm25": BM25_WEIGHT}
    for final_rank, document in enumerate(ensemble_documents, start=1):
        metadata = _metadata_for_document(
            getattr(document, "metadata", {}) or {},
            getattr(document, "page_content", ""),
            processed_data_path,
        )
        hybrid_id = metadata["hybrid_id"]
        branch_rank: dict[str, int] = {}
        branch_scores: dict[str, float] = {}
        rrf_score = 0.0
        for branch_name, weight in weights.items():
            item = branch_maps[branch_name].get(hybrid_id)
            if item is None:
                continue
            rank = int(item["rank"])
            branch_rank[branch_name] = rank
            branch_scores[branch_name] = float(item["score"])
            rrf_score += float(weight) / (RRF_C + rank - 1)

        metadata.update(
            {
                "retriever": "hybrid",
                "rrf_score": rrf_score,
                "branch_rank": dict(branch_rank),
                "vector_score": branch_scores.get("vector"),
                "bm25_score": branch_scores.get("bm25"),
            }
        )
        output_doc = Document(
            page_content=getattr(document, "page_content", ""),
            metadata=metadata,
        )
        output_documents.append(output_doc)
        page_index = metadata.get("page")
        source_suffix = Path(str(metadata.get("source") or "")).suffix.casefold()
        page_number = (
            page_index + 1
            if source_suffix == ".pdf"
            and not isinstance(page_index, bool)
            and isinstance(page_index, int)
            else None
        )
        records.append(
            {
                "search_scope": "",
                "match_type": "hybrid",
                "retriever": "ensemble_rrf",
                "rank": final_rank,
                "branch_rank": dict(branch_rank),
                "relevance_score": branch_scores.get("vector"),
                "vector_score": branch_scores.get("vector"),
                "bm25_score": branch_scores.get("bm25"),
                "rrf_score": rrf_score,
                "score_threshold": float(score_threshold),
                "configured_top_k": int(final_k),
                "effective_k": int(candidate_k),
                "hybrid_id": hybrid_id,
                "chunk_id": str(metadata.get("chunk_id") or hybrid_id),
                "source_name": os.path.basename(str(metadata.get("source") or "未知文件")),
                "page_number": page_number,
                "content": getattr(document, "page_content", ""),
                "content_length": len(getattr(document, "page_content", "") or ""),
            }
        )

    run_record = {
        "retriever": "ensemble_rrf",
        "weights": [VECTOR_WEIGHT, BM25_WEIGHT],
        "rrf_c": RRF_C,
        "candidate_k": int(candidate_k),
        "candidate_count": sum(
            len(branch_data["results"]) for branch_data in collector.values()
        ),
        "passed_count": len(records),
        "branch_candidate_counts": {
            branch_name: len(branch_data["results"])
            for branch_name, branch_data in collector.items()
        },
        "branch_raw_candidate_counts": {
            branch_name: branch_data["raw_candidate_count"]
            for branch_name, branch_data in collector.items()
        },
        "source_filter": source_filter or "",
        "source_relative_filter": source_relative_filter or "",
        "fallback": False,
    }
    for record in records:
        record["search_scope"] = "hybrid"
    if trace_callback:
        trace_callback({"run": run_record, "records": records})
    return output_documents, records, run_record


def load_bm25_index_if_fresh(
    *,
    index_path: str | Path | None = None,
    processed_data_path: str | Path,
) -> BM25Index:
    """載入 BM25，並拒絕沒有同步指紋的舊索引。"""

    target = Path(index_path or DEFAULT_BM25_PATH).expanduser().resolve()
    if not target.exists():
        raise HybridRetrieverUnavailable(f"找不到 BM25 索引：{target}")
    try:
        from build_bm25_index_v6 import processed_data_fingerprint

        processed_root = Path(processed_data_path).expanduser().resolve()
        expected = processed_data_fingerprint(processed_root)
        return _load_bm25_index_cached(
            str(target),
            str(processed_root),
            expected,
        )
    except HybridRetrieverUnavailable:
        raise
    except Exception as error:
        raise HybridRetrieverUnavailable(f"BM25 索引無法載入：{error}") from error


@lru_cache(maxsize=4)
def _load_bm25_index_cached(
    index_path: str,
    processed_data_path: str,
    expected_fingerprint: str,
) -> BM25Index:
    del processed_data_path
    index = BM25Index.load(index_path)
    metadata = getattr(index, "index_metadata", {}) or {}
    if not metadata.get("fingerprint"):
        raise HybridRetrieverUnavailable(
            "BM25 索引缺少資料同步指紋，請先重建索引。"
        )
    if metadata.get("fingerprint") != expected_fingerprint:
        raise HybridRetrieverUnavailable(
            "BM25 索引與目前 processed data 不一致，請先同步重建。"
        )
    return index


def get_bm25_index_status(
    *,
    index_path: str | Path | None = None,
    processed_data_path: str | Path,
) -> dict[str, Any]:
    """提供管理介面使用的 BM25 狀態，不讓檢查例外阻擋問答。"""

    target = Path(index_path or DEFAULT_BM25_PATH).expanduser().resolve()
    try:
        index = load_bm25_index_if_fresh(
            index_path=target,
            processed_data_path=processed_data_path,
        )
    except HybridRetrieverUnavailable as error:
        return {
            "available": False,
            "path": str(target),
            "reason": str(error),
            "document_count": 0,
        }
    return {
        "available": True,
        "path": str(target),
        "reason": "",
        "document_count": len(index.documents),
        "tokenizer_type": index.tokenizer.tokenizer_type,
    }


def rebuild_bm25_index(
    *,
    processed_data_path: str | Path,
    output_path: str | Path | None = None,
    dictionary_manifest_output: str | Path | None = None,
) -> dict[str, Any]:
    """以暫存檔重建 BM25，完成後再原子替換正式索引。"""

    from build_bm25_index_v6 import build_index

    destination = Path(output_path or DEFAULT_BM25_PATH).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = destination.with_name(
        f"{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    manifest_destination = Path(
        dictionary_manifest_output
        or PROJECT_ROOT / "BM25" / "dictionary_manifest_v6.json"
    ).expanduser().resolve()
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_destination.with_name(
        f"{manifest_destination.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        summary = build_index(
            processed_data=processed_data_path,
            output=temporary_index,
            dictionary_manifest_output=temporary_manifest,
        )
        os.replace(temporary_index, destination)
        os.replace(temporary_manifest, manifest_destination)
        _load_bm25_index_cached.cache_clear()
        summary["output"] = str(destination)
        summary["dictionary_manifest_output"] = str(manifest_destination)
        return summary
    finally:
        for temporary_path in (temporary_index, temporary_manifest):
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
