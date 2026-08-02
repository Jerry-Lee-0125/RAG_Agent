"""Phoenix 本機追蹤工具。

本模組把 Phoenix / OpenTelemetry 相關細節集中在一處，讓主程式與
RAG 工具只需要呼叫簡單函式：

1. ``get_phoenix_status``：取得目前是否能連線到 Phoenix。
2. ``start_question_trace``：開始記錄一輪完整問答。
3. ``finish_question_trace``：保存回答、工具與檢索結果後結束問答。
4. ``traced_operation``：為 Retriever 等局部工作建立子 Span。

設計原則：

- Phoenix 套件沒安裝或服務沒有啟動時，問答系統仍可正常使用。
- 預設只連接 ``127.0.0.1``，不使用 Phoenix Cloud。
- 保存完整測試內容，但不保存密碼、金鑰及絕對路徑。
- 不記錄 Agent 未公開的內部推理，只記錄可觀察的輸入、工具與輸出。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "http://127.0.0.1:6006"
DEFAULT_PROJECT_NAME = "pathology-rag-v6"
RETRY_SECONDS = 10.0

_state_lock = threading.Lock()
_tracer_provider = None
_tracer = None
_registered = False
_last_probe_at = 0.0
_status = {
    "enabled": True,
    "available": False,
    "state": "尚未檢查",
    "detail": "",
    "endpoint": DEFAULT_ENDPOINT,
    "project_name": DEFAULT_PROJECT_NAME,
    "capture_content": True,
}


def _env_flag(name: str, default: bool) -> bool:
    """將環境變數安全轉成布林值。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _safe_endpoint() -> str:
    """取得 Phoenix 網頁基底位址，只回傳不含憑證的 URL。

    ``phoenix.otel.register`` 在明確傳入 HTTP endpoint 時，要求使用完整的
    ``/v1/traces`` 路徑；但狀態探測、管理員畫面與 feedback API 需要的是
    Phoenix 網頁基底位址。這裡統一移除該路徑，避免兩種用途混在一起。
    """
    endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT",
        DEFAULT_ENDPOINT,
    ).rstrip("/")
    trace_path = "/v1/traces"
    if endpoint.endswith(trace_path):
        endpoint = endpoint[: -len(trace_path)]
    return endpoint.rstrip("/")


def _otel_http_endpoint(endpoint: str) -> str:
    """將 Phoenix 網頁基底位址轉成 OTLP/HTTP Trace 接收位址。"""
    return endpoint.rstrip("/") + "/v1/traces"


def _sanitize_error_text(value: Any) -> str:
    """移除錯誤訊息中的 Windows / Linux 絕對路徑。"""
    text = str(value or "")
    text = re.sub(r"[A-Za-z]:\\[^\s\"']+", "<PATH>", text)
    text = re.sub(
        r"(?<![:\w])/(?:[^/\s]+/)+[^/\s]*",
        "<PATH>",
        text,
    )
    return text[:4000]


def _json_text(value: Any) -> str:
    """將結構化資料轉成 Phoenix 可保存的 JSON 字串。"""
    return json.dumps(value, ensure_ascii=False, default=str)


def _probe_phoenix(endpoint: str, timeout: float = 1.5) -> tuple[bool, str]:
    """用輕量 HTTP 請求確認 Phoenix 服務是否可連線。"""
    try:
        request = Request(endpoint + "/", method="GET")
        with urlopen(request, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
        return status_code < 500, f"HTTP {status_code}"
    except HTTPError as exc:
        # 401 / 403 代表服務存在，只是啟用了驗證；SDK 可另用 API Key。
        if exc.code in {401, 403}:
            return True, f"HTTP {exc.code}（需要驗證）"
        return False, f"HTTP {exc.code}"
    except (URLError, OSError, TimeoutError) as exc:
        return False, _sanitize_error_text(exc)


def initialize_phoenix_tracing(force_probe: bool = False) -> dict[str, Any]:
    """初始化 Phoenix tracing；重複呼叫不會重複註冊 provider。

    服務尚未啟動時只更新狀態，不會讓 import 或問答流程失敗。若先前
    無法連線，Streamlit 日後重新執行時每隔十秒會自動重試。
    """
    global _last_probe_at, _registered, _tracer_provider, _tracer, _status

    endpoint = _safe_endpoint()
    project_name = os.getenv(
        "PHOENIX_PROJECT_NAME",
        DEFAULT_PROJECT_NAME,
    ).strip() or DEFAULT_PROJECT_NAME
    enabled = _env_flag("PHOENIX_ENABLED", True)
    capture_content = _env_flag("PHOENIX_CAPTURE_CONTENT", True)

    now = time.monotonic()
    with _state_lock:
        if (
            not force_probe
            and now - _last_probe_at < RETRY_SECONDS
            and _status["endpoint"] == endpoint
        ):
            return dict(_status)

        _last_probe_at = now
        _status.update(
            {
                "enabled": enabled,
                "endpoint": endpoint,
                "project_name": project_name,
                "capture_content": capture_content,
            }
        )

        if not enabled:
            _status.update(
                {
                    "available": False,
                    "state": "已停用",
                    "detail": "PHOENIX_ENABLED=false",
                }
            )
            return dict(_status)

        reachable, probe_detail = _probe_phoenix(endpoint)
        if not reachable:
            _status.update(
                {
                    "available": False,
                    "state": "未連線",
                    "detail": probe_detail,
                }
            )
            return dict(_status)

        if not _registered:
            try:
                from phoenix.otel import register

                _tracer_provider = register(
                    project_name=project_name,
                    # HTTP exporter 必須送到 /v1/traces；單純的 6006 根目錄
                    # 只是 Phoenix 網頁介面，無法接收 OTLP Trace。
                    endpoint=_otel_http_endpoint(endpoint),
                    protocol="http/protobuf",
                    batch=True,
                    auto_instrument=True,
                )
                _tracer = _tracer_provider.get_tracer(__name__)
                _registered = True
            except Exception as exc:  # Phoenix 是可選功能，不能中斷主程式。
                _status.update(
                    {
                        "available": False,
                        "state": "初始化失敗",
                        "detail": _sanitize_error_text(exc),
                    }
                )
                return dict(_status)

        _status.update(
            {
                "available": True,
                "state": "已連線",
                "detail": probe_detail,
            }
        )
        return dict(_status)


def get_phoenix_status(force_probe: bool = False) -> dict[str, Any]:
    """回傳適合顯示在管理員側邊欄的 Phoenix 狀態。"""
    return initialize_phoenix_tracing(force_probe=force_probe)


class _NullSpan:
    """Phoenix 不可用時提供相同介面的空物件。"""

    def set_attribute(self, *_args, **_kwargs):
        return None

    def set_attributes(self, *_args, **_kwargs):
        return None

    def set_input(self, *_args, **_kwargs):
        return None

    def set_output(self, *_args, **_kwargs):
        return None

    def record_exception(self, *_args, **_kwargs):
        return None

    def set_status(self, *_args, **_kwargs):
        return None


def _set_input(span: Any, value: Any) -> None:
    """相容 OpenInferenceSpan 與標準 OpenTelemetry Span。"""
    setter = getattr(span, "set_input", None)
    if callable(setter):
        setter(value)
    else:
        span.set_attribute("input.value", _json_text(value))


def _set_output(span: Any, value: Any) -> None:
    """相容 OpenInferenceSpan 與標準 OpenTelemetry Span。"""
    setter = getattr(span, "set_output", None)
    if callable(setter):
        setter(value)
    else:
        span.set_attribute("output.value", _json_text(value))


def _set_attributes(span: Any, attributes: dict[str, Any] | None) -> None:
    """只寫入 OpenTelemetry 支援的簡單型別，複合值改存 JSON。"""
    for key, value in (attributes or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (str, bool, int, float)) for item in value
        ):
            span.set_attribute(key, list(value))
        else:
            span.set_attribute(key, _json_text(value))


def _set_span_status(span: Any, error: Exception | None = None) -> None:
    """將成功或錯誤狀態寫入 Span；套件缺少時安靜略過。"""
    try:
        from opentelemetry.trace import Status, StatusCode

        if error is None:
            span.set_status(Status(StatusCode.OK))
        else:
            span.record_exception(error)
            span.set_status(
                Status(
                    StatusCode.ERROR,
                    _sanitize_error_text(error),
                )
            )
    except Exception:
        return


@dataclass
class QuestionTraceHandle:
    """保存一輪問答根 Span 與其作用中 Context。"""

    context_manager: Any = None
    span: Any = None
    active: bool = False
    started_at: float = 0.0
    trace_id: str = ""
    span_id: str = ""


def start_question_trace(
    question: str,
    *,
    execution_id: str,
    session_id: str,
    user_role: str,
    model_name: str,
    score_threshold: float,
    top_k: int,
    memory_enabled: bool,
    memory_rounds: int,
    active_document_name: str = "",
) -> QuestionTraceHandle:
    """開始記錄一次完整問答，並讓後續 LangChain Span 成為其子節點。"""
    status = initialize_phoenix_tracing()
    handle = QuestionTraceHandle(started_at=time.perf_counter())
    if not status["available"] or _tracer is None:
        return handle

    try:
        context_manager = _tracer.start_as_current_span(
            "pathology.question",
            attributes={"openinference.span.kind": "CHAIN"},
        )
        span = context_manager.__enter__()
        handle.context_manager = context_manager
        handle.span = span
        handle.active = True
        try:
            span_context = span.get_span_context()
            handle.trace_id = f"{span_context.trace_id:032x}"
            handle.span_id = f"{span_context.span_id:016x}"
        except Exception:
            # ID 只影響後續回饋同步，不影響主要追蹤。
            handle.trace_id = ""
            handle.span_id = ""

        if status["capture_content"]:
            _set_input(span, question)
        else:
            _set_input(span, {"question_length": len(question)})

        _set_attributes(
            span,
            {
                "pathology.app_version": "v6",
                "pathology.execution_id": execution_id,
                "session.id": session_id,
                "pathology.user_role": user_role,
                "llm.model_name": model_name,
                "rag.score_threshold": float(score_threshold),
                "rag.top_k": int(top_k),
                "memory.enabled": bool(memory_enabled),
                "memory.rounds": int(memory_rounds),
                "document.active_name": active_document_name,
                "phoenix.capture_content": status["capture_content"],
            },
        )
    except Exception:
        # 即使 tracing 本身出錯，也不能影響病理科問答。
        if handle.active and handle.context_manager is not None:
            try:
                handle.context_manager.__exit__(None, None, None)
            except Exception:
                pass
        return QuestionTraceHandle(started_at=handle.started_at)

    return handle


def _safe_document_names(candidates: list[Any]) -> list[str]:
    """從候選文件移除 source 絕對路徑，只留下畫面名稱。"""
    safe_names = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            display_name = str(candidate.get("display_name", "")).strip()
            if display_name:
                safe_names.append(display_name)
    return safe_names


def finish_question_trace(
    handle: QuestionTraceHandle,
    *,
    answer: str,
    search_state: dict[str, Any],
    tool_calls: list[dict[str, Any]] | None = None,
    direct_tool_answer: bool = False,
    error: Exception | None = None,
) -> None:
    """完成根 Trace，保存安全的結構化結果並關閉 Context。"""
    if not handle.active or handle.span is None:
        return

    span = handle.span
    try:
        status = get_phoenix_status()
        if status["capture_content"]:
            _set_output(span, answer)
        else:
            _set_output(span, {"answer_length": len(answer or "")})

        retrieval_records = search_state.get("retrieval_records", [])
        resolved_document = search_state.get("resolved_document")
        resolved_name = ""
        if isinstance(resolved_document, dict):
            resolved_name = str(
                resolved_document.get("display_name", "")
            ).strip()

        _set_attributes(
            span,
            {
                "pathology.result_status": "error" if error else "success",
                "pathology.total_latency_ms": round(
                    (time.perf_counter() - handle.started_at) * 1000,
                    2,
                ),
                "agent.tool_calls": tool_calls or [],
                "agent.tool_results": search_state.get("tool_results", []),
                "agent.direct_tool_answer": bool(direct_tool_answer),
                "rag.retrieval_records": retrieval_records,
                "rag.retrieval_runs": search_state.get(
                    "retrieval_runs",
                    [],
                ),
                "rag.hybrid_status": search_state.get(
                    "hybrid_status",
                    {},
                ),
                "rag.returned_chunk_count": len(
                    search_state.get("docs", [])
                    or search_state.get("topic_docs", [])
                ),
                "answer.source_count": len(
                    {
                        record.get("source_name")
                        for record in retrieval_records
                        if (
                            isinstance(record, dict)
                            and record.get("match_type") in {"vector", "hybrid"}
                            and record.get("source_name")
                        )
                    }
                ),
                "answer.refused": any(
                    marker in (answer or "")
                    for marker in (
                        "找不到相關資訊",
                        "找不到與此問題相符",
                        "目前資料不足",
                        "資料庫中沒有任何文件",
                        "目前病理科規範資料中找不到",
                    )
                ),
                "rag.topic_document_names": search_state.get(
                    "topic_document_names",
                    [],
                ),
                "rag.filename_matches": search_state.get(
                    "filename_matches",
                    [],
                ),
                "rag.vector_document_names": search_state.get(
                    "vector_document_names",
                    [],
                ),
                "document.specific_status": search_state.get(
                    "specific_document_status"
                ),
                "document.resolved_name": resolved_name,
                "document.candidate_names": _safe_document_names(
                    search_state.get("document_candidates", [])
                ),
            },
        )
        if error is not None:
            span.set_attribute(
                "error.message",
                _sanitize_error_text(error),
            )
        _set_span_status(span, error=error)
    except Exception:
        # 完成追蹤時的序列化或 exporter 錯誤不可覆蓋原本回答。
        pass
    finally:
        try:
            handle.context_manager.__exit__(
                type(error) if error else None,
                error,
                error.__traceback__ if error else None,
            )
        except Exception:
            pass
        handle.active = False
        # 回饋可能在回答完成後立刻送出，先嘗試把批次 Span 刷到 Phoenix。
        force_flush = getattr(_tracer_provider, "force_flush", None)
        if callable(force_flush):
            try:
                force_flush(timeout_millis=2000)
            except Exception:
                pass


def record_user_feedback(
    span_id: str,
    *,
    positive: bool,
) -> tuple[bool, str]:
    """把 Streamlit 的正負向回饋附加到 Phoenix 根 Span。

    Phoenix annotation API 使用 span ID；API Key 若存在只放在 HTTP Header，
    不會寫進 Trace 或錯誤訊息。
    """
    span_id = str(span_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{16}", span_id):
        return False, "這筆回答沒有可用的 Phoenix Span ID。"

    status = initialize_phoenix_tracing(force_probe=True)
    if not status["available"]:
        return False, "Phoenix 目前未連線，回饋未同步。"

    label = "thumbs-up" if positive else "thumbs-down"
    score = 1 if positive else 0
    payload = {
        "data": [
            {
                "span_id": span_id,
                "name": "user_feedback",
                "annotator_kind": "HUMAN",
                "result": {
                    "label": label,
                    "score": score,
                },
                "metadata": {
                    "app_version": "v6",
                    "source": "streamlit",
                },
            }
        ]
    }
    request_headers = {"Content-Type": "application/json"}
    api_key = os.getenv("PHOENIX_API_KEY", "").strip()
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"

    try:
        request = Request(
            status["endpoint"] + "/v1/span_annotations?sync=true",
            data=_json_text(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        with urlopen(request, timeout=3.0) as response:
            status_code = getattr(response, "status", 200)
        if status_code >= 400:
            return False, f"Phoenix 回傳 HTTP {status_code}。"
        return True, "回饋已同步到 Phoenix。"
    except HTTPError as exc:
        return False, f"Phoenix 回傳 HTTP {exc.code}。"
    except (URLError, OSError, TimeoutError) as exc:
        return False, _sanitize_error_text(exc)


class TracedOperation:
    """包裝子 Span，提供簡單的輸出與屬性方法。"""

    def __init__(self, span: Any):
        self.span = span

    def set_output(self, value: Any) -> None:
        _set_output(self.span, value)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        _set_attributes(self.span, attributes)


@contextmanager
def traced_operation(
    name: str,
    *,
    span_kind: str,
    input_value: Any,
    attributes: dict[str, Any] | None = None,
):
    """建立 Agent 根 Trace 下的自訂子 Span；不可用時回傳空 Span。"""
    status = initialize_phoenix_tracing()
    if not status["available"] or _tracer is None:
        yield TracedOperation(_NullSpan())
        return

    try:
        context_manager = _tracer.start_as_current_span(
            name,
            attributes={
                "openinference.span.kind": str(span_kind).upper(),
            },
        )
        span = context_manager.__enter__()
    except Exception:
        # 建立 Span 失敗屬於追蹤問題，不應阻止原本的檢索工作。
        yield TracedOperation(_NullSpan())
        return

    error = None
    try:
        if status["capture_content"]:
            _set_input(span, input_value)
        else:
            _set_input(span, {"content_recording": False})
        _set_attributes(span, attributes)
    except Exception:
        # 欄位寫入失敗時仍執行原本工作，Span 至少保留名稱與時間。
        pass

    try:
        operation = TracedOperation(span)
        yield operation
        _set_span_status(span)
    except Exception as exc:
        error = exc
        try:
            _set_span_status(span, error=exc)
        except Exception:
            pass
        # yield 內的原始 Retriever / Agent 工作失敗，必須交回主程式處理。
        raise
    finally:
        try:
            context_manager.__exit__(
                type(error) if error else None,
                error,
                error.__traceback__ if error else None,
            )
        except Exception:
            pass
