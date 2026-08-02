"""V6 混合式搜尋使用的 BM25 關鍵字索引。

BM25 依照關鍵字出現情況排列文件；Jieba 負責中文斷詞，自訂詞典則讓
「肺癌」等專業名詞盡量保持完整。詞典包含病理學、醫事檢驗及院內
自訂詞彙。BM25 排名會與 Chroma 向量排名一起交給 RRF 合併。

重要原則：保留否定詞與檢驗結果詞；多字中文查詢至少要命中兩字以上
詞組，避免「火星人」只因「火」字而找到「滅火器」。英文縮寫、數字、
單位與儀器型號仍保留為獨立關鍵字。
"""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, SCHEMA_VERSION}
TOKENIZER_TYPE = "jieba_cut_for_search"
TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
NS = {"table": TABLE_NS, "text": TEXT_NS}

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[a-z0-9]+(?:[._/+:-][a-z0-9]+)*", re.IGNORECASE)


def normalize_text(value: Any) -> str:
    """將詞彙與查詢統一成可重現的比較形式。"""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00a0", " ")
    return " ".join(text.split()).casefold().strip()


def split_aliases(value: Any, *, english: bool = False) -> list[str]:
    """拆分詞庫中的分號別名與英文 ``{=縮寫}``。"""

    text = str(value or "")
    if english:
        text = re.sub(r"\s*\{=([^{}]+)\}", r"; \1", text)
    aliases: list[str] = []
    for part in re.split(r"[;；]", text):
        normalized = normalize_text(part)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return aliases


def _column_value(row: Sequence[str], header_index: Mapping[str, int], name: str) -> str:
    index = header_index.get(name)
    if index is None or index >= len(row):
        return ""
    return str(row[index] or "").strip()


def read_ods_rows(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """以標準函式庫讀取 ODS 的第一張工作表。"""

    ods_path = Path(path)
    with zipfile.ZipFile(ods_path) as archive:
        root = ET.fromstring(archive.read("content.xml"))

    tables = root.findall(".//table:table", NS)
    if not tables:
        raise ValueError(f"ODS 找不到工作表：{ods_path}")

    rows: list[list[str]] = []
    for table_row in tables[0].findall("table:table-row", NS):
        row: list[str] = []
        for cell in table_row:
            if not cell.tag.endswith("table-cell"):
                continue
            repeated = int(cell.attrib.get(f"{{{TABLE_NS}}}number-columns-repeated", "1"))
            paragraphs = ["".join(p.itertext()) for p in cell.findall(".//text:p", NS)]
            value = "\n".join(paragraphs)
            row.extend([value] * repeated)
        row_repeated = int(table_row.attrib.get(f"{{{TABLE_NS}}}number-rows-repeated", "1"))
        rows.extend([row] * row_repeated)

    if not rows:
        return [], []
    return rows[0], rows[1:]


def _ods_term_entries(path: str | Path, source_layer: str) -> list[dict[str, Any]]:
    header, rows = read_ods_rows(path)
    header_index = {normalize_text(name): index for index, name in enumerate(header)}
    required = {"id", "英文名稱", "中文名稱"}
    missing = required - set(header_index)
    if missing:
        raise ValueError(f"{path} 缺少欄位：{', '.join(sorted(missing))}")

    entries: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        source_id = _column_value(row, header_index, "id")
        english = _column_value(row, header_index, "英文名稱")
        chinese = _column_value(row, header_index, "中文名稱")
        if not english and not chinese:
            continue
        entries.append(
            {
                "source_layer": source_layer,
                "source_id": source_id or f"row-{row_number}",
                "canonical_zh": chinese,
                "canonical_en": english,
                "aliases_zh": split_aliases(chinese),
                "aliases_en": split_aliases(english, english=True),
                "category": "official",
                "priority": "standard",
                "review_status": "source",
                "notes": "",
            }
        )
    return entries


def read_hospital_terms_csv(path: str | Path) -> list[dict[str, Any]]:
    """讀取院內用語 CSV；空白列會被略過。"""

    entries: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"canonical_zh", "canonical_en", "aliases_zh", "aliases_en"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"院內用語 CSV 缺少欄位：{', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            canonical_zh = (row.get("canonical_zh") or "").strip()
            canonical_en = (row.get("canonical_en") or "").strip()
            aliases_zh = split_aliases(row.get("aliases_zh") or "")
            aliases_en = split_aliases(row.get("aliases_en") or "", english=True)
            if not canonical_zh and not canonical_en and not aliases_zh and not aliases_en:
                continue
            entries.append(
                {
                    "source_layer": "hospital_custom",
                    "source_id": f"hospital-custom-row-{row_number}",
                    "canonical_zh": canonical_zh,
                    "canonical_en": canonical_en,
                    "aliases_zh": aliases_zh,
                    "aliases_en": aliases_en,
                    "category": (row.get("category") or "hospital").strip(),
                    "priority": (row.get("priority") or "high").strip(),
                    "review_status": (row.get("review_status") or "pending").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
            )
    return entries


def build_term_dictionary(
    pathology_ods: str | Path,
    medical_laboratory_ods: str | Path,
    hospital_csv: str | Path | None = None,
) -> dict[str, Any]:
    """建立分層詞庫清單，供 tokenizer 與 BM25 索引共用。"""

    entries = _ods_term_entries(pathology_ods, "pathology_core")
    entries.extend(_ods_term_entries(medical_laboratory_ods, "medical_laboratory") )
    if hospital_csv and Path(hospital_csv).exists():
        entries.extend(read_hospital_terms_csv(hospital_csv))

    term_set: set[str] = set()
    for entry in entries:
        for field in ("canonical_zh", "canonical_en"):
            normalized = normalize_text(entry[field])
            if normalized:
                term_set.add(normalized)
        term_set.update(entry["aliases_zh"])
        term_set.update(entry["aliases_en"])

    by_layer = Counter(entry["source_layer"] for entry in entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "entries": entries,
        "terms": sorted(term_set, key=lambda value: (-len(value), value)),
        "stats": {
            "entry_count": len(entries),
            "term_count": len(term_set),
            "entry_count_by_layer": dict(sorted(by_layer.items())),
        },
    }


class JiebaTokenizer:
    """以 jieba 搜尋引擎模式搭配病理自訂詞典的 tokenizer。

    文件建立索引與使用者查詢都必須使用同一個 tokenizer。自訂詞彙會在
    建立 tokenizer 時加入 jieba，因此兩份 ODS 詞庫、院內用語與索引檔中
    保存的 tokenizer_terms 都能使用同一套斷詞規則。
    """

    tokenizer_type = TOKENIZER_TYPE

    def __init__(self, terms: Iterable[str] = (), *, hmm: bool = False) -> None:
        try:
            import jieba
        except ImportError as error:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "BM25 的 JiebaTokenizer 需要安裝 jieba；"
                "請先執行 pip install -r requirements_v6.txt。"
            ) from error

        normalized_terms = {
            normalize_text(term)
            for term in terms
            if normalize_text(term)
        }
        self.terms = tuple(
            sorted(normalized_terms, key=lambda value: (-len(value), value))
        )
        self.hmm = bool(hmm)
        self._tokenizer = jieba.Tokenizer()

        # ODS 的中文名稱、英文名稱與別名都會成為 jieba 自訂詞典。
        # 不指定固定詞頻，讓 jieba 依其 user dictionary 規則補上適當頻率。
        for term in self.terms:
            self._tokenizer.add_word(term)

    @staticmethod
    def _is_cjk(character: str) -> bool:
        return bool(character and _CJK_RE.fullmatch(character))

    def tokenize(self, text: str) -> list[str]:
        normalized = normalize_text(text)
        if not normalized:
            return []

        tokens: list[str] = []
        for piece in self._tokenizer.lcut_for_search(normalized, HMM=self.hmm):
            piece = normalize_text(piece)
            if not piece:
                continue
            if _CJK_RE.search(piece):
                tokens.append(piece)
            else:
                # 保留英文縮寫、型號、數字、單位與含斜線/連字號的詞。
                tokens.extend(_LATIN_RE.findall(piece))

        # 搜尋模式通常會產生長詞與短詞；若某個單字沒有被該版本的 jieba
        # 輸出，仍保留中文單字，避免「癌」查不到「肺癌」等複合詞。
        existing = set(tokens)
        for character in normalized:
            if self._is_cjk(character) and character not in existing:
                tokens.append(character)
                existing.add(character)
        return tokens


# 保留舊名稱，避免外部程式匯入時立即失效；新的 BM25 索引一律使用 jieba。
DictionaryTokenizer = JiebaTokenizer


@dataclass(frozen=True)
class BM25Result:
    document_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


class BM25Index:
    """可以保存成 JSON，並提供 RRF 混合檢索使用的 BM25 索引。"""

    def __init__(
        self,
        tokenizer: JiebaTokenizer,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        dictionary_manifest: Mapping[str, Any] | None = None,
        index_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 必須大於 0")
        if not 0 <= b <= 1:
            raise ValueError("b 必須介於 0 與 1 之間")
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.dictionary_manifest = dict(dictionary_manifest or {})
        self.index_metadata = dict(index_metadata or {})
        self.documents: list[dict[str, Any]] = []
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._doc_lengths: list[int] = []
        self._avgdl = 0.0

    def add_documents(self, documents: Iterable[Mapping[str, Any]]) -> None:
        existing_ids = {str(document["document_id"]) for document in self.documents}
        for raw_document in documents:
            document_id = str(raw_document.get("document_id") or "").strip()
            if not document_id:
                raise ValueError("BM25 文件缺少 document_id")
            if document_id in existing_ids:
                raise ValueError(f"BM25 文件 ID 重複：{document_id}")
            text = str(raw_document.get("text") or "")
            metadata = dict(raw_document.get("metadata") or {})
            self.documents.append(
                {"document_id": document_id, "text": text, "metadata": metadata}
            )
            existing_ids.add(document_id)
        self._rebuild_postings()

    def _rebuild_postings(self) -> None:
        self._postings = defaultdict(dict)
        self._doc_lengths = []
        for document_index, document in enumerate(self.documents):
            counts = Counter(self.tokenizer.tokenize(document["text"]))
            self._doc_lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self._postings[token][document_index] = frequency
        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0

    def search(self, query: str, *, top_k: int = 10) -> list[BM25Result]:
        if top_k <= 0 or not self.documents:
            return []
        query_tokens = self.tokenizer.tokenize(query)

        # tokenizer 為了支援「癌」查找「肺癌」，索引中會保留中文單字。
        # 但對「火星人」這類多字查詢，若直接使用「火／星／人」，
        # 會把只含「火」的「滅火器」文件誤當成相關結果。因此多字
        # 中文查詢只保留兩字以上的中文詞組；單字查詢仍維持原本行為。
        normalized_query = normalize_text(query)
        cjk_character_count = sum(
            1
            for character in normalized_query
            if _CJK_RE.fullmatch(character)
        )
        if cjk_character_count >= 2:
            query_tokens = [
                token
                for token in query_tokens
                if not _CJK_RE.fullmatch(token)
            ]

        query_counts = Counter(query_tokens)
        if not query_counts:
            return []

        document_count = len(self.documents)
        scores: defaultdict[int, float] = defaultdict(float)
        for token, query_frequency in query_counts.items():
            posting = self._postings.get(token)
            if not posting:
                continue
            document_frequency = len(posting)
            idf = math.log(1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            for document_index, term_frequency in posting.items():
                length_norm = 1.0 - self.b + self.b * self._doc_lengths[document_index] / max(self._avgdl, 1.0)
                denominator = term_frequency + self.k1 * length_norm
                scores[document_index] += query_frequency * idf * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            BM25Result(
                document_id=self.documents[index]["document_id"],
                score=score,
                text=self.documents[index]["text"],
                metadata=dict(self.documents[index]["metadata"]),
            )
            for index, score in ranked
            if score > 0
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tokenizer_type": self.tokenizer.tokenizer_type,
            "document_count": len(self.documents),
            "vocabulary_size": len(self._postings),
            "average_document_length": self._avgdl,
            "k1": self.k1,
            "b": self.b,
            "dictionary_term_count": len(self.tokenizer.terms),
        }

    def save(
        self,
        path: str | Path,
        *,
        dictionary_manifest: Mapping[str, Any] | None = None,
        index_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if dictionary_manifest is not None:
            self.dictionary_manifest = dict(dictionary_manifest)
        if index_metadata is not None:
            self.index_metadata = dict(index_metadata)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "settings": {"k1": self.k1, "b": self.b},
            "tokenizer": {
                "type": self.tokenizer.tokenizer_type,
                "hmm": self.tokenizer.hmm,
            },
            "tokenizer_terms": list(self.tokenizer.terms),
            "documents": self.documents,
            "stats": self.stats(),
        }
        if self.index_metadata:
            payload["index_metadata"] = self.index_metadata
        if self.dictionary_manifest:
            payload["dictionary_manifest"] = self.dictionary_manifest
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema_version = payload.get("schema_version")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"不支援的 BM25 schema_version：{schema_version}")
        settings = payload.get("settings") or {}
        tokenizer_settings = payload.get("tokenizer") or {}
        tokenizer_type = tokenizer_settings.get("type", TOKENIZER_TYPE)
        if tokenizer_type != TOKENIZER_TYPE:
            raise ValueError(f"不支援的 BM25 tokenizer：{tokenizer_type}")
        index = cls(
            JiebaTokenizer(
                payload.get("tokenizer_terms") or [],
                hmm=bool(tokenizer_settings.get("hmm", False)),
            ),
            k1=float(settings.get("k1", 1.2)),
            b=float(settings.get("b", 0.75)),
            dictionary_manifest=payload.get("dictionary_manifest") or {},
            index_metadata=payload.get("index_metadata") or {},
        )
        index.add_documents(payload.get("documents") or [])
        return index


def write_dictionary_manifest(dictionary: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(dictionary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
