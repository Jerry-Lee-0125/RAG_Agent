"""病理科文件與 Chroma 向量資料庫核心功能。

這個模組負責把已整理好的文件放進 Chroma：

1. 呼叫共同文件處理器讀取 PDF、DOCX、CSV。
2. 使用相同規則建立 600 字、重疊 150 字的文字塊。
3. 使用 Ollama 的 bge-m3 模型把文字轉成向量。
4. 將向量寫入 Chroma，供 Agent 依語意搜尋。
5. 強制重建 Chroma 時，同步重建 BM25 關鍵字索引。

Chroma 與 BM25 共用同一批文字塊，避免兩邊切割方式不同。
資料處理流程可以簡化為：

文件 → 共同文字塊 → Chroma 向量索引＋BM25 關鍵字索引
"""

import os
import re

# Streamlit 用來快取耗時的嵌入模型，避免每次畫面重新執行都重新載入。
import streamlit as st

# Chroma、LangChain 文件物件與 Ollama 嵌入模型的相關套件。
# 實際讀檔與切塊統一由 build_bm25_index_v6 提供的共同處理器完成。
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from build_bm25_index_v6 import load_processed_documents
from hybrid_retriever_v6 import (
    get_bm25_index_status,
    rebuild_bm25_index,
)


# 所有資料路徑都以本檔案所在位置為基準，避免從不同工作目錄啟動時
# 找不到資料夾。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 清理完成、可以進入知識庫的實體文件存放位置。
processed_data_path = os.path.abspath(
    os.path.expanduser(
        os.getenv(
            "PATHOLOGY_PROCESSED_DATA_PATH",
            os.path.join(BASE_DIR, "01_processed_data"),
        )
    )
)
# Chroma 將向量索引持久化在此資料夾，程式重開後仍可沿用。
chroma_path = os.path.abspath(
    os.path.expanduser(
        os.getenv(
            "PATHOLOGY_CHROMA_PATH",
            os.path.join(BASE_DIR, "02_db", "chroma_db"),
        )
    )
)
# 第一次啟動時若資料夾不存在，自動建立；已存在則不會清除內容。
os.makedirs(processed_data_path, exist_ok=True)


@st.cache_resource
def get_embeddings():
    """建立並快取本機 Ollama 嵌入模型。

    bge-m3 的工作是將文字轉成一串數字向量；內容意思越相近，
    產生的向量通常也越接近。`st.cache_resource` 會讓整個
    Streamlit 工作階段共用同一個模型物件，減少重複載入成本。
    """
    return OllamaEmbeddings(model="bge-m3")


# 模組載入時先取得共用的嵌入模型，後續新增文件與搜尋都使用同一套模型。
embeddings = get_embeddings()


def get_vector_db():
    """建立指向既有 Chroma 資料夾的連線物件。

    這裡不會重新處理所有文件，只是告訴 Chroma：
    1. 向量索引存在哪裡。
    2. 查詢文字要使用哪個嵌入模型轉成向量。
    """
    return Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings,
    )


def _records_to_chroma_documents(records):
    """把共同文字塊轉成 Chroma 可以保存的文件格式。"""

    documents = []
    root = os.path.abspath(processed_data_path)
    for record in records:
        text = str(record.get("text") or "").strip()
        if not text:
            continue
        metadata = dict(record.get("metadata") or {})
        relative_source = str(
            metadata.get("source_relative") or metadata.get("source") or ""
        ).replace("\\", "/")
        source_path = os.path.abspath(os.path.join(root, relative_source))
        metadata["source"] = source_path
        metadata["source_relative"] = relative_source
        metadata.setdefault("hybrid_id", str(record.get("document_id") or ""))
        metadata.setdefault("chunk_id", metadata["hybrid_id"])
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def add_file_to_db(file_path, vectordb):
    """使用共同切塊規則，將單一文件加入 Chroma。"""

    ext = os.path.splitext(file_path)[1].casefold()
    if ext not in {".pdf", ".docx", ".csv"}:
        return False
    records, _ = load_processed_documents(
        processed_data_path,
        paths=[file_path],
        chunk_size=600,
        chunk_overlap=150,
    )
    split_docs = _records_to_chroma_documents(records)
    if not split_docs:
        print(
            f"⚠️ 略過寫入：檔案 {os.path.basename(file_path)} "
            "無法提取純文字內容。"
        )
        return False
    vectordb.add_documents(
        split_docs,
        ids=[document.metadata["hybrid_id"] for document in split_docs],
    )
    return True


def remove_file_from_db(file_path, vectordb):
    """刪除某份文件在向量資料庫中的所有文字塊。

    新增文件時，LangChain 會把原始路徑存入每個文字塊的
    `metadata["source"]`。這裡用相同路徑查出所有 ID 後一次刪除，
    避免使用者刪掉實體檔案後，舊內容仍可被檢索到。

    注意：此函式只刪除 Chroma 索引，不會刪除硬碟上的原始檔案；
    實體檔案由 Streamlit 主程式處理。
    """
    try:
        # 透過 Metadata 中的 source (來源路徑) 找到該檔案所有的 Chunk ID
        result = vectordb.get(where={"source": file_path})
        ids_to_delete = result.get("ids", [])
        
        # 一份長文件通常對應多個 ID，必須全部刪除。
        if ids_to_delete:
            vectordb.delete(ids=ids_to_delete)
    except Exception as e:
        # 刪除失敗不讓整個網頁中斷，但會在主控台留下可除錯資訊。
        print(f"從資料庫移除 {file_path} 時發生錯誤: {e}")


def build_vector_db(vectordb, ui_placeholder=None):
    """使用同一批文字塊重建 Chroma，並同步重建 BM25。"""

    log_text = "### 🔄 混合檢索資料庫重建進度\n\n"

    def update_ui(message):
        nonlocal log_text
        log_text += f"{message}\n\n"
        if ui_placeholder:
            ui_placeholder.markdown(log_text)

    update_ui("✅ **[階段 1/5]** 正在清空舊有向量資料庫...")
    try:
        vectordb.delete_collection()
    except Exception:
        pass

    update_ui("✅ **[階段 2/5]** 正在讀取並切割共同文件片段...")
    records, document_stats = load_processed_documents(
        processed_data_path,
        chunk_size=600,
        chunk_overlap=150,
    )
    split_docs = _records_to_chroma_documents(records)

    if split_docs:
        update_ui("✅ **[階段 3/5]** 正在建立 Chroma 向量索引...")
        new_vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            ids=[document.metadata["hybrid_id"] for document in split_docs],
            persist_directory=chroma_path,
        )
    else:
        new_vectordb = get_vector_db()

    update_ui("✅ **[階段 4/5]** 正在建立 Jieba＋自訂詞典 BM25 索引...")
    try:
        bm25_summary = rebuild_bm25_index(
            processed_data_path=processed_data_path,
        )
        st.session_state["bm25_sync_warning"] = ""
    except Exception as error:
        bm25_summary = {"error": str(error)}
        st.session_state["bm25_sync_warning"] = (
            "BM25 同步失敗，已保留舊索引；目前問答將暫時使用純向量搜尋。\n"
            f"{error}"
        )
        update_ui(
            "⚠️ BM25 同步失敗，Chroma 已完成重建；"
            "問答將暫時退回純向量搜尋。"
        )
    update_ui(
        "✅ **[階段 5/5]** 混合索引完成："
        f"{document_stats.get('document_count', 0)} 個 chunk。"
    )
    del bm25_summary
    return new_vectordb
