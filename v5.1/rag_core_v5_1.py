"""病理科 RAG 文件處理與 Chroma 向量資料庫核心功能。

這個模組負責 RAG（檢索增強生成）流程中的「資料準備」部分：

1. 從 PDF、Word 或 CSV 檔案讀取文字。
2. 清理多餘空白，讓文字格式較一致。
3. 把長文件切成可檢索的小段文字（Chunk）。
4. 使用 Ollama 的 bge-m3 模型把文字轉成向量。
5. 將向量寫入 Chroma，供 Agent 日後依語意搜尋。

主程式只需要呼叫本模組提供的函式，不必知道文件切塊與資料庫的
實作細節。資料處理流程可簡化為：

原始檔案 → 讀取文字 → 清理文字 → 切塊 → 產生向量 → 寫入 Chroma
"""

import os
import re

# Streamlit 用來快取耗時的嵌入模型，避免每次畫面重新執行都重新載入。
import streamlit as st

# LangChain / Chroma 負責文件載入、切塊、向量化與向量資料庫存取。
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


# 所有資料路徑都以本檔案所在位置為基準，避免從不同工作目錄啟動時
# 找不到資料夾。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 清理完成、可以進入知識庫的實體文件存放位置。
processed_data_path = os.path.join(BASE_DIR, "01_processed_data")
# Chroma 將向量索引持久化在此資料夾，程式重開後仍可沿用。
chroma_path = os.path.join(BASE_DIR, "02_db", "chroma_db")
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


def add_file_to_db(file_path, vectordb):
    """將單一檔案讀取、切塊並新增至 Chroma。

    參數：
        file_path: 要處理的實體檔案路徑。
        vectordb: 已建立的 Chroma 連線物件。

    回傳：
        True 代表至少有一個文字塊成功寫入；False 代表格式不支援、
        文件沒有可擷取文字，或切塊後沒有內容。

    這是「局部更新」流程，上傳一份新文件時只處理該文件，不必重建
    整個知識庫，因此比全部重建更省時間與運算資源。
    """
    # 只保留小寫副檔名，讓 .PDF 與 .pdf 能使用同一套判斷。
    ext = os.path.splitext(file_path)[1].lower()
    
    # 根據副檔名選擇對應的 LangChain 載入器
    if ext == '.pdf':
        # PDF 通常以「每一頁」建立一份 Document，並保留頁碼 metadata。
        docs = PyPDFLoader(file_path).load()
    elif ext == '.docx':
        # Word 文件會轉成純文字；一般不會提供像 PDF 一樣的頁碼。
        docs = Docx2txtLoader(file_path).load()
    elif ext == '.csv':
        # utf-8-sig 可相容含 BOM 的中文 CSV，降低 Excel 匯出後的亂碼問題。
        docs = CSVLoader(file_path, encoding='utf-8-sig').load()
    else:
        # 其他格式應先由主程式轉成 DOCX 或 CSV，再交給此函式。
        return False
    
    if docs:
        # 一個檔案可能產生多份 Document，例如 PDF 每一頁各一份，
        # 因此要逐份清理 page_content。
        for doc in docs:
            raw_text = doc.page_content
            # 1. 將連續的空白行強制壓縮成標準的雙換行
            clean_text = re.sub(r'\n\s*\n+', '\n\n', raw_text)
            # 2. 清除每一行開頭的空白與 Tab，保持對齊
            clean_text = "\n".join([line.lstrip() for line in clean_text.split('\n')])
            # 3. 清除整個段落頭尾多餘的空白與隱藏字元
            doc.page_content = clean_text.strip()
            
        # 文字塊太大會混入過多主題；太小則容易失去上下文。
        # 目前採 600 字並重疊 150 字，讓跨切點的句子仍保有前後語意。
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,     
            chunk_overlap=150,  # 重疊 150 字，避免跨段落語意被切斷
            separators=["\n\n", "\n", "。", "，", " ", ""]  # 優先依照段落和標點符號做切割
        )
        # split_documents 會保留原 Document 的 source、page 等 metadata，
        # 日後才能顯示來源檔名、頁碼，或依 source 限定文件搜尋。
        split_docs = text_splitter.split_documents(docs)
        
        if split_docs:
            # Chroma 會透過 embeddings 將每個文字塊轉成向量後寫入索引。
            vectordb.add_documents(split_docs)
            return True
            
    # 如果讀取出來是空的（例如純圖片的 PDF）
    print(f"⚠️ 略過寫入：檔案 {os.path.basename(file_path)} 無法提取純文字內容。")
    return False # ❌ 無效檔案，回傳 False


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
    """清除舊索引，重新掃描資料夾並建立整套 Chroma 資料庫。

    適用情境：
        - 向量資料庫內容與實體檔案不同步。
        - 切塊或嵌入設定改變，需要重新產生全部向量。
        - 管理員手動要求完整重建。

    參數：
        vectordb: 要清空的既有 Chroma 物件。
        ui_placeholder: 可選的 Streamlit 佔位元件，用來即時更新進度。

    回傳：
        新建立的 Chroma 物件；若資料夾中沒有文件，回傳空資料庫連線。

    完整流程：
        清空索引 → 掃描三種格式 → 清理 → 切塊 → 向量化並寫入
    """
    
    log_text = "### 🔄 向量資料庫重建進度\n\n"
    
    def update_ui(msg):
        """累積進度訊息，並在有傳入 Streamlit 元件時更新畫面。"""
        nonlocal log_text
        # 使用累加方式保留前面階段，使用者可看到完整處理進度。
        log_text += f"{msg}\n\n"
        if ui_placeholder:
            ui_placeholder.markdown(log_text)
    
    
    try:
        # 清空向量資料庫內的所有資料，而不是刪除實體資料夾，避免遇到「檔案被鎖定」的報錯
        update_ui("✅ **[階段 1/5]** 正在清空舊有向量資料庫...")
        vectordb.delete_collection()
    except Exception as e:
        # 初次建立或空資料庫可能沒有 collection；此狀況不影響後續重建。
        pass

    # 重新讀取資料夾內所有支援的檔案
    update_ui("✅ **[階段 2/5]** 正在掃描並讀取硬碟中的病理科檔案...")
    # **/* 代表連子資料夾也會一併掃描。
    pdf_docs = DirectoryLoader(
        processed_data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    ).load()
    docx_docs = DirectoryLoader(
        processed_data_path,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
    ).load()
    csv_docs = DirectoryLoader(
        processed_data_path,
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={'encoding': 'utf-8-sig'},
    ).load()

    # 後續清理與切塊邏輯與檔案格式無關，因此合併成同一份清單。
    docs = pdf_docs + docx_docs + csv_docs
    
    # 若沒有任何文件，回傳空的 Chroma 實例
    if not docs:
        return get_vector_db()
    
    update_ui("✅ **[階段 3/5]** 正在執行資料清洗...")
    for doc in docs:
        raw_text = doc.page_content
        # 壓縮連續的空白行
        clean_text = re.sub(r'\n\s*\n+', '\n\n', raw_text)
        # 清除每一行開頭的空白與 Tab
        clean_text = "\n".join([line.lstrip() for line in clean_text.split('\n')])
        # 清除頭尾多餘字元
        doc.page_content = clean_text.strip()

    update_ui("✅ **[階段 4/5]** 正在進行文本分割...")    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=150,  
        separators=["\n\n", "\n", "。", "，", " ", ""]  
    )
    # metadata 會跟著切塊保留下來，供來源顯示與限定文件檢索使用。
    split_docs = text_splitter.split_documents(docs)

    # 建立並儲存新的 Chroma 向量資料庫
    update_ui("✅ **[階段 5/5]** 嵌入模型轉換向量並寫入資料庫...")
    # from_documents 會一次完成向量轉換、建立 collection 與持久化。
    new_vectordb = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        persist_directory=chroma_path 
    )
    return new_vectordb  
