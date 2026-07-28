"""病理科本機 LLM、Agent、Tool 與執行紀錄核心功能。

這個模組位在 Streamlit 介面與 RAG 向量資料庫之間，主要負責：

1. 建立並快取 Ollama 對話模型。
2. 建立病理科 Agent，告訴模型何時應使用哪一個工具。
3. 提供四個工具：全庫搜尋、限定文件搜尋、文件清單、主題式文件清單。
4. 用 `search_state` 把工具執行結果帶回主程式。
5. 建立可公開顯示的 Agent 執行紀錄，不揭露模型隱藏推理內容。

問答流程可簡化為：

使用者問題 → Agent 選工具 → Chroma 檢索 → 工具回傳文件片段
→ Agent 依片段作答 → 主程式顯示答案、來源及執行紀錄

此模組不負責畫出完整頁面；Streamlit 畫面與串流事件的接收工作
由 `0725_rag_agent_v5_1.py` 處理。
"""

import os
import time
import unicodedata
from datetime import datetime

# Streamlit 只在此用來快取模型資源。
import streamlit as st

# LangChain 建立 Agent 與工具；ChatOllama 連接本機 Ollama 服務。
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

from rag_core_v5_1 import processed_data_path


@st.cache_resource
def get_llm(model_name, keep_alive_time=None):
    """建立並快取支援工具呼叫的 Ollama 對話模型。

    參數：
        model_name: Ollama 中已安裝的模型名稱。
        keep_alive_time: 可選的模型常駐時間，例如 "30s"。

    `temperature=0.0` 讓回答較穩定；`num_ctx=8192` 是模型單次可參考的
    上下文長度。快取鍵會包含函式參數，因此不同模型各自保有一個實例。
    """
    # 先建立所有模型共用的固定參數。
    llm_kwargs = {
        "model": model_name,
        "num_ctx": 8192,
        "temperature": 0.0,
    }

    if keep_alive_time is not None:
        # 未傳入時沿用 Ollama 預設；傳入時控制模型離開記憶體的時間。
        llm_kwargs["keep_alive"] = keep_alive_time

    return ChatOllama(**llm_kwargs)


def _content_to_text(content):
    """將 ChatOllama 的不同回傳格式統一整理成純文字。

    部分模型直接回傳字串，部分模型則回傳由多個區塊組成的清單。
    主程式只需要可顯示的文字，因此只收集 `type="text"` 的區塊，
    不把其他類型（例如工具呼叫資料）混入畫面。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


def create_pathology_agent(
    chat_model,
    vectordb,
    score_threshold,
    top_k_setting,
    active_document=None,
):
    """建立嘉基病理科問答 Agent，並回傳 Agent 與共享搜尋狀態。

    參數：
        chat_model: 已建立且支援工具呼叫的 Ollama 對話模型。
        vectordb: Chroma 向量資料庫連線。
        score_threshold: 語意相似度最低門檻。
        top_k_setting: 一次最多取回的文字塊數量。
        active_document: 目前限定的文件；沒有限定時為 None。

    回傳：
        `(agent, search_state)`。Agent 用來執行問答；search_state 是
        Agent 內部工具與 Streamlit 主程式共用的字典，主程式可藉此
        顯示來源、候選文件與工具狀態。

    每次使用者提問都會重新呼叫此函式，因此 search_state 只保存
    「本次回答」的結果，不會把前一輪檢索片段混進來。
    """
    # 這個可變字典會被下方工具閉包更新，Agent 執行後主程式仍可讀取。
    search_state = {
        # 一般搜尋或限定文件搜尋實際取回的 LangChain Document。
        "docs": [],
        # 工具名稱與參數的簡化紀錄，供畫面與短期記憶使用。
        "tool_calls": [],
        # 部分檔名同時符合多份文件時，交給使用者選擇的候選清單。
        "document_candidates": [],
        # 成功解析為唯一文件時，保存該文件的名稱與絕對路徑。
        "resolved_document": None,
        # 限定搜尋狀態，例如 resolved、multiple、not_found、no_content。
        "specific_document_status": None,
    }

    def list_available_documents():
        """掃描知識庫資料夾，回傳排序後的合法文件資料。

        每份文件包含：
            display_name: 相對路徑，提供畫面與模型辨識。
            file_name: 不含資料夾的檔名。
            source: Chroma metadata 使用的絕對來源路徑。

        隱藏檔與 Office 暫存檔不應進入文件清單，因此會直接略過。
        """
        documents = []
        for root, dir_names, file_names in os.walk(processed_data_path):
            # 原地修改 dir_names 可阻止 os.walk 進入隱藏或暫存資料夾。
            dir_names[:] = sorted(
                dir_name
                for dir_name in dir_names
                if not dir_name.startswith((".", "~"))
            )
            for file_name in sorted(file_names):
                if file_name.startswith((".", "~")):
                    continue
                # source 必須和向量資料庫 metadata 中保存的路徑一致，
                # 限定文件搜尋時才能用 filter 正確比對。
                source_path = os.path.abspath(os.path.join(root, file_name))
                relative_path = os.path.relpath(
                    source_path,
                    processed_data_path,
                ).replace(os.sep, "/")
                documents.append({
                    "display_name": relative_path,
                    "file_name": file_name,
                    "source": source_path,
                })
        # casefold 比 lower 更適合處理跨語系的大小寫排序。
        return sorted(
            documents,
            key=lambda item: item["display_name"].casefold(),
        )

    def normalize_document_name(value):
        """正規化檔名，降低全形字元、引號與大小寫造成的比對差異。

        NFKC 會把相容字元轉成一致形式；casefold 用於不分大小寫比對。
        僅正規化「比對用字串」，不會更改硬碟上的真實檔名。
        """
        normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
        return normalized.strip("「」『』\"' ").casefold()

    def resolve_document_candidates(document_name):
        """把使用者輸入的檔名解析成零份、一份或多份候選文件。

        比對順序：
            1. 完整相對路徑、完整檔名、移除副檔名後完全相同。
            2. 若無完全相同，再做保守的部分名稱比對。

        回傳 `(候選清單, 狀態)`；狀態可能是 invalid、exact、
        partial 或 not_found。
        """
        requested = normalize_document_name(document_name)
        if not requested:
            return [], "invalid"

        # 不接受絕對路徑或 `..`，避免使用者讓 Agent 掃描知識庫以外的位置。
        path_parts = requested.replace("\\", "/").split("/")
        if os.path.isabs(str(document_name)) or ".." in path_parts:
            return [], "invalid"

        available_documents = list_available_documents()
        # 先找完全相符，避免已有唯一檔名時又納入名稱相似的其他版本。
        exact_matches = []
        for document in available_documents:
            display_name = normalize_document_name(document["display_name"])
            file_name = normalize_document_name(document["file_name"])
            display_stem = normalize_document_name(
                os.path.splitext(document["display_name"])[0]
            )
            file_stem = normalize_document_name(
                os.path.splitext(document["file_name"])[0]
            )
            if requested in {
                display_name,
                file_name,
                display_stem,
                file_stem,
            }:
                exact_matches.append(document)

        if exact_matches:
            return exact_matches, "exact"

        # 完全相符為零份時才做部分比對；多份結果會交由主程式顯示選單。
        partial_matches = []
        for document in available_documents:
            searchable_names = {
                normalize_document_name(document["display_name"]),
                normalize_document_name(document["file_name"]),
                normalize_document_name(
                    os.path.splitext(document["display_name"])[0]
                ),
                normalize_document_name(
                    os.path.splitext(document["file_name"])[0]
                ),
            }
            if any(requested in candidate for candidate in searchable_names):
                partial_matches.append(document)

        return partial_matches, "partial" if partial_matches else "not_found"

    @tool
    def search_pathology_documents(query: str) -> str:
        """搜尋嘉義基督教醫院病理科內部文件。

        只要問題涉及病理科流程、儀器、品質管理、表單、作業規範或文件內容，
        就應優先使用本工具。請輸入一個完整、具體的繁體中文搜尋問題。

        這是「全知識庫」語意搜尋，不限定來源檔案。工具說明也會提供給
        LLM，Agent 會根據這段說明判斷何時應呼叫本工具。
        """
        # 去除空白可避免把無效問題送入嵌入模型。
        query = (query or "").strip()
        if not query:
            search_state["docs"] = []
            return "搜尋問題不可為空白。"

        # 保存實際使用的搜尋句，供執行紀錄與對話記憶摘要使用。
        search_state["tool_calls"].append({
            "name": "search_pathology_documents",
            "query": query,
        })
        # similarity_score_threshold 同時套用「最低相似度」與「最多幾筆」。
        # 門檻過高可能完全找不到片段，過低則可能納入不相關內容。
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": top_k_setting,
            },
        )
        # invoke 會先把 query 向量化，再到 Chroma 找最接近的文字塊。
        docs = retriever.invoke(query)
        search_state["docs"] = docs

        if not docs:
            # 明確要求 Agent 不使用自身常識補答案，降低院內規範幻覺風險。
            return (
                "【檢索結果】目前病理科文件中找不到與此問題相符的內容。"
                "請不要使用模型自身知識補充答案，應直接告知使用者目前資料不足。"
            )

        # 將每個 Document 轉成模型容易閱讀、也保有來源資訊的文字區塊。
        result_blocks = []
        for index, doc in enumerate(docs, start=1):
            source_name = os.path.basename(
                doc.metadata.get("source", "未知文件")
            )
            # PyPDFLoader 的 page 從 0 開始，所以顯示時加 1。
            page_num = doc.metadata.get("page")
            page_text = (
                f"，第 {page_num + 1} 頁"
                if page_num is not None
                else ""
            )
            result_blocks.append(
                f"【參考片段 {index}｜來源：{source_name}{page_text}】\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(result_blocks)

    @tool
    def search_specific_document(document_name: str, query: str) -> str:
        """限定單一病理科文件搜尋內容。

        當使用者明確提到文件名稱，或目前已選定限定文件時使用。
        document_name 必須是已匯入系統的完整或部分檔名，
        query 是要在該文件內查詢的具體問題。

        與全庫搜尋不同，本工具會在 Chroma 查詢中加入 source filter，
        確保取回片段全部來自同一份已解析文件。
        """
        # 每次限定搜尋前先清空上一次結果，避免狀態誤判。
        document_name = (document_name or "").strip()
        query = (query or "").strip()
        search_state["docs"] = []
        search_state["document_candidates"] = []
        search_state["resolved_document"] = None
        search_state["specific_document_status"] = None
        search_state["tool_calls"].append({
            "name": "search_specific_document",
            "document_name": document_name,
            "query": query,
        })

        if not document_name or not query:
            search_state["specific_document_status"] = "invalid"
            return "文件名稱與查詢問題都不可為空白。"

        # 先把自然語言中的檔名解析成知識庫內的真實檔案。
        candidates, match_type = resolve_document_candidates(document_name)
        if match_type == "invalid":
            search_state["specific_document_status"] = "invalid"
            return (
                "指定的文件名稱無效。請從系統已匯入的文件中選擇，"
                "不要輸入絕對路徑或上層目錄符號。"
            )

        if not candidates:
            search_state["specific_document_status"] = "not_found"
            return (
                f"目前系統中找不到名稱符合「{document_name}」的文件。"
                "請確認檔名，或先詢問有哪些相關文件。"
            )

        if len(candidates) > 1:
            # 不自行猜測版本，將候選資料交給 Streamlit 產生下拉選單。
            search_state["specific_document_status"] = "multiple"
            search_state["document_candidates"] = candidates
            return (
                f"找到 {len(candidates)} 份名稱符合「{document_name}」的文件，"
                "請從下方選擇要限定查詢的版本：\n\n"
                + "\n".join(
                    f"- {candidate['display_name']}"
                    for candidate in candidates
                )
            )

        # 只有唯一候選時才真正執行資料庫檢索。
        resolved_document = candidates[0]
        search_state["specific_document_status"] = "resolved"
        search_state["resolved_document"] = resolved_document

        # `filter` 會比對切塊時保留下來的 metadata["source"]。
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": top_k_setting,
                "filter": {"source": resolved_document["source"]},
            },
        )
        docs = retriever.invoke(query)
        search_state["docs"] = docs

        if not docs:
            search_state["specific_document_status"] = "no_content"
            return (
                f"已限定文件「{resolved_document['display_name']}」，"
                "但沒有找到與問題相符的內容。請改用更具體的關鍵字。"
            )

        result_blocks = []
        for index, doc in enumerate(docs, start=1):
            source_name = os.path.basename(
                doc.metadata.get("source", "未知文件")
            )
            page_num = doc.metadata.get("page")
            page_text = (
                f"，第 {page_num + 1} 頁"
                if page_num is not None
                else ""
            )
            result_blocks.append(
                f"【限定文件片段 {index}｜來源：{source_name}{page_text}】\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(result_blocks)

    @tool
    def list_pathology_documents() -> str:
        """列出目前已匯入病理科知識庫的文件名稱。

        這個工具只讀取硬碟檔名，不進行向量搜尋，也不要求模型推測
        文件內容。工具結果可直接顯示，避免 LLM 改寫或遺漏檔名。
        """
        search_state["tool_calls"].append({
            "name": "list_pathology_documents"
        })
        files = [
            document["display_name"]
            for document in list_available_documents()
        ]
        if not files:
            return "目前病理科知識庫沒有已匯入的文件。"
        return (
            "目前已匯入的病理科文件：\n"
            + "\n".join(f"- {file_name}" for file_name in files)
        )

    @tool
    def find_pathology_documents_by_topic(topic: str) -> str:
        """依照主題列出知識庫中可能相關的文件名稱。

        結果來源有兩種：
            1. 主題文字直接出現在檔名中。
            2. 對主題進行向量搜尋後，由命中的文字塊反查來源檔案。

        兩種結果合併並去除重複，最後只回傳檔名，不讓模型自行替
        文件分類或臆測內容。
        """
        topic = (topic or "").strip()
        search_state["topic_docs"] = []
        search_state["topic_document_names"] = []

        if not topic:
            return "查詢主題不可為空白。"

        search_state["tool_calls"].append({
            "name": "find_pathology_documents_by_topic",
            "topic": topic,
        })

        # 先準備 source 絕對路徑與畫面相對路徑的對照表。
        available_documents = list_available_documents()
        files = [
            document["display_name"]
            for document in available_documents
        ]
        source_display_names = {
            os.path.normcase(document["source"]): document["display_name"]
            for document in available_documents
        }
        # 第一條路徑：直接比對檔名，適合文件名稱本身就包含主題的情況。
        filename_matches = [
            file_name
            for file_name in files
            if topic.lower() in file_name.lower()
        ]

        # 第二條路徑：從內容語意找相關文件。主題清單需要較廣的候選，
        # 因此 k 至少取 20，再依來源檔案去重。
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": score_threshold,
                "k": max(top_k_setting, 20),
            },
        )
        docs = retriever.invoke(topic)
        search_state["topic_docs"] = docs

        # list 保留顯示順序，set 則負責快速判斷是否已加入。
        related_files = []
        seen_files = set()

        def add_related_file(file_name):
            """依原有順序加入檔名，並略過空值與重複項目。"""
            if file_name and file_name not in seen_files:
                seen_files.add(file_name)
                related_files.append(file_name)

        for file_name in filename_matches:
            add_related_file(file_name)

        # 一份文件可能命中多個文字塊，add_related_file 只會保留一次。
        for doc in docs:
            source_path = os.path.abspath(
                doc.metadata.get("source", "未知文件")
            )
            source_name = source_display_names.get(
                os.path.normcase(source_path),
                os.path.basename(source_path),
            )
            add_related_file(source_name)

        search_state["topic_document_names"] = related_files

        if not related_files:
            return (
                f"目前系統中沒有找到與「{topic}」明確相關的病理科文件。"
            )

        return (
            f"有，系統中找到 {len(related_files)} 份可能與「{topic}」"
            "相關的病理科文件：\n\n"
            + "\n".join(f"- {file_name}" for file_name in related_files)
        )

    # 將 Streamlit 已選定的文件轉成系統提示詞，讓後續追問持續限定
    # 同一份文件；沒有選定時則禁止 Agent 從舊對話自行推測限定文件。
    active_document_name = ""
    if isinstance(active_document, dict):
        active_document_name = str(
            active_document.get("display_name", "")
        ).strip()
    active_document_instruction = (
        f"\n目前使用者已限定文件：「{active_document_name}」。"
        "\n只要是詢問文件內容、流程、規範、表單或操作的後續問題，"
        "必須使用 search_specific_document，並將此檔名作為 document_name。"
        "\n若使用者在新問題中明確指定另一份文件，則以新文件名稱為準，"
        "重新解析並更新限定文件。"
        if active_document_name
        else (
            "\n目前沒有設定限定文件。即使歷史對話曾提到或選定文件，"
            "也不可將其視為目前限定文件；只有使用者在新問題中明確指定"
            "文件時，才能使用 search_specific_document。"
        )
    )

    # 系統提示詞是 Agent 的最高層工作規則，重點是：
    # - 院內文件問題必須先使用工具。
    # - 只依檢索內容回答，資料不足時明確說明。
    # - 文件清單直接照工具結果顯示，不交給模型重組。
    # - 不提供超出院內文件範圍的醫療建議。
    system_prompt = f"""
你是嘉義基督教醫院病理科的內部文件問答 Agent，所有對使用者的最終回答都必須使用繁體中文。
{active_document_instruction}

你的工作原則：
1. 使用者明確提到某份文件名稱，或目前已有上述限定文件，且問題是在查詢文件內容時，
   必須使用 search_specific_document，不可改用全資料庫搜尋。
2. 若使用者詢問知識庫有哪些文件，使用 list_pathology_documents，並直接列出工具回傳的檔名清單。
   不要自行改寫成 Overview、Category、Key Documents、What They Cover 等英文表格，
   不要替文件分類、摘要或推測內容；檔名請保留原文，不要翻譯。
3. 若使用者詢問有哪些某主題相關文件，使用 find_pathology_documents_by_topic，
   並直接列出工具回傳的檔名清單。不要自行分類、摘要、翻譯或推測文件涵蓋內容。
4. 若使用者詢問某流程怎麼做、規範內容是什麼、表單如何填寫或儀器如何操作，
   且沒有指定或限定文件，使用 search_pathology_documents 搜尋文件內容後再回答。
5. 只能根據工具回傳的文件內容回答，不得使用模型自身知識補充、推測或捏造院內規定。
6. 如果工具找不到相關資料，請明確回答「抱歉，在目前的病理科規範資料中找不到相關資訊」，
   並建議使用者提供更具體的流程名稱或關鍵字。
7. 回答流程或規範時，請使用繁體中文條列式步驟；回答時盡量標示使用到的來源檔名與頁碼。
8. 不提供診斷、治療或其他超出病理科內部文件範圍的醫療建議。
9. 不要揭露你的內部推理過程，只呈現必要的工具使用結果、結論與來源。
10. 除了檔案名稱、英文縮寫、儀器型號或文件原文用語之外，不要在最終回答中使用英文。
"""

    # create_agent 會建立可循環執行「模型判斷 → 工具 → 模型回答」的
    # LangGraph Agent。四個工具的 docstring 也會成為模型選工具的依據。
    agent = create_agent(
        model=chat_model,
        tools=[
            search_pathology_documents,
            search_specific_document,
            list_pathology_documents,
            find_pathology_documents_by_topic,
        ],
        system_prompt=system_prompt,
        name="pathology_rag_agent",
    )
    # search_state 與工具共用同一個物件，主程式可在串流結束後讀取結果。
    return agent, search_state


# 工具內部名稱必須固定，畫面則用較易懂的中文名稱顯示。
# 集中在此對照可避免主程式各處重複判斷。
TOOL_DISPLAY_NAMES = {
    "search_pathology_documents": (
        "病理文件搜尋（search_pathology_documents）"
    ),
    "list_pathology_documents": (
        "病理文件清單（list_pathology_documents）"
    ),
    "find_pathology_documents_by_topic": (
        "主題式文件清單（find_pathology_documents_by_topic）"
    ),
    "search_specific_document": (
        "限定文件搜尋（search_specific_document）"
    ),
}


def get_tool_display_name(tool_name):
    """將工具的程式名稱轉成適合顯示於介面的中文名稱。

    若未收錄在對照表中，保留原始名稱，方便開發階段辨識新工具。
    """
    return TOOL_DISPLAY_NAMES.get(tool_name, tool_name or "未知工具")


def append_agent_execution_event(
    execution_log,
    started_at,
    title,
    detail,
    state="info",
    event_type="progress",
    **metadata,
):
    """新增一筆可公開、可保存的 Agent 執行事件。

    事件內容只包含時間、工具名稱、搜尋條件、結果筆數與處理狀態等
    可安全顯示的資訊，不保存或揭露模型內部的逐字推理。

    `metadata` 可依事件種類附加 tool_name、query、document_count 等欄位。
    函式會把事件加入 execution_log，並同時回傳該事件。
    """
    # perf_counter 適合計算經過時間，不受系統時間校正影響。
    elapsed_seconds = time.perf_counter() - started_at
    previous_elapsed_seconds = 0.0
    if execution_log:
        # 上一事件的累積秒數用來算出本階段單獨花費的時間。
        previous_elapsed = execution_log[-1].get("elapsed_seconds")
        if isinstance(previous_elapsed, (int, float)):
            previous_elapsed_seconds = float(previous_elapsed)

    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 累積時間保留於紀錄中，供總耗時、分析及除錯使用。
        "elapsed_seconds": round(elapsed_seconds, 2),
        # 前端顯示本事件與上一事件之間的獨立處理時間。
        "duration_seconds": round(
            max(0.0, elapsed_seconds - previous_elapsed_seconds),
            2,
        ),
        "title": title,
        "detail": detail,
        "state": state,
        "event_type": event_type,
    }
    # 額外欄位直接併入事件，讓不同事件可保留各自需要的資訊。
    event.update(metadata)
    execution_log.append(event)
    return event


def format_agent_execution_event(
    event,
    index=None,
    previous_elapsed_seconds=None,
):
    """將一筆執行事件整理成可由 Streamlit 顯示的 Markdown。

    新版事件直接保存 `duration_seconds`；相容舊紀錄時，若沒有此欄位，
    便用目前與上一筆的累積秒數相減補算。
    """
    state_icons = {
        "complete": "✅",
        "running": "⏳",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️",
    }
    icon = state_icons.get(event.get("state"), "•")
    index_text = f"{index}. " if index is not None else ""

    duration = event.get("duration_seconds")
    # 舊版紀錄只有 elapsed_seconds，因此保留回推階段耗時的相容邏輯。
    if (
        not isinstance(duration, (int, float))
        and isinstance(previous_elapsed_seconds, (int, float))
    ):
        elapsed = event.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            duration = max(0.0, elapsed - previous_elapsed_seconds)

    duration_text = (
        f"`耗時 {duration:.2f} 秒` "
        if isinstance(duration, (int, float))
        else ""
    )
    timestamp = event.get("timestamp")
    timestamp_text = f"`{timestamp}` " if timestamp else ""
    return (
        f"{icon} **{index_text}{event.get('title', '')}**  \n"
        f"{timestamp_text}{duration_text}{event.get('detail', '')}"
    )


def emit_live_agent_event(
    status_container,
    execution_log,
    started_at,
    seen_event_keys,
    event_key,
    title,
    detail,
    state="info",
    event_type="progress",
    **metadata,
):
    """顯示一筆即時事件，並同步加入可保存的執行紀錄。

    `event_key` 是去重鍵。LangGraph 的串流事件可能讓同一節點被觀察
    多次，因此先用 `seen_event_keys` 檢查，避免畫面與歷史紀錄重複。

    回傳新事件；若同一 event_key 已處理過，回傳 None。
    """
    if event_key in seen_event_keys:
        return None

    # 先登記鍵值再寫入與顯示，確保同一執行流程只出現一次。
    seen_event_keys.add(event_key)
    event = append_agent_execution_event(
        execution_log=execution_log,
        started_at=started_at,
        title=title,
        detail=detail,
        state=state,
        event_type=event_type,
        **metadata,
    )
    # status_container 是主程式建立的 st.status，可在 Agent 執行期間更新。
    status_container.markdown(
        format_agent_execution_event(event, index=len(execution_log))
    )
    return event
