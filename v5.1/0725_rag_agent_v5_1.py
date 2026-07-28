"""
病理科 RAG + LLM Agent 檢索問答系統的 Streamlit 主程式。

本檔案負責「使用者介面與流程協調」，核心工作包含：

1. 顯示聊天畫面、側邊欄設定、來源片段及 Agent 執行紀錄。
2. 提供管理員登入、文件上傳、轉檔、刪除及資料庫重建。
3. 使用 Session State 保存聊天歷史、向量資料庫連線與限定文件。
4. 將使用者問題、近期記憶與檢索設定交給 Agent。
5. 同時接收 Agent 的文字串流與狀態更新，逐步顯示回答。
6. 保存答案、來源、工具呼叫與安全執行紀錄，供重新整理後查閱。

三個程式模組的分工：

- 0725_rag_agent_v5_1.py：Streamlit 畫面、檔案管理、對話與串流控制。
- rag_core_v5_1.py：文件讀取、文字清洗、切塊及 Chroma 向量資料庫。
- agent_core_v5_1.py：Ollama 模型、Agent、檢索工具與執行事件。

一般問答的完整流程：

輸入問題 → 組合近期對話 → 建立 Agent → Agent 選擇工具
→ Chroma 找文件片段 → Agent 依片段回答 → 顯示答案與來源
→ 將本輪結果存入 Session State





2026-07-25 Agent v5

- 1.新增限定文件搜尋
    支援依完整或部分檔名限定 Chroma source，避免混入其他文件片段。

- 2.新增多版本選擇與持續限定
    多份名稱相符文件由使用者選擇，並在後續內容問題中持續限定同一份文件。

2026-07-27 Agent v5.1

- 1.集中管理系統參數
    LLM 模型、檢索相似度門檻、檢索文本數量、記憶模式與記憶輪數
    全部移至管理員模式。一般使用者沿用目前設定，但不顯示調整元件。

"""

# Python 標準函式庫：處理路徑、檔名、轉檔程序、時間與文字清理。
import os
import math
import glob
import shutil
import subprocess
import time

# 第三方套件：Streamlit 負責網頁，Pandas 處理表格與 Excel/CSV。
import streamlit as st
import pandas as pd
from datetime import datetime

# AIMessageChunk 用於辨識模型串流回傳的文字片段。
from langchain.messages import AIMessageChunk

# RAG 核心：管理實體文件與 Chroma 向量資料庫。
from rag_core_v5_1 import (
    processed_data_path,
    get_vector_db,
    add_file_to_db,
    remove_file_from_db,
    build_vector_db,
)
# Agent 核心：建立模型、工具型 Agent，以及格式化安全執行紀錄。
from agent_core_v5_1 import (
    get_llm,
    _content_to_text,
    create_pathology_agent,
    get_tool_display_name,
    emit_live_agent_event,
    format_agent_execution_event,
)

# 這行必須在其他 Streamlit 畫面元件之前呼叫。
# layout="wide" 讓檔案表格、來源片段與聊天內容有較寬的顯示空間。
st.set_page_config(page_title="病理科檢索問答系統", layout="wide")

# ==========================================
# 1. Agent 執行紀錄與來源顯示
# ==========================================
def render_agent_execution_log(execution_log, expanded=False):
    """顯示回答完成後仍可展開查閱的 Agent 執行紀錄。

    execution_log 是事件字典清單，每筆包含標題、說明、時間與狀態。
    `expanded` 決定折疊區一開始是否展開。此處只顯示可公開的步驟，
    不顯示模型的隱藏推理內容。
    """
    if not execution_log:
        return

    with st.expander("Agent 執行紀錄（可查閱）", expanded=expanded):
        st.caption(
            "此處只顯示問題分類、工具呼叫、檢索結果與生成狀態。"
        )
        # 保留上一筆累積秒數，讓舊版紀錄也可算出單一階段耗時。
        previous_elapsed_seconds = 0.0
        for index, event in enumerate(execution_log, start=1):
            st.markdown(
                format_agent_execution_event(
                    event,
                    index=index,
                    previous_elapsed_seconds=previous_elapsed_seconds,
                )
            )
            elapsed_seconds = event.get("elapsed_seconds")
            if isinstance(elapsed_seconds, (int, float)):
                previous_elapsed_seconds = float(elapsed_seconds)


def render_source_documents(docs):
    """將 Agent 本次取回的 LangChain Document 顯示成來源折疊區。

    每個 Document 的 metadata 會保留 source；PDF 通常另有從 0 開始
    的 page。畫面將頁碼加 1，以符合一般使用者閱讀習慣。
    """
    if not docs:
        st.write("沒有找到相關的參考資料。")
        return

    st.markdown("#### 🔍 Agent 檢索來源片段：")
    for index, doc in enumerate(docs, start=1):
        # 介面只顯示檔名，不顯示伺服器上的完整絕對路徑。
        source_name = os.path.basename(doc.metadata.get("source", "未知"))
        page_num = doc.metadata.get("page")
        header_text = f"來源 {index}: {source_name}"
        if page_num is not None:
            header_text += f" (第 {page_num + 1} 頁)"

        with st.expander(header_text):
            st.markdown(
                f'<div style="font-size: 0.85em; color: #505050;">'
                f"{doc.page_content}</div>",
                unsafe_allow_html=True,
            )

# ==========================================
# 2. 檔案管理介面
# ==========================================
@st.dialog("📁 檔案管理", width="large")
def file_management_center():
    """顯示管理員專用的檔案管理中心。

    兩個分頁分別負責：
        1. 上傳：檢查加密與同名檔，必要時轉檔，再局部寫入 Chroma。
        2. 管理：刪除實體文件及其對應的全部向量文字塊。

    這是有副作用的管理功能，因此只會由已登入的管理員按鈕開啟。
    """
    # 兩個頁籤共用同一個對話框，操作完成後透過 st.rerun 更新主畫面。
    tab_upload, tab_manage = st.tabs(["📤 上傳新文件", "🗑️ 管理現有文件"])

    # --- 分頁 1：上傳功能 ---
    with tab_upload:
        st.write("支援檔案類型: pdf, docx, doc, csv, xls, xlsx")
        
        # 建立檔案上傳器，支援多檔案同時上傳
        uploaded_files = st.file_uploader(
            "選擇檔案", 
            type=['pdf', 'docx', 'doc', 'csv', 'xls', 'xlsx'], # 限定上傳檔案類型，避免上傳不支援的格式
            accept_multiple_files=True, # 支援多檔案上傳
            key="dialog_uploader"
            )
        
        if st.button("上傳並更新資料庫", icon="💾", use_container_width=True):
            if uploaded_files:
                save_count = 0 # 紀錄成功寫入資料庫的檔案數量
                
                failed_records = [] # 用來收集上傳失敗的檔案清單，方便後續顯示給使用者看
                total_files = len(uploaded_files) # 取得總檔案數
                
                # 在網頁畫面上建立進度條與狀態文字的佔位符
                progress_bar = st.progress(0)
                status_text = st.empty()
                
         
                with st.spinner("系統正在處理檔案，請稍候..."):
                    # 使用 enumerate 取得目前的索引值 (idx)，方便計算進度
                    for idx, file in enumerate(uploaded_files):
                        
                        if file.name.startswith('~'): # 略過系統產生的暫存檔 (例如打開 Word 時產生的 ~$ 檔案)
                            continue
                        
                        # 即時更新網頁上的狀態文字，讓使用者知道目前處理到哪一份文件
                        current_step = idx + 1
                        status_text.write(f"⏳ 正在處理 ({current_step}/{total_files})：**{file.name}** ...")
                        
                        file_path = os.path.join(processed_data_path, file.name)
                        name, ext = os.path.splitext(file.name)
                        ext = ext.lower() # 將副檔名轉為小寫，避免因大小寫差異導致判斷錯誤
                        
                        # 1. --- 攔截加密檔案(防呆機制) ---
                        # 在寫入硬碟前，先檢查檔案是否被密碼保護，避免後續轉檔程式錯誤
                        is_encrypted = False
                        try:
                            import pypdf
                            import msoffcrypto
                            
                            if ext == '.pdf':
                                if pypdf.PdfReader(file).is_encrypted:
                                    is_encrypted = True
                            elif ext in ['.doc', '.docx', '.xls', '.xlsx']:
                                if msoffcrypto.OfficeFile(file).is_encrypted():
                                    is_encrypted = True
                                    
                            # 檢查完畢後，必須將檔案讀取指標歸零 (Seek 0)
                            # 否則下面寫入硬碟時會從檢查結束的地方開始讀，導致存出 0 byte 的損壞檔案
                            file.seek(0) 
                        except Exception:
                            file.seek(0) # 萬一檢查套件失敗，依然歸零放行，交給後面的 except 捕捉

                        if is_encrypted:
                            # 發現加密，加入失敗清單並直接跳過這個檔案，不寫入硬碟
                            failed_records.append({"檔案名稱": file.name, "失敗原因": "檔案已加密，請解除密碼後再上傳"})
                            continue 
              
                
                        # 2. --- 檢查重複檔名 (安全阻擋機制) ---
                        # 避免覆蓋舊檔案導致 Chroma 向量資料庫出現對應不上的資訊
                        # 掃描資料夾內是否有相同主檔名的檔案 (例如找 細胞學檢查規範.*)
                        search_pattern = os.path.join(processed_data_path, f"{name}.*")
                        existing_files = glob.glob(search_pattern)
                        
                        # 如果上傳的是 Excel，也要一併檢查是否已經有之前拆解出來的 CSV 檔
                        search_pattern_csv = os.path.join(processed_data_path, f"{name}_*.csv")
                        existing_files.extend(glob.glob(search_pattern_csv))
                        
                        # 如果找到任何同名的舊檔案，立刻攔阻
                        if existing_files:
                            failed_records.append({
                                "檔案名稱": file.name, 
                                "失敗原因": "已存在同名檔案。為避免資料遺失，請先至「管理現有文件」手動刪除舊檔後再上傳"
                            })
                            continue # 直接跳過這個檔案，不寫入硬碟也不轉檔，繼續處理下一個檔案
                            
                        # 3. --- 正式寫入硬碟 ---
                        # 確定沒有加密與同名衝突後，才將檔案暫存到硬碟中
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        
                        # 4. --- 轉檔與資料庫寫入 ---
                        try:
                            # [狀況 A] 處理舊版 Word (.doc) -> 呼叫本機端 LibreOffice 在背景轉成 .docx 檔
                            if ext == '.doc':
                                soffice_path = shutil.which('libreoffice') or shutil.which('soffice') or r"C:\Program Files\LibreOffice\program\soffice.exe"
                                subprocess.run([soffice_path, '--headless', '--convert-to', 'docx', '--outdir', processed_data_path, file_path], check=True, timeout=60) # --headless 代表不開啟軟體畫面，在背景執行轉檔
                                os.remove(file_path) # 轉檔成功後刪除原始 .doc
                                
                                new_docx_path = os.path.join(processed_data_path, f"{name}.docx")
                                if add_file_to_db(new_docx_path, st.session_state.vectordb):
                                    save_count += 1
                                else:
                                    os.remove(new_docx_path) 
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "轉檔後無法提取純文字"})
                                
                            # [狀況 B] 處理 Excel (.xls, .xlsx) -> 拆解成多個 CSV 工作表
                            # Excel檔中可能帶有多個工作表，透過 pandas 將每個工作表獨立拆分成 CSV，有助於 LLM 精準檢索
                            elif ext in ['.xls', '.xlsx']:
                                excel_dict = pd.read_excel(file_path, sheet_name=None, dtype=str)
                                valid_csv_count = 0
                                for sheet_name, df in excel_dict.items():
                                    # 資料清洗：移除整行或整列都是空值的無效數據
                                    df.dropna(how='all', inplace=True)
                                    df.dropna(how='all', axis=1, inplace=True)
                                    if not df.empty:
                                        csv_filename = f"{name}_{sheet_name}.csv"
                                        csv_path = os.path.join(processed_data_path, csv_filename)
                                        df.to_csv(csv_path, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 編碼，確保繁體中文不會變成亂碼
                                        
                                        if add_file_to_db(csv_path, st.session_state.vectordb):
                                            valid_csv_count += 1
                                        else:
                                            os.remove(csv_path) 
                                            
                                os.remove(file_path) # 拆解完成後刪除原始的 Excel 檔案
                                if valid_csv_count > 0:
                                    save_count += 1
                                else:
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "無法提取有效表格資料"})
                            
                            # [狀況 C] 不需要轉檔的 PDF, DOCX, CSV -> 直接寫入向量資料庫
                            else:
                                if add_file_to_db(file_path, st.session_state.vectordb):
                                    save_count += 1
                                else:
                                    os.remove(file_path) 
                                    # 加入失敗清單
                                    failed_records.append({"檔案名稱": file.name, "失敗原因": "純圖片或無法解析的內容"})
                                
                        except Exception as e:
                            # 錯誤捕捉：如果轉檔或寫入過程崩潰，將剛剛寫入硬碟的殘留檔案刪除，保持資料夾乾淨
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            error_msg = str(e).lower()
                            # 攔截 Pandas 拋出的加密錯誤，轉換為中文提示
                            if "encrypted" in error_msg or "password" in error_msg:
                                failed_records.append({"檔案名稱": file.name, "失敗原因": "檔案已加密或受密碼保護，請解鎖後再上傳"})
                            else:
                                failed_records.append({"檔案名稱": file.name, "失敗原因": f"系統處理錯誤 ({e})"})
                            
                    # 單個檔案處理完畢後，更新進度條的百分比
                    progress_percentage = current_step / total_files
                    progress_bar.progress(progress_percentage)
                    
                # 迴圈結束後，清除網頁上的進度條與狀態文字，保持版面乾淨
                status_text.empty()
                progress_bar.empty()
                
                
                # 5. --- 上傳結果呈現 ---
                if save_count == len(uploaded_files):
                    # 情況一：全部成功，記錄提示訊息並重新整理網頁
                    st.session_state["show_success_toast"] = f"✅ 成功處理全部 {save_count} 份檔案！"
                    st.rerun()
                else:
                    # 情況二：有部分失敗情況，顯示資料表讓使用者知道哪些檔案有問題
                    if save_count > 0:
                        st.success(f"✅ 已成功寫入 {save_count} 份檔案。")
                    
                    st.error(f"⚠️ 發現 {len(failed_records)} 份檔案無法寫入，上傳失敗，請人工確認內容：")
                    
                    # 利用 Pandas DataFrame 呈現乾淨的表格，hide_index=True 去除最左邊的數字序號
                    df_failed = pd.DataFrame(failed_records)
                    st.dataframe(df_failed, hide_index=True, use_container_width=True)
            else:
                st.error("請先選擇要上傳的檔案。") # 若使用者沒選擇檔案就按下按鈕的防呆提示

    # --- 分頁 2：管理/刪除功能 ---
    with tab_manage:
        file_data = []
        
        # 掃描資料夾內現有的文件，抓取目前所有可檢索的病理科文件
        for root, _, files in os.walk(processed_data_path):
            for file in files:
                if not file.startswith('.') and not file.startswith('~$'): # 排除隱藏檔，如 Office 產生的暫存檔（~$ 開頭）
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, processed_data_path)
                    # 取得原始檔案大小 (Bytes) 並轉換為 MB
                    raw_size_mb = os.path.getsize(abs_path) / (1024 * 1024) 
                    # 乘以 100 進行無條件進位後，再除以 100，保留到小數點後兩位
                    size_mb = math.ceil(raw_size_mb * 100) / 100
                    mtime_str = datetime.fromtimestamp(os.path.getmtime(abs_path)).strftime('%Y-%m-%d %H:%M') # 取得檔案最後修改時間(檔案上傳時間)
                    
                    file_data.append({"選取刪除": False, "檔案路徑": rel_path, "檔案大小 (MB)": size_mb, "檔案上傳時間": mtime_str})

        if file_data:
            st.write(f"目前資料庫內共有 {len(file_data)} 筆可檢索檔案：")
            df = pd.DataFrame(file_data)
            # 使用 data_editor 產生帶有核取方塊的表格
            # disabled 是用於鎖定其他欄位，避免改到文字內容
            edited_df = st.data_editor(df, column_config={"選取刪除": st.column_config.CheckboxColumn("標記刪除", default=False)}, disabled=["檔案路徑", "檔案大小 (MB)", "檔案上傳時間"], hide_index=True, width="stretch")
            
            # 過濾出被勾選要刪除的檔案
            selected_files = edited_df[edited_df["選取刪除"] == True]["檔案路徑"].tolist()
            
            if selected_files:
                st.warning(f"⚠️ 您已選取 {len(selected_files)} 個檔案。")
                if st.button("確認刪除並更新資料庫", type="primary", width="stretch"):
                    with st.spinner("正在實體刪除檔案並清理向量資料庫..."):
                        for rel_path in selected_files:
                            file_path = os.path.join(processed_data_path, rel_path)
                            try:
                                # 1. 先從 Chroma 資料庫中移除該檔案的檢索塊
                                remove_file_from_db(file_path, st.session_state.vectordb)
                                # 2. 將硬碟中的實體檔案刪除
                                os.remove(file_path)
                                active_document = st.session_state.get(
                                    "active_document"
                                )
                                if (
                                    isinstance(active_document, dict)
                                    and os.path.normcase(
                                        os.path.abspath(
                                            str(
                                                active_document.get(
                                                    "source",
                                                    "",
                                                )
                                            )
                                        )
                                    ) == os.path.normcase(
                                        os.path.abspath(file_path)
                                    )
                                ):
                                    st.session_state.active_document = None
                                st.session_state.document_candidates = []
                            except Exception as e:
                                st.error(f"刪除失敗: {rel_path}, 錯誤: {e}")
                    
                    # 刪除完成後，寫入成功訊息並重新整理畫面
                    st.session_state["show_success_toast"] = f"✅ 已成功移除 {len(selected_files)} 個檔案！"
                    st.rerun()
                    
# ==========================================
# 3. 重建資料庫：二次確認對話框
# ==========================================
# 使用 st.dialog 建立彈出式視窗，避免使用者誤觸按鈕就直接重建
@st.dialog("⚠️ 警告：強制重建資料庫")
def confirm_rebuild_dialog():
    """顯示二次確認視窗，確認後完整重建向量資料庫。

    重建會先清空現有 Chroma 索引，再重新掃描實體資料夾。由於過程
    可能耗時，畫面會傳入一個佔位元件，讓 `build_vector_db` 即時更新
    五個階段的進度。取消或完成後皆重新執行頁面以關閉對話框。
    """
    main_container = st.empty() # 建立一個空容器
    
    # 將警告文字放進這個容器中
    with main_container.container():
        st.error("您確定要清空並重建整個病理科知識庫嗎？")
        st.write("這項操作將會：")
        st.write("1. 刪除目前資料庫中的所有檢索索引。")
        st.write("2. 重新掃描硬碟中所有的檔案並重新建立索引。")
        st.write("此過程可能需要數分鐘的時間，且期間系統無法進行問答。")
        
        # 設定取消與確認按鈕
        col1, col2 = st.columns(2)
        with col1:
            cancel_btn = st.button("取消操作", use_container_width=True)
        with col2:
            confirm_btn = st.button("確認重建", type="primary", use_container_width=True)

    # 如果點擊了取消，直接重新執行網頁，對話框會自動關閉
    if cancel_btn:
        st.rerun() 

    # 如果點擊了確認重建
    if confirm_btn:
        # 清空主畫面容器
        main_container.empty()
        
        # 在清空後的畫面上，建立一個新的佔位符，用來顯示更新的進度文字
        ui_placeholder = st.empty()
        
        # 傳入佔位符，開始執行重建並顯示進度
        with st.spinner("系統正在重建向量資料庫，請稍候..."):
            st.session_state.vectordb = build_vector_db(st.session_state.vectordb, ui_placeholder)
            
        st.session_state["show_success_toast"] = "✅ 向量資料庫重建完成！"
        st.rerun()      


# ==========================================
# 4. 主要網頁介面（側邊欄與問答區）
# ==========================================
st.title("🔬 病理科檢索問答系統")
st.divider()

# --- 4-1. 側邊欄：一般使用者登入入口與管理員設定 ---
# 所有可調整參數都先保存在 Session State。一般使用者不會看到控制元件，
# 但問答流程仍會讀取管理員設定；管理員登出後，設定會在本次工作階段保留。
MODEL_OPTIONS = [
    "gemma3:4b",
    "weitsung50110/llama-3-taiwan:8b-instruct-dpo-q8_0",
    "gpt-oss:20b",
]
SYSTEM_SETTING_DEFAULTS = {
    # Agent 優先預設使用較適合工具呼叫的 gpt-oss。
    "selected_model": "gpt-oss:20b",
    "score_threshold": 0.3,
    # 避免一次放入過多文字塊，降低模型注意力分散的情況。
    "top_k_setting": 4,
    "enable_memory": True,
    "memory_rounds": 3,
}

# 將所有系統參數一次恢復成上方定義的預設值。
# 此函式會在「管理員登入成功」與「管理員登出」時執行；管理員調整
# 元件所造成的一般頁面重新執行不會呼叫，因此登入期間仍可暫時調整。
def reset_system_settings_to_defaults():
    """將模型、檢索與記憶參數全部恢復為系統預設值。"""
    for setting_name, default_value in SYSTEM_SETTING_DEFAULTS.items():
        st.session_state[setting_name] = default_value


# Streamlit 會保留舊版控制元件的 Session State。版本變更時強制重設一次，
# 確保目前已開啟的瀏覽器工作階段也會立即套用這次指定的預設值。
SYSTEM_SETTINGS_STATE_VERSION = "v5.1-defaults-20260727-02"
settings_state_needs_migration = (
    st.session_state.get("_system_settings_state_version")
    != SYSTEM_SETTINGS_STATE_VERSION
)

if settings_state_needs_migration:
    reset_system_settings_to_defaults()
    st.session_state["_system_settings_state_version"] = (
        SYSTEM_SETTINGS_STATE_VERSION
    )
else:
    # 相容未來新增的參數：只補上缺少欄位，不覆蓋管理員已調整的設定。
    for setting_name, default_value in SYSTEM_SETTING_DEFAULTS.items():
        if setting_name not in st.session_state:
            st.session_state[setting_name] = default_value

# 若日後移除某個模型選項，避免舊 Session State 讓 selectbox 發生錯誤。
if st.session_state.selected_model not in MODEL_OPTIONS:
    st.session_state.selected_model = SYSTEM_SETTING_DEFAULTS["selected_model"]

# 初始化身分狀態，預設為一般使用者。
if "role" not in st.session_state:
    st.session_state.role = "user"

with st.sidebar:
    st.header("系統狀態")

    # role 控制管理功能與參數元件是否顯示；一般問答不需要管理員權限。
    if st.session_state.role == "user":
        # === 一般使用者看到的畫面 ===
        st.info("目前身分：一般使用者")
        st.divider()

        st.subheader("系統可以幫您做什麼？")
        # 使用日常用語介紹四個 Agent Tool，不要求使用者理解程式名詞。
        with st.container(border=True):
            st.markdown(
                """
                🔍 **搜尋病理文件內容**  
                從所有院內文件中，找出和問題最相關的內容再回答。

                📌 **查詢指定文件**  
                說出文件名稱後，只搜尋該文件，避免混入其他文件內容。

                📚 **列出現有文件**  
                想知道系統有哪些資料時，可以直接請系統列出文件清單。

                🗂️ **依主題尋找文件**  
                輸入「品質管理」或「儀器操作」等主題，找出可能相關的文件。
                """
            )
        st.success(
            "您不需要選擇工具，直接用中文輸入問題，"
            "系統會自動判斷要使用哪一項功能。"
        )

        with st.expander("系統管理員登入"):
            # 使用 type="password" 隱藏輸入的文字
            admin_pwd = st.text_input("請輸入管理員解鎖碼", type="password")
            if st.button("解鎖管理員權限", use_container_width=True):
                # 密碼放在 .streamlit/secrets.toml，不直接寫死在原始碼中。
                if admin_pwd == st.secrets["admin_password"]:
                    # 每次登入都從固定預設值開始，不沿用上一次管理員調整。
                    reset_system_settings_to_defaults()
                    st.session_state.role = "admin"
                    st.rerun() # 密碼正確，重新整理畫面以顯示被隱藏的按鈕
                else:
                    st.error("密碼錯誤，請重新輸入！")
                    
    elif st.session_state.role == "admin":
        # === 管理員解鎖後看到的畫面 ===
        st.success("目前身分：系統管理員")
        
        # 檔案異動與資料庫重建具有副作用，只在管理員模式提供。
        st.subheader("管理員功能")
        if st.button("開啟檔案管理", icon="📁", use_container_width=True, key="admin_file_btn"):
            file_management_center() 
            
        if st.button("強制重建向量資料庫", type="primary", use_container_width=True, key="admin_rebuild_btn"):
            confirm_rebuild_dialog()

        st.divider()
        st.subheader("模型與檢索設定")
        st.caption(
            "每次登入管理員模式時，下方參數都會從系統預設值開始；"
            "本次登入期間可暫時調整，登出後會全部恢復預設。"
        )

        # 只有支援 tool calling 的模型，才能由 Agent 自動呼叫 RAG 工具。
        st.selectbox(
            "選擇 LLM 模型",
            options=MODEL_OPTIONS,
            key="selected_model",
            help=(
                "Agent 需要模型支援 tool calling；若目前模型無法執行工具，"
                "請改用已安裝且支援工具呼叫的模型。"
            ),
        )

        # 這兩個參數會交給 Chroma retriever，控制檢索品質與內容數量。
        st.slider(
            "檢索相似度門檻",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="score_threshold",
            help=(
                "數值越高，只採納高度相關的文字，但可能找不到資料；"
                "數值越低，會取回更多邊緣相關內容。"
            ),
        )
        st.slider(
            "檢索文本數量",
            min_value=1,
            max_value=6,
            step=1,
            key="top_k_setting",
            help="設定每次向量搜尋最多取回幾個文字塊，並非文件總數。",
        )

        st.divider()
        st.subheader("對話記憶設定")
        st.toggle(
            "啟用記憶模式",
            key="enable_memory",
            help="開啟後，Agent 會參考近期幾輪對話",
        )

        # 記憶模式關閉時仍保留目前輪數設定，但暫時鎖定滑桿。
        with st.container(border=True):
            st.slider(
                "對話記憶輪數",
                min_value=1,
                max_value=5,
                step=1,
                key="memory_rounds",
                disabled=not st.session_state.enable_memory,
                help="設定系統要參考過去幾輪對話（一問一答為一輪）。",
            )
            

        st.divider()
        # Callback 會在頁面重新執行、控制元件建立之前先執行，因此可安全
        # 重設已綁定 widget key 的 Session State。
        def logout_admin():
            """登出管理員，並立即清除本次登入期間的參數調整。"""
            reset_system_settings_to_defaults()
            st.session_state.role = "user"

        st.button(
            "登出管理員",
            use_container_width=True,
            key="admin_logout_btn",
            on_click=logout_admin,
        )

    # 接收並顯示來自其他操作（如上傳、刪除、重建）的全域成功提示訊息
    # 透過 st.session_state 傳遞訊息，顯示完畢後立刻刪除，避免重新整理網頁時又重複彈出
    if "show_success_toast" in st.session_state:
        st.toast(st.session_state["show_success_toast"])
        del st.session_state["show_success_toast"] 
    
    st.divider()
    # 清除個人聊天紀錄不是系統參數，一般使用者仍可使用。
    if st.button("清除目前對話記憶/紀錄"):
        st.session_state.messages = []
        st.session_state.active_document = None
        st.session_state.document_candidates = []
        st.rerun()

# 後續問答流程只讀取 Session State，不依賴管理員控制元件是否顯示。
selected_model = st.session_state.selected_model
score_threshold = st.session_state.score_threshold
top_k_setting = st.session_state.top_k_setting
enable_memory = st.session_state.enable_memory
memory_rounds = st.session_state.memory_rounds

# get_llm 有 st.cache_resource；只要管理員選擇的模型不變，就沿用已載入物件。
llm = get_llm(selected_model)

# --- 4-2. 初始化跨次重新執行所需的 Session State ---
# Streamlit 每次點擊、輸入或選擇都會從檔案開頭重新執行，因此需要：
# messages：完整聊天紀錄、來源與 Agent 執行紀錄。
# vectordb：Chroma 連線，避免每次互動都重新建立。
# active_document：後續問題持續限定的單一文件。
# document_candidates：部分檔名命中多版本時等待使用者選擇的文件。
if "messages" not in st.session_state:
    st.session_state.messages = [] # 初始化空陣列，用來存放歷史對話紀錄
if "vectordb" not in st.session_state:
    st.session_state.vectordb = get_vector_db() # 建立向量資料庫連線，避免每次重整都重新讀取硬碟
if "active_document" not in st.session_state:
    st.session_state.active_document = None
if "document_candidates" not in st.session_state:
    st.session_state.document_candidates = []


def is_valid_selectable_document(document):
    """確認候選文件格式正確、檔案存在，且位於知識庫資料夾內。

    這個檢查同時用於 Agent 解析結果與歷史候選清單，可防止過期路徑、
    被刪除文件或資料夾外的路徑成為限定文件。
    """
    if not isinstance(document, dict):
        return False
    source = str(document.get("source", "")).strip()
    if not source:
        return False
    # 先轉絕對路徑，再用 commonpath 確認 source 沒有跳出資料根目錄。
    source_path = os.path.abspath(source)
    data_root = os.path.abspath(processed_data_path)
    try:
        if os.path.commonpath([source_path, data_root]) != data_root:
            return False
    except ValueError:
        return False
    return os.path.isfile(source_path)


# 應用重新啟動或管理員刪除檔案後，先清除已失效的限定狀態。
if (
    st.session_state.active_document
    and not is_valid_selectable_document(st.session_state.active_document)
):
    st.session_state.active_document = None
    st.session_state.document_candidates = []
    st.warning("先前限定的文件已不存在，系統已自動解除文件限定。")


def build_direct_tool_memory_summary(
    tool_calls,
    document_count=None,
    specific_document_status=None,
):
    """把直接顯示的工具結果改寫成精簡的 Agent 短期記憶。

    文件清單可能很長，若每輪都把全部檔名送回模型，容易浪費上下文
    長度，也可能干擾下一題。因此畫面仍保存完整工具答案，但送給
    Agent 的歷史只保留查詢種類、主題、數量與限定文件狀態。
    """
    # 只有最後一次工具呼叫代表這輪直接回答的主要意圖。
    latest_tool_call = tool_calls[-1] if tool_calls else {}
    tool_name = latest_tool_call.get("name", "")

    if tool_name == "list_pathology_documents":
        count_text = (
            f"目前資料庫共有 {document_count} 份文件"
            if document_count is not None
            else "已列出目前資料庫中的所有文件"
        )
        return (
            f"上一輪是全部文件清單查詢，{count_text}。"
            "完整檔名清單已顯示給使用者，但不放入 Agent 短期記憶。"
        )

    if tool_name == "find_pathology_documents_by_topic":
        topic = str(
            latest_tool_call.get("topic")
            or latest_tool_call.get("query")
            or ""
        ).strip()
        topic_text = f"「{topic}」" if topic else "指定主題"
        count_text = (
            f"找到 {document_count} 份可能相關文件"
            if document_count is not None
            else "已顯示可能相關的文件"
        )
        return (
            f"上一輪是主題式文件清單查詢，查詢主題為 {topic_text}，"
            f"{count_text}。完整檔名清單已顯示給使用者，"
            "但不放入 Agent 短期記憶。"
        )

    if tool_name == "search_specific_document":
        document_name = str(
            latest_tool_call.get("document_name", "")
        ).strip()
        document_text = (
            f"「{document_name}」"
            if document_name
            else "指定文件"
        )
        if specific_document_status == "multiple":
            return (
                f"上一輪嘗試限定搜尋 {document_text}，"
                f"找到 {document_count or 0} 份候選文件，"
                "正在等待使用者選擇版本；完整候選清單不放入 "
                "Agent 短期記憶。"
            )
        if specific_document_status == "no_content":
            return (
                f"上一輪已限定搜尋 {document_text}，"
                "但沒有找到符合問題的文件片段。"
            )
        return (
            f"上一輪嘗試限定搜尋 {document_text}，"
            "但文件名稱無效或沒有符合的已匯入文件。"
        )

    return (
        "上一輪是文件清單查詢，完整檔名清單已顯示給使用者，"
        "但不放入 Agent 短期記憶。"
    )


def build_direct_tool_user_notice(tool_calls, document_count=None):
    """在文件清單下方建立下一步操作提示。

    工具內容本身保持原文直接顯示；此函式只補充「接下來可以輸入
    文件名稱或更具體問題」等介面說明，不修改工具結果。
    """
    latest_tool_call = tool_calls[-1] if tool_calls else {}
    tool_name = latest_tool_call.get("name", "")

    if tool_name == "find_pathology_documents_by_topic":
        topic = str(
            latest_tool_call.get("topic")
            or latest_tool_call.get("query")
            or ""
        ).strip()
        topic_text = f"「{topic}」" if topic else "指定主題"
        if document_count == 0:
            return (
                f"目前未找到與{topic_text}明確相關的文件。"
                "請嘗試使用其他關鍵字，或輸入更具體的問題。"
            )
        return (
            f"以上為系統依{topic_text}找到的可能相關文件。"
            "如需查詢特定文件內容，請輸入文件名稱或更具體的問題。"
        )

    if tool_name == "list_pathology_documents":
        return (
            "以上為目前已匯入系統的文件清單。"
            "如需查詢特定文件內容，請輸入文件名稱或具體問題。"
        )

    if tool_name == "search_specific_document":
        return ""

    return (
        "以上為系統找到的文件清單。"
        "如需查詢特定文件內容，請輸入文件名稱或更具體的問題。"
    )


def get_agent_memory_content(message):
    """取得要放入 Agent 對話歷史的文字。

    優先順序：
        1. 若已有 agent_memory_content，使用預先建立的精簡版。
        2. 若是舊版直接工具回答，現場建立相容的精簡摘要。
        3. 一般訊息直接使用畫面上的完整 content。

    畫面顯示不受影響，仍使用 `message["content"]`。
    """
    if message.get("agent_memory_content"):
        return str(message["agent_memory_content"])

    # 相容修改前已存在於 Session State 的文件清單型歷史紀錄。
    if message.get("direct_tool_answer"):
        return build_direct_tool_memory_summary(
            tool_calls=message.get("tool_calls", []),
            document_count=message.get("document_count"),
            specific_document_status=message.get(
                "specific_document_status"
            ),
        )

    return str(message.get("content", ""))


def render_document_candidate_selector(message, message_index):
    """顯示多版本候選選單，並把使用者選擇設為目前限定文件。

    當部分檔名同時命中多份文件時，Agent 不自行猜測版本，而是將
    候選清單保存在訊息中。使用者確認後：
        - 更新 active_document。
        - 清空暫存候選。
        - 更新該歷史訊息的短期記憶內容。
        - 重新執行頁面，顯示目前限定文件控制列。
    """
    candidates = message.get("document_candidates", [])
    if not candidates:
        return

    selected_document_name = message.get("selected_document_name")
    if selected_document_name:
        st.success(
            f"已限定文件：{selected_document_name}。"
            "請重新輸入要查詢的內容。"
        )
        return

    # 歷史紀錄中的候選文件可能已被管理員刪除，顯示前重新驗證。
    valid_candidates = [
        candidate
        for candidate in candidates
        if is_valid_selectable_document(candidate)
    ]
    if not valid_candidates:
        st.warning("候選文件已不存在，請重新查詢文件名稱。")
        return

    # 每輪問答使用獨立元件 key，避免歷史訊息中的選單彼此衝突。
    execution_key = (
        message.get("execution_id")
        or f"message-{message_index}"
    )
    option_key = f"document_candidate_option_{execution_key}"
    button_key = f"document_candidate_confirm_{execution_key}"
    candidate_names = [
        candidate["display_name"]
        for candidate in valid_candidates
    ]
    selected_name = st.selectbox(
        "請選擇要限定查詢的文件版本",
        options=candidate_names,
        index=None,
        placeholder="請選擇文件",
        key=option_key,
    )
    if st.button(
        "確認選擇",
        type="primary",
        disabled=selected_name is None,
        key=button_key,
    ):
        # selected_name 一定來自 valid_candidates 的 options，可安全找出原資料。
        selected_document = next(
            candidate
            for candidate in valid_candidates
            if candidate["display_name"] == selected_name
        )
        st.session_state.active_document = selected_document
        st.session_state.document_candidates = []
        message["selected_document_name"] = selected_name
        message["agent_memory_content"] = (
            f"使用者已從多個候選版本中選定「{selected_name}」"
            "作為目前限定文件，正在等待新的查詢問題。"
        )
        message["specific_document_status"] = "selected"
        st.rerun()


# --- 4-3. 依 Session State 重畫歷史對話、來源及候選選單 ---
for message_index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # 顯示已完成回答的 Agent 執行紀錄；agent_steps 用於相容 v2 舊紀錄。
        saved_execution_log = message.get("execution_log") or message.get("agent_steps")
        if saved_execution_log:
            render_agent_execution_log(saved_execution_log, expanded=False)

        st.markdown(message["content"]) # 顯示對話文字內容

        if message.get("direct_tool_answer"):
            direct_tool_notice = build_direct_tool_user_notice(
                tool_calls=message.get("tool_calls", []),
                document_count=message.get("document_count"),
            )
            if direct_tool_notice:
                st.info(direct_tool_notice)
        
        # 檢查該筆歷史紀錄是否有檢索來源資料 (docs)
        if (
            not message.get("direct_tool_answer")
            and "docs" in message
            and message["docs"]
        ):
            st.markdown("#### 🔍 檢索來源片段：")
            for i, doc in enumerate(message["docs"]):
                source_name = os.path.basename(doc.metadata.get('source', '未知'))
                page_num = doc.metadata.get('page')
                
                # 如果是 PDF 通常會有頁碼資訊，一併顯示出來
                if page_num is not None:
                    header_text = f"來源 {i+1}: {source_name} (第 {page_num + 1} 頁)"
                else:
                    header_text = f"來源 {i+1}: {source_name}"
                
                # 建立可摺疊的面板，讓版面保持乾淨，不會被長篇的醫療文本塞滿
                with st.expander(header_text):
                    st.markdown(f'<div style="font-size: 0.85em; color: #505050;">{doc.page_content}</div>', unsafe_allow_html=True)

        render_document_candidate_selector(message, message_index)
        
        
        


# 限定文件狀態放在聊天輸入框上方，並提供使用者主動取消的按鈕。
if st.session_state.active_document:
    with st.container(border=True):
        active_document_column, clear_document_column = st.columns(
            [5, 1]
        )
        active_document_column.markdown(
            "📌 **目前限定文件：** "
            f"{st.session_state.active_document['display_name']}"
        )
        if clear_document_column.button(
            "取消限定",
            key="clear_active_document",
            use_container_width=True,
        ):
            st.session_state.active_document = None
            st.session_state.document_candidates = []
            st.rerun()


# ==========================================
# 5. Agent 問答與串流流程
# ==========================================
# `:=` 會同時取得輸入值並判斷是否有送出；沒有新問題時此區不執行。
if prompt_input := st.chat_input("請輸入關於病理科流程的問題..."):
    # 先保存使用者訊息，確保稍後組合 Agent 記憶時包含目前問題。
    st.session_state.messages.append({
        "role": "user",
        "content": prompt_input,
    })

    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        # 沒有任何可檢索文件時，提早中止，避免 Agent 在無來源的情況下
        # 依模型自身知識回答院內流程。
        valid_files = [
            file_name
            for _, _, file_names in os.walk(processed_data_path)
            for file_name in file_names
            if not file_name.startswith((".", "~"))
        ]
        if not valid_files:
            fallback_msg = "目前資料庫中沒有任何文件，請先點擊左側「開啟檔案管理」上傳病理科檔案。"
            st.error(fallback_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": fallback_msg,
                "docs": [],
            })
            st.stop()

        # 一輪等於一則 user 加一則 assistant；額外的 +1 是目前新問題。
        # 關閉記憶時只送出最後一則使用者訊息。
        if enable_memory:
            max_messages = memory_rounds * 2 + 1  # 近期 N 輪 + 目前問題
            history_for_agent = st.session_state.messages[-max_messages:]
        else:
            history_for_agent = [st.session_state.messages[-1]]

        # 將 Streamlit 自訂訊息格式轉成 Agent 接受的 role/content 格式。
        # 文件清單型回答會由 get_agent_memory_content 自動改用精簡摘要。
        agent_messages = [
            {
                "role": "user" if message["role"] == "user" else "assistant",
                "content": get_agent_memory_content(message),
            }
            for message in history_for_agent
        ]

        # 先建立預設值，即使 create_pathology_agent 或串流途中出錯，
        # except 區塊仍可安全取得空結果並保存錯誤紀錄。
        search_state = {
            "docs": [],
            "tool_calls": [],
            "document_candidates": [],
            "resolved_document": None,
            "specific_document_status": None,
        }
        # 本輪回答使用獨立紀錄與識別碼，重新整理後仍能對應歷史元件。
        execution_log = []
        run_started_at = time.perf_counter()
        execution_id = datetime.now().strftime("AGENT-%Y%m%d-%H%M%S-%f")
        live_status = None
        try:
            # 每一題建立新的 Agent 與 search_state，避免上一題的工具結果
            # 殘留；模型與 vectordb 本身則由快取或 Session State 共用。
            agent, search_state = create_pathology_agent(
                chat_model=llm,
                vectordb=st.session_state.vectordb,
                score_threshold=score_threshold,
                top_k_setting=top_k_setting,
                active_document=st.session_state.active_document,
            )

            # st.status 顯示安全的執行階段；實際回答文字放在另一個佔位元件。
            live_status = st.status(
                "Agent 正在分析問題並選擇適合的工具...",
                expanded=True,
            )
            answer_placeholder = st.empty()
            # LangGraph 串流中同一事件可能在不同模式出現，使用 set 去重。
            seen_event_keys = set()
            # detected_tool_calls 從 model update 取得，作為 search_state 尚未
            # 寫入時的備援工具紀錄。
            detected_tool_calls = []
            # 模型文字會逐段加入此清單，再合併顯示。
            streamed_answer_parts = []
            final_answer = ""
            direct_tool_answer = ""
            # 這兩種工具已產生可直接呈現的檔名清單，無須再讓 LLM 改寫。
            direct_answer_tool_names = {
                "list_pathology_documents",
                "find_pathology_documents_by_topic",
            }
            stop_after_direct_tool_answer = False
            first_text_received = False

            # 第一筆固定事件代表主程式已正式開始處理本輪問題。
            emit_live_agent_event(
                status_container=live_status,
                execution_log=execution_log,
                started_at=run_started_at,
                seen_event_keys=seen_event_keys,
                event_key="question_received",
                title="接收問題",
                detail="已接收使用者問題，正在判斷問題類型與工具需求。",
                state="running",
                event_type="input",
                execution_id=execution_id,
            )

            # 同時訂閱兩種 LangGraph 串流：
            # messages：取得模型逐字／逐段文字。
            # updates：取得模型節點、工具呼叫與工具結果等完整狀態。
            agent_stream = agent.stream(
                {"messages": agent_messages},
                stream_mode=["messages", "updates"],
                version="v2",
            )
            for stream_chunk in agent_stream:
                # v2 串流事件以 type 區分資料種類，實際內容放在 data。
                chunk_type = stream_chunk.get("type")
                chunk_data = stream_chunk.get("data")

                # ---------- A. messages：把模型回答逐段顯示在畫面 ----------
                # 只接受 model 節點的 AIMessageChunk，不顯示工具內容或
                # reasoning 等非最終回答區塊。
                if chunk_type == "messages":
                    if not isinstance(chunk_data, tuple) or len(chunk_data) != 2:
                        continue

                    token, metadata = chunk_data
                    if not isinstance(token, AIMessageChunk):
                        continue
                    if (
                        not isinstance(metadata, dict)
                        or metadata.get("langgraph_node") != "model"
                    ):
                        continue

                    # 不同 ChatOllama / LangChain 版本可能把文字放在 text
                    # 或 content，因此先讀 text，沒有再用共用函式轉換。
                    token_text = getattr(token, "text", "")
                    if not isinstance(token_text, str):
                        token_text = ""
                    if not token_text:
                        token_text = _content_to_text(
                            getattr(token, "content", "")
                        )
                    if not token_text:
                        continue

                    # 工具結果已決定直接顯示時，忽略可能晚到的模型文字。
                    if direct_tool_answer:
                        continue

                    if not first_text_received:
                        first_text_received = True
                        emit_live_agent_event(
                            status_container=live_status,
                            execution_log=execution_log,
                            started_at=run_started_at,
                            seen_event_keys=seen_event_keys,
                            event_key="answer_stream_started",
                            title="開始串流生成回答",
                            detail="已收到第一段回答文字，正在逐段輸出最終回答。",
                            state="running",
                            event_type="generation",
                        )

                    # 尾端方塊符號表示仍在串流；完成後會由正式答案取代。
                    streamed_answer_parts.append(token_text)
                    answer_placeholder.markdown(
                        "".join(streamed_answer_parts) + "▌"
                    )
                    continue

                # ---------- B. updates：處理模型決策、工具呼叫及完成結果 ----------
                if chunk_type != "updates" or not isinstance(chunk_data, dict):
                    continue

                for node_name, node_update in chunk_data.items():
                    if not isinstance(node_update, dict):
                        continue

                    # 一個節點更新可能帶有多則訊息；model 判斷通常取最後一則，
                    # tools 則會逐則處理，以支援一次呼叫多個工具。
                    node_messages = node_update.get("messages", [])
                    if not node_messages:
                        continue
                    latest_message = node_messages[-1]

                    if node_name == "model":
                        # 有 tool_calls 表示模型尚未產生最終答案，而是決定
                        # 先執行一個或多個工具。
                        tool_calls = getattr(latest_message, "tool_calls", []) or []
                        if tool_calls:
                            for tool_call in tool_calls:
                                # LangChain 將工具參數放在 args；不同工具可能
                                # 使用 query、topic 或 document_name。
                                tool_name = tool_call.get("name")
                                tool_args = tool_call.get("args") or {}
                                tool_call_id = tool_call.get("id") or (
                                    f"{tool_name}-{len(detected_tool_calls) + 1}"
                                )
                                if isinstance(tool_args, dict):
                                    search_query = str(
                                        tool_args.get("query")
                                        or tool_args.get("topic")
                                        or ""
                                    ).strip()
                                    requested_document_name = str(
                                        tool_args.get("document_name")
                                        or ""
                                    ).strip()
                                else:
                                    search_query = str(tool_args).strip()
                                    requested_document_name = ""

                                # 保存標準化後的工具呼叫，供日後顯示與備援。
                                detected_tool_calls.append({
                                    "name": tool_name,
                                    "query": search_query,
                                    "document_name": requested_document_name,
                                    "id": tool_call_id,
                                })

                                # 依工具種類產生一般使用者看得懂的問題分類說明。
                                if tool_name == "search_pathology_documents":
                                    decision_detail = (
                                        "判斷為病理科文件內容問題，需要搜尋知識庫。"
                                    )
                                elif tool_name == "list_pathology_documents":
                                    decision_detail = (
                                        "判斷為知識庫文件清單問題，需要讀取文件清單。"
                                    )
                                elif tool_name == "find_pathology_documents_by_topic":
                                    decision_detail = (
                                        "判斷為主題式文件清單問題，需要依主題搜尋相關文件。"
                                    )
                                elif tool_name == "search_specific_document":
                                    decision_detail = (
                                        "判斷為特定文件內容問題，需要限定文件搜尋。"
                                    )
                                else:
                                    decision_detail = (
                                        f"判斷需要呼叫工具：{get_tool_display_name(tool_name)}。"
                                    )

                                # event_key 固定，代表本輪只需要一筆分類完成事件。
                                emit_live_agent_event(
                                    status_container=live_status,
                                    execution_log=execution_log,
                                    started_at=run_started_at,
                                    seen_event_keys=seen_event_keys,
                                    event_key="question_classified",
                                    title="完成問題類型判斷",
                                    detail=decision_detail,
                                    state="complete",
                                    event_type="decision",
                                    tool_name=tool_name,
                                )

                                # 工具名稱、搜尋句與限定文件都屬於可安全公開資訊。
                                tool_detail = (
                                    f"準備呼叫 {get_tool_display_name(tool_name)}"
                                )
                                if search_query:
                                    tool_detail += f"；搜尋問題：{search_query}"
                                if requested_document_name:
                                    tool_detail += (
                                        f"；限定文件：{requested_document_name}"
                                    )
                                tool_detail += "。"

                                emit_live_agent_event(
                                    status_container=live_status,
                                    execution_log=execution_log,
                                    started_at=run_started_at,
                                    seen_event_keys=seen_event_keys,
                                    event_key=f"tool_call:{tool_call_id}",
                                    title="呼叫文件工具",
                                    detail=tool_detail,
                                    state="running",
                                    event_type="tool_call",
                                    tool_name=tool_name,
                                    query=search_query,
                                    tool_call_id=tool_call_id,
                                )
                        else:
                            # model 節點沒有工具呼叫時，通常代表這是 Agent 的
                            # 最終回答；保留完整內容作為串流文字的備援。
                            completed_text = _content_to_text(
                                getattr(latest_message, "content", "")
                            ).strip()
                            if completed_text and not direct_tool_answer:
                                final_answer = completed_text
                                if not detected_tool_calls:
                                    emit_live_agent_event(
                                        status_container=live_status,
                                        execution_log=execution_log,
                                        started_at=run_started_at,
                                        seen_event_keys=seen_event_keys,
                                        event_key="question_classified",
                                        title="完成問題類型判斷",
                                        detail=(
                                            "判斷為不需要文件工具的一般對話，"
                                            "由 Agent 直接回答。"
                                        ),
                                        state="complete",
                                        event_type="decision",
                                    )

                    elif node_name == "tools":
                        # 工具節點可能一次完成多個 ToolMessage，因此逐筆處理。
                        for completed_message in node_messages:
                            completed_tool_name = getattr(
                                completed_message, "name", None
                            )
                            completed_tool_call_id = getattr(
                                completed_message, "tool_call_id", None
                            )
                            # 某些版本只提供 tool_call_id，需回查先前模型事件
                            # 才能知道是哪個工具。
                            if not completed_tool_name and completed_tool_call_id:
                                matching_tool_call = next(
                                    (
                                        tool_call
                                        for tool_call in reversed(detected_tool_calls)
                                        if tool_call.get("id") == completed_tool_call_id
                                    ),
                                    None,
                                )
                                if matching_tool_call:
                                    completed_tool_name = matching_tool_call.get("name")
                            # 若串流訊息仍缺名稱，最後從工具實際寫入的
                            # search_state 紀錄推回名稱。
                            if (
                                not completed_tool_name
                                and search_state.get("tool_calls")
                            ):
                                completed_tool_name = search_state["tool_calls"][-1].get(
                                    "name"
                                )

                            # ToolMessage.content 是可供 Agent 或畫面使用的結果文字。
                            completed_tool_content = _content_to_text(
                                getattr(completed_message, "content", "")
                            ).strip()

                            # 依工具類型讀取 search_state，產生結果數量與狀態說明。
                            if completed_tool_name == "search_pathology_documents":
                                doc_count = len(search_state.get("docs", []))
                                if doc_count:
                                    result_detail = (
                                        f"文件搜尋完成，共找到 {doc_count} "
                                        "筆符合門檻的相關文件片段。"
                                    )
                                    result_state = "complete"
                                else:
                                    result_detail = (
                                        "文件搜尋完成，但沒有找到符合門檻的"
                                        "相關文件片段。"
                                    )
                                    result_state = "warning"
                            elif completed_tool_name == "list_pathology_documents":
                                doc_count = len(valid_files)
                                result_detail = (
                                    f"文件清單讀取完成，目前共有 {doc_count} 份文件。"
                                )
                                result_state = "complete"
                            elif completed_tool_name == "find_pathology_documents_by_topic":
                                doc_count = len(
                                    search_state.get("topic_document_names", [])
                                )
                                topic = ""
                                if search_state.get("tool_calls"):
                                    topic = search_state["tool_calls"][-1].get(
                                        "topic",
                                        "",
                                    )
                                if doc_count:
                                    result_detail = (
                                        f"主題式文件清單搜尋完成，共找到 {doc_count} "
                                        "份可能相關的文件。"
                                    )
                                    result_state = "complete"
                                else:
                                    topic_text = (
                                        f"「{topic}」"
                                        if topic
                                        else "此主題"
                                    )
                                    result_detail = (
                                        f"主題式文件清單搜尋完成，但沒有找到與 {topic_text} "
                                        "明確相關的文件。"
                                    )
                                    result_state = "warning"
                            elif completed_tool_name == "search_specific_document":
                                specific_status = search_state.get(
                                    "specific_document_status"
                                )
                                if specific_status == "multiple":
                                    doc_count = len(
                                        search_state.get(
                                            "document_candidates",
                                            [],
                                        )
                                    )
                                    result_detail = (
                                        f"找到 {doc_count} 份名稱相符的文件，"
                                        "需要使用者選擇要限定的版本。"
                                    )
                                    result_state = "warning"
                                elif specific_status == "resolved":
                                    doc_count = len(
                                        search_state.get("docs", [])
                                    )
                                    resolved_name = (
                                        search_state.get(
                                            "resolved_document"
                                        )
                                        or {}
                                    ).get("display_name", "指定文件")
                                    result_detail = (
                                        f"已限定「{resolved_name}」搜尋，"
                                        f"共找到 {doc_count} 筆相關文件片段。"
                                    )
                                    result_state = "complete"
                                elif specific_status == "no_content":
                                    doc_count = 0
                                    result_detail = (
                                        "已限定指定文件，但沒有找到符合門檻的"
                                        "相關文件片段。"
                                    )
                                    result_state = "warning"
                                else:
                                    doc_count = 0
                                    result_detail = (
                                        "限定文件解析未完成，請確認文件名稱。"
                                    )
                                    result_state = "warning"
                            else:
                                doc_count = None
                                result_detail = (
                                    f"{get_tool_display_name(completed_tool_name)} "
                                    "已執行完成。"
                                )
                                result_state = "complete"

                            # 優先用 LangChain 提供的 call id 形成唯一事件鍵。
                            event_suffix = (
                                completed_tool_call_id
                                or f"{completed_tool_name}-{len(execution_log)}"
                            )
                            emit_live_agent_event(
                                status_container=live_status,
                                execution_log=execution_log,
                                started_at=run_started_at,
                                seen_event_keys=seen_event_keys,
                                event_key=f"tool_result:{event_suffix}",
                                title="取得工具執行結果",
                                detail=result_detail,
                                state=result_state,
                                event_type="tool_result",
                                tool_name=completed_tool_name,
                                document_count=doc_count,
                                tool_call_id=completed_tool_call_id,
                            )

                            # 限定文件若無法進入正常回答（多候選、無檔案、
                            # 參數無效或無內容），工具文字就是最清楚的最終回覆。
                            specific_direct_statuses = {
                                "multiple",
                                "not_found",
                                "invalid",
                                "no_content",
                            }
                            # 文件清單不需要 LLM 摘要；直接顯示可避免翻譯、
                            # 漏列或自行推測檔案內容。
                            should_display_directly = (
                                completed_tool_name in direct_answer_tool_names
                                or (
                                    completed_tool_name
                                    == "search_specific_document"
                                    and search_state.get(
                                        "specific_document_status"
                                    ) in specific_direct_statuses
                                )
                            )
                            if should_display_directly and completed_tool_content:
                                # 清除先前可能串流出的模型片段，以工具原始結果為準。
                                direct_tool_answer = completed_tool_content
                                stop_after_direct_tool_answer = True
                                final_answer = completed_tool_content
                                streamed_answer_parts = []
                                answer_placeholder.markdown(direct_tool_answer)
                                emit_live_agent_event(
                                    status_container=live_status,
                                    execution_log=execution_log,
                                    started_at=run_started_at,
                                    seen_event_keys=seen_event_keys,
                                    event_key=f"direct_tool_answer:{event_suffix}",
                                    title="直接顯示工具結果",
                                    detail=(
                                        "工具結果已可直接回覆使用者，"
                                        "不再交由模型重新改寫。"
                                    ),
                                    state="complete",
                                    event_type="direct_answer",
                                    tool_name=completed_tool_name,
                                    document_count=doc_count,
                                    tool_call_id=completed_tool_call_id,
                                )
                                # 工具結果已是最終答案，停止處理本次節點。
                                break

                    # 跳出 node_update 迴圈，避免直接答案後仍處理其他節點。
                    if stop_after_direct_tool_answer:
                        break

                # 再跳出最外層 Agent 串流迴圈。
                if stop_after_direct_tool_answer:
                    break

            # 清單型 Tool 已提供最終答案，不再讓 Agent 進入第二次 LLM 推論。
            if stop_after_direct_tool_answer:
                # generator 若支援 close，主動關閉可減少不必要的後續運算。
                close_agent_stream = getattr(agent_stream, "close", None)
                if callable(close_agent_stream):
                    close_agent_stream()

            # 最終答案優先順序：直接工具結果 → model 完整訊息 →
            # 已接收的文字串流。這讓不同模型／LangChain 版本都能正常顯示。
            ans = (
                direct_tool_answer
                or final_answer
                or "".join(streamed_answer_parts).strip()
            )
            if not ans:
                ans = (
                    "抱歉，Agent 沒有產生可顯示的回答，"
                    "請確認 Ollama 模型是否支援工具呼叫與串流輸出。"
                )

            # 用不含游標符號的正式答案覆蓋串流中的暫時畫面。
            answer_placeholder.markdown(ans)
            total_seconds = round(time.perf_counter() - run_started_at, 2)
            emit_live_agent_event(
                status_container=live_status,
                execution_log=execution_log,
                started_at=run_started_at,
                seen_event_keys=seen_event_keys,
                event_key="answer_completed",
                title="回答生成完成",
                detail=f"最終回答已完整輸出，總執行時間 {total_seconds:.2f} 秒。",
                state="complete",
                event_type="complete",
                total_seconds=total_seconds,
            )
            live_status.update(
                label=f"✅ Agent 已完成回答（{total_seconds:.2f} 秒）",
                state="complete",
                expanded=False,
            )

            # 工具本身寫入的紀錄最接近實際執行結果；若尚未寫入，
            # 則使用從 model 串流偵測到的工具呼叫。
            completed_tool_calls = (
                search_state.get("tool_calls", [])
                or detected_tool_calls
            )
            # 限定文件搜尋成功解析唯一文件後，將它保存到 Session State，
            # 使後續追問可持續在同一份文件內檢索。
            resolved_document = search_state.get("resolved_document")
            active_document_changed = False
            if is_valid_selectable_document(resolved_document):
                previous_active_document = st.session_state.active_document
                previous_active_source = ""
                if isinstance(previous_active_document, dict):
                    previous_active_source = str(
                        previous_active_document.get("source", "")
                    )
                resolved_source = str(
                    resolved_document.get("source", "")
                )
                # 比較正規化後的絕對路徑，判斷是否需要重新執行頁面。
                active_document_changed = (
                    os.path.normcase(
                        os.path.abspath(previous_active_source)
                    )
                    != os.path.normcase(
                        os.path.abspath(resolved_source)
                    )
                )
                st.session_state.active_document = resolved_document

            # 多版本候選需保存到本輪訊息，重新執行後選單仍會存在。
            document_candidates = search_state.get(
                "document_candidates",
                [],
            )
            if document_candidates:
                st.session_state.document_candidates = document_candidates
            elif search_state.get("specific_document_status") is not None:
                st.session_state.document_candidates = []

            # 直接工具答案會另外保存結果數量，供提示文字與短期記憶使用。
            direct_tool_document_count = None
            if direct_tool_answer and completed_tool_calls:
                latest_tool_name = completed_tool_calls[-1].get("name")
                if latest_tool_name == "list_pathology_documents":
                    direct_tool_document_count = len(valid_files)
                elif latest_tool_name == "find_pathology_documents_by_topic":
                    direct_tool_document_count = len(
                        search_state.get("topic_document_names", [])
                    )
                elif latest_tool_name == "search_specific_document":
                    direct_tool_document_count = len(document_candidates)

            # 清單型答案顯示下一步提示；內容型答案則顯示實際來源片段。
            if direct_tool_answer:
                direct_tool_notice = build_direct_tool_user_notice(
                    tool_calls=completed_tool_calls,
                    document_count=direct_tool_document_count,
                )
                if direct_tool_notice:
                    st.info(direct_tool_notice)
            else:
                render_source_documents(search_state.get("docs", []))

            if is_valid_selectable_document(resolved_document):
                st.success(
                    "目前已限定文件："
                    f"{resolved_document['display_name']}。"
                    "後續文件內容問題會持續限定此文件。"
                )

            # 畫面與歷史保存完整答案，但超長文件清單只送精簡摘要給
            # 下一輪 Agent，避免占滿有限的上下文長度。
            agent_memory_content = ans
            if direct_tool_answer:
                agent_memory_content = build_direct_tool_memory_summary(
                    tool_calls=completed_tool_calls,
                    document_count=direct_tool_document_count,
                    specific_document_status=search_state.get(
                        "specific_document_status"
                    ),
                )

            # 一筆 assistant 歷史不只保存文字，也保存重畫畫面與延續
            # 文件限定所需的完整結構化資料。
            assistant_message = {
                "role": "assistant",
                "content": ans,
                "agent_memory_content": agent_memory_content,
                "docs": search_state.get("docs", []),
                "direct_tool_answer": bool(direct_tool_answer),
                "document_count": direct_tool_document_count,
                "execution_log": execution_log,
                "execution_id": execution_id,
                "tool_calls": completed_tool_calls,
                "document_candidates": document_candidates,
                "specific_document_status": search_state.get(
                    "specific_document_status"
                ),
                "active_document_name": (
                    st.session_state.active_document.get("display_name")
                    if isinstance(
                        st.session_state.active_document,
                        dict,
                    )
                    else None
                ),
            }
            st.session_state.messages.append(assistant_message)
            if document_candidates:
                # 本輪立即顯示選單；日後頁面重跑則由歷史迴圈再次顯示。
                render_document_candidate_selector(
                    assistant_message,
                    len(st.session_state.messages) - 1,
                )
            elif active_document_changed:
                # 回答與歷史已保存，重新執行後立即顯示限定文件控制列。
                st.rerun()

        except Exception as e:
            # 常見原因包括 Ollama 未啟動、模型不存在、模型不支援工具呼叫，
            # 或 LangChain 串流介面版本不相容。對使用者顯示簡短訊息，
            # 詳細例外則放入折疊區供維護人員查看。
            error_msg = (
                "⚠️ Agent 執行失敗。請確認 Ollama 服務已啟動、模型名稱正確，"
                "且目前模型支援 tool calling 與串流輸出。"
            )
            if live_status is not None:
                emit_live_agent_event(
                    status_container=live_status,
                    execution_log=execution_log,
                    started_at=run_started_at,
                    seen_event_keys=locals().get("seen_event_keys", set()),
                    event_key="agent_error",
                    title="Agent 執行失敗",
                    detail=str(e),
                    state="error",
                    event_type="error",
                )
                live_status.update(
                    label="❌ Agent 執行失敗",
                    state="error",
                    expanded=True,
                )
            st.error(error_msg)
            with st.expander("查看錯誤細節"):
                st.exception(e)
            # 即使失敗也保存訊息與執行紀錄，避免頁面重跑後錯誤消失。
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "docs": search_state.get("docs", []),
                "execution_log": execution_log,
                "execution_id": execution_id,
            })

# ==========================================
# 6. 固定於頁尾的使用提醒
# ==========================================
# 使用自訂 CSS 固定位置；unsafe_allow_html=True 是套用此 HTML/CSS 所必需。
st.markdown(
    """
    <style>
    .disclaimer-text {
        position: fixed;
        bottom: 5px; /* 距離網頁最底部的距離 */
        left: 0;
        right: 0;
        text-align: center;
        font-size: 12px;
        color: #888888;
        background-color: transparent;
        z-index: 999; /* 確保不會被其他元件遮擋 */
    }
    
    </style>
    <div class="disclaimer-text">
        ⚕️ 病理科助手為 AI 輔助系統，有時可能產生錯誤資訊。結果僅供參考，請務必與專業醫療規範確認。
    </div>
    """,
    unsafe_allow_html=True
)           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
