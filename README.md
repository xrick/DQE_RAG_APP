# DQE RAG 應用程式

## 專案概述

DQE (Design Quality Estimate) RAG 應用程式是一個檢索增強生成（Retrieval-Augmented Generation, RAG）系統，旨在利用大型語言模型（LLM）的強大能力，提供一個智慧查詢解決方案，協助使用者從過去的專案問題與解決方案的知識庫中，快速找到相關的經驗與解答。

此應用程式透過結合內部結構化知識庫與外部網路搜尋，為使用者提供全面且精準的回答，特別適用於工程、設計與品質管理領域，有助於提升問題解決的效率與品質。

## 主要功能

* **混合式檢索**：整合內部 **Milvus** 向量資料庫與外部 **Google Serper** 網路搜尋，提供全面性的資料檢索。
* **智慧問答**：採用 **Ollama** 驅動的大型語言模型，能夠理解使用者問題，並生成結合內外部資料的精確回答。
* **即時串流回應**：透過 **FastAPI** 的 `StreamingResponse`，將處理過程與結果分步即時回傳至前端，提升使用者體驗。
* **結構化資料呈現**：能將從資料庫檢索到的結構化資料，動態生成易於閱讀的 Markdown 表格。
* **非同步處理**：應用程式核心邏輯採用非同步（async/await）設計，確保在高負載下的高效能與高並行處理能力。
* **彈性化搜尋**：支援精確搜尋與模糊搜尋，並可自訂搜尋相似度閾值，滿足不同場景下的查詢需求。

## 系統架構

本專案採用前後端分離架構，後端由 **FastAPI** 驅動，負責核心 RAG 流程。

```mermaid
graph TD
    subgraph Frontend
        A[使用者介面]
    end

    subgraph Backend (FastAPI)
        B[API 端點 /api/ai-chat-stream]
        C{核心處理流程}
        D[內部檢索 Milvus]
        E[外部檢索 Google Serper]
        F[LLM 生成 Ollama]
    end

    A -- HTTP 請求 --> B
    B --> C
    C -- 內部查詢 --> D
    C -- 外部搜尋 --> E
    subgraph RAG Pipeline
        D -- 內部資料 --> F
        E -- 外部資料 --> F
    end
    F -- 綜合回答 --> C
    C -- StreamingResponse --> A
```

## 專案結構

```
notebook_rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── security.py
│   │   ├── api/
│   │   │   ├── endpoints/
│   │   │   └── dependencies.py
│   │   └── services/
│   │       ├── rag_service.py
│   │       └── external_apis/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   └── services/
└── docker-compose.yml
```

## 使用的技術

* **後端框架**: FastAPI
* **向量資料庫**: Milvus
* **大型語言模型 (LLM)**: Ollama
* **外部搜尋 API**: Google Serper
* **核心函式庫**: LangChain, PyTableWriter
* **環境管理**: python-dotenv

## 安裝與設定

1.  **複製專案**
    ```bash
    git clone https://github.com/your-repo/DQE_RAG_APP.git
    cd DQE_RAG_APP
    ```

2.  **安裝依賴套件**
    從 `web_develop_milvus/` 目錄中找到 `requirements.txt` 檔案，並執行以下指令安裝所有必要的 Python 套件：
    ```bash
    pip install -r web_develop_milvus/requirements.txt
    ```

3.  **設定環境變數**
    在專案根目錄下建立一個 `.env` 檔案，並根據 `web_develop_milvus/main_milvus.py` 中的 `load_dotenv()` 部分，設定必要的環境變數，例如：
    ```
    EMBEDDEDING_MODEL="your-embedding-model-name"
    EMBEDDEDING_MODEL_PATH="/path/to/your/embedding/model"
    SERPER_API_KEY="your-serper-api-key"
    ```

## 啟動應用程式

使用 `uvicorn` 來啟動 FastAPI 應用程式：

```bash
uvicorn web_develop_milvus.main_milvus:app --host 0.0.0.0 --port 8000 --reload
```
* `--reload` 參數可以在程式碼變動時自動重啟伺服器，方便開發。

## API 端點

### `POST /api/ai-chat-stream`

此為主要的聊天 API 端點，接收使用者問題並以串流方式回傳結果。

**請求格式**:
```json
{
  "message": "請描述一下關於XXX的問題",
  "search_action": 3,
  "search_threshold": 0.3
}
```

* `message` (str): 使用者的問題。
* `search_action` (int):
    * `1`: 僅進行內部精確搜尋。
    * `3`: 進行內部搜尋，並整合外部網路搜尋結果。
* `search_threshold` (float): 內部搜尋的相似度閾值，範圍為 0 到 1。

**回應格式**:
回應為 Newline Delimited JSON (NDJSON) 格式的串流，前端可即時解析並呈現各階段結果：

* 內部搜尋結果 (`internal_results`)
* 處理狀態更新 (`status`)
* 最終結合生成的答案 (`final_answer_chunk`)
* 處理完成信號 (`done`)
* 錯誤訊息 (`error`)
