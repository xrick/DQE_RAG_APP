import os
from contextlib import asynccontextmanager
from typing import List, Dict, AsyncGenerator, Any
from fastapi import FastAPI, Request, HTTPException #,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import logging
import asyncio
import re
import html
import json
from time import time
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style

# from libs.RAG.Retriever.CustomRetriever import CustomFAISSRetriever;
from libs.RAG.LLM.LLMInitializer import LLMInitializer;
from libs.RAG.Tools.ContentRetriever import GoogleSerperRetriever, CompositiveGoogleSerperSummarizer
from libs.utils.text_processing import format_serper_results_to_markdown
from libs.RAG.DB.MilvusQuery import MilvusQuery


###setup debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

##########################################
# 初始化FastAPI應用及其它程式起始需要初始的程式碼
##########################################


'''
start of fastapi startup: lifespan
'''
# using milvus vdb
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 全局變量
    # global faiss_retriever, faiss_retriever_qsrc, faiss_retriever_module, dfObj, dfObj_aitrial, llm
    global milvus_qry, llm, keys_order, _outputfields, _collectionName, headers, retrieval_num
    try:
        logging.info("開始初始化服務...")
        
        # 加載環境變數
        load_dotenv()
        encoding_model_name = os.getenv("EMBEDDEDING_MODEL")
        encoding_model_path = os.getenv("EMBEDDEDING_MODEL_PATH")
        retrieval_num = 10;
        LLM_MODEL='deepseek-r1:7b'#'phi4:latest';
        logging.info(f"Encoding model: {encoding_model_name}");
        logging.info(f"Encoding model path: {encoding_model_path}");

        logging.info("初始化內部資料...")
        _dbName="dqe_kb_db"
        _collectionName='qualityQA'
        keys_order = [
            "problemtype",
            "module",
            "severity",
            "description",
            "causeAnalysis",
            "improve",
            "experience"
        ]
        headers = ["问题瞭型","模块", "严重度(A/B/C)", "问题现象描述", "原因分析", "改善对策", "经验萃取"]
        _outputfields = ["problemtype", "module", "severity", "causeAnalysis", "description", "improve", "experience"]
        # 初始化milvus client object
        _embedding_path = encoding_model_path#"/home/mapleleaf/LCJRepos/Embedding_Models/jina-embeddings-v2-base-zh"
        logging.info(f"encoding_model_path is {_embedding_path}")
        milvus_qry = MilvusQuery(database=_dbName,embedding_model_path=_embedding_path)
        milvus_qry.load_collection(collection_name=_collectionName)
        logging.info("完成MilvusQuery物件初始化")
        
        # 初始化 LLM 模型
        llm = LLMInitializer().init_ollama_model()
        logging.info("llm 初始化成功")

        logging.info("Maple-Leaf AI KB 服務初始化完成")
        
        yield  # 應用啟動後，繼續執行其餘邏輯

    except Exception as e:
        logging.error(f"初始化過程中出錯: {e}")
        raise RuntimeError(f"初始化失敗: {e}")
    finally:
        # 如果需要，可以在這裡添加清理邏輯
        logging.info("應用即將關閉，清理資源...")


'''
end of lifespan
'''
app = FastAPI(lifespan=lifespan)

# CORS 設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生產環境中應該設置具體的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 設置模板目錄
templates = Jinja2Templates(directory="templates")
# 設置靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")


# stackexchange = StackExchangeAPIWrapper(result_separator='\n\n')
# googleserper = http.client.HTTPSConnection("google.serper.dev")
# SERPER_KEY = "3026422e86bf19796a06b5f15433e9c0cd9bd88a"

"""
问题现象描述:{submessages['question']}2
1.模块:{submessages['module']}0
2.严重度(A/B/C):{submessages['severity']}1
3.原因分析:{submessages['cause']}3
4.改善对策:{submessages['improve']}4
5.经验萃取:{submessages['experience']}5
6.评审后优化:{submessages['judge']}6
7.评分:{submessages['score']}7
"""

""" stackexchange return messages clean and format"""
def clean_text(text):
    # 移除所有 HTML 標籤（包括 span class="highlight"）
    text = re.sub(r'<[^>]+>', '', text)
    # 解碼 HTML 特殊符號
    text = html.unescape(text)
    return text.strip()

def format_qa_content(content:str=None):  
    # 分割問題塊
    questions = content.split('\n\n')
    questions = [q.strip() for q in questions if q.strip()]
    formatted_content = []
    for q in questions:
        # print(f"length of q is {len(q)}")
        question_end = q.find('\n')
        if question_end != -1:
            title = q[:question_end+1].strip()
            content = clean_text(q[question_end:].strip())
            formatted_q = f"{title}\nAnswer:\n{content}"
            formatted_content.append(formatted_q)
    final_content = "\n\n".join(formatted_content)
    return final_content

'''funtions for formatting content from google serper api'''


#check the is the text only contain english
def is_english(text):
    return bool(re.match(r'^[a-zA-Z\s]+$', text))

def gen_data_src_list(dict_data=None):
    row_list = [v for v in dict_data.values]
    return row_list
    # for _, v in dict_data.items:
        # row_list = [v for v in dict_data.values]

def sanitize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # 保留必要的符號
    special_chars = ['|', '-', ':', '<br>']
    for char in special_chars:
        text = text.replace(f' {char} ', char)
    # 移除多餘的空格
    text = ' '.join(text.split())
    return text


def search_similar_questions(question:str=None)->Any:
    raw_ret = milvus_qry.query(query=question, collection_name=_collectionName, output_fields=_outputfields, limit=retrieval_num)
    return raw_ret

def generate_markdown_table(value_matrix):
    writer = MarkdownTableWriter(
        tablename="回覆表格",
        headers=headers,
        value_matrix=value_matrix,
        column_alignments=["left"] * len(headers),  # Explicit alignment
        margin=1  # Add margin for better readability
    )
    # Add table style
    writer.style = Style(
        align_header="center",
        border_chars={
            "left": "|",
            "right": "|",
            "header": "-",
            "center": "|",
        }
    )
    return writer.dumps()

"""======================Generation of Multi-Rows======================"""
    
async def generate_content_milvus(message: str = None, ret_list:List[Dict[str,Any]]=None, search_action: int = 1, threshold:float=0.1) -> str:
    if message is None:
        raise ValueError("query string is none, please input query string.")
    
    value_matrix = milvus_qry.extract_entity_values_to_list(ret_list[0], keys_order, threshold)
    ret_md_table = generate_markdown_table(value_matrix=value_matrix);
    #進行與外部搜尋資料整合
    summaryObj = CompositiveGoogleSerperSummarizer(llm=llm)
    gen_data = summaryObj.generate_content(query=message, internal_content=value_matrix)
    print(f"整合後未去除推理資料:\n{gen_data}");
    gen_data = gen_data[gen_data.index("</think>"):]
    """make responses and return"""
    ret_dict = {
        'primary_msg':gen_data,
        # 'stackexchangemsg': format_response,
        # 'googleserper': gs_responses,
        'status_code':200
    }
    return ret_dict



# 根路由：渲染首頁
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 聊天內容模板
@app.get("/chat-content", response_class=HTMLResponse)
async def get_chat_content():
    try:
        with open("templates/chat/ai-chat-content.html", "r", encoding="utf-8") as file:
            content = file.read()
        return HTMLResponse(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat content: {str(e)}")



def replace_chinese_punctuation(text):
    if pd.isna(text):  # 處理 NaN 值
        return "N"
    # 更新替換對照表
    punctuation_map = {
        '，': ',',
        '。': '.',
        '：': ':',
        '；': ';',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '！': '!',
        '？': '?',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '、': ',',
        '「': '"',
        '」': '"',
        '『': "'",
        '』': "'",
        '\n':"<br>"
    }
    
    # 進行替換
    result = str(text)
    for ch, en in punctuation_map.items():
        result = result.replace(ch, en)
    
    # 使用 json.dumps 處理特殊字符轉義
    return json.dumps(result, ensure_ascii=False)[1:-1]  # 移除 json.dumps 產生的外層引號


def merge_and_remove_duplicated_items(list1:List=None, list2:List=None):
    return list(set(list1,list2))

async def do_google_serper_search(query:str=None)->str:
    google_serper = GoogleSerperRetriever(llm)
    _query = replace_chinese_punctuation(query);
    loop = asyncio.get_event_loop()
    ret_data = await loop.run_in_executor(None, google_serper.perform_query, _query)
    # ret_data = await google_serper.perform_query(query=_query)
    return ret_data
    

# 錯誤處理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

'''###############################################################'''
@app.post("/api/ai-chat-stream")
async def api_ai_chat_stream(request: Request):
    """
    Handles chat requests with streaming response.
    Streams internal results first, then performs web search concurrently,
    and finally streams the combined generated answer.
    """
    # try:
    data = await request.json()
    message = data.get("message")
    search_action = int(data.get("search_action", 1)) # Default to precise
    search_threshold = float(data.get("search_threshold", 0.3))

    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Return the streaming response, using the async generator
    # Use application/x-ndjson (newline delimited json) for easy frontend parsing
    return StreamingResponse(
        process_chat_stream(message, search_action, search_threshold),
        media_type="application/x-ndjson" # Newline Delimited JSON
    )
    # except Exception as e:
    #     logging.error(f"Error in stream endpoint setup: {e}", exc_info=True)
    #     # StreamingResponse can't easily return HTTP errors after starting,
    #     # so initial validation errors are raised as HTTPException.
    #     # Errors during generation are handled within the generator.
    #     raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")

async def process_chat_stream(message: str, search_action: int, search_threshold: float) -> AsyncGenerator[str, None]:
    start_time = time()
    internal_results_markdown = "nodata"
    # web_search_results = None
    # final_answer = ""

    try:
        # === Step 1 & 2: Internal Search and Yield ===
        logging.info("Step 1: Performing internal search...")
        
        chk_ifnodata = "ragsearch"
        internal_data_source = None

        # (Include your existing logic for search_action 1 and 2 here to populate _poses, _dists, chk_ifnodata)
        
        logging.info(f"***search_action***:{search_action}")
        if search_action == 1 or search_action == 3:
            _raw_ret = search_similar_questions(message);
            logging.info(_raw_ret[0])
            internal_data_source = milvus_qry.extract_entity_values_to_list(raw_entity_list=_raw_ret[0], keys=keys_order, threshold=float(search_threshold))
            # ... (rest of your thresholding logic for action 1) ...
            message = replace_chinese_punctuation(message);
            if not internal_data_source: chk_ifnodata="nodata"
            # else: internal_data_source = _raw_ret

        # Add handling if search_action is not 1 or 2 but requires internal data later?

        if chk_ifnodata != "nodata" and internal_data_source:
            # value_matrix = internal_data_source
            internal_results_markdown = generate_markdown_table(value_matrix=internal_data_source);

            internal_results_data = {
                "type": "internal_results",
                "content": internal_results_markdown
            }
            yield json.dumps(internal_results_data) + "\n"
            # logging.info("Step 2: Streamed internal search results.")
        else:
                internal_results_data = {"type": "internal_results", "content": "找不到任何内部资料"}
                yield json.dumps(internal_results_data) + "\n"
                logging.info("Step 2: Streamed internal search result (no data found).")

        # === Step 3: Initiate Concurrent Web Search ===
        web_search_task = None
        # Decide if web search is needed (e.g., always, or based on search_action, or if internal results are insufficient)
        # Let's assume for this example, we *always* do it if internal search happened or if action == 3
        should_web_search = search_action == 3 #True#(chk_ifnodata != "nodata") or (search_action == 3)

        if should_web_search:
                logging.info("Step 3: Initiating web search concurrently...")
                yield json.dumps({"type": "status", "message": "进行网路搜寻..."}) + "\n"
                # Use asyncio.create_task to run it in the background
                web_search_task = asyncio.create_task(do_google_serper_search(query=message))
        else:
                logging.info("Step 3: Skipping web search based on conditions.")


        # === Step 4 & 5: Wait for Web Search and Generate/Stream Final Answer ===
        # final_answer_data = {"type": "final_answer", "content": ""}
        logging.info(f"*****web_search_task*****:{web_search_task}")
        if web_search_task and should_web_search:
            logging.info("Step 4: Waiting for web search completion...")
            raw_web_results = await web_search_task
            logging.info("Step 4: Web search completed.")
            # Format web results if needed (your existing logic)
            web_search_results_markdown = format_serper_results_to_markdown(serper_data=raw_web_results)

            # Prepare for final generation using LLM
            # Option A: Use existing CompositiveGoogleSerperSummarizer (needs modification potentially)
            # Option B: Create a new prompt/chain combining internal_results_markdown and web_search_results_markdown

            # --- Using a simplified direct LLM call for demonstration ---
            # You should replace this with your actual summarization/generation logic
            # Ensure your LLM call is async (e.g., llm.ainvoke or similar)
            logging.info("Step 5: Generating final combined answer...")
            yield json.dumps({"type": "status", "message": "正在进行资料整理..."}) + "\n"

            # Construct the prompt input
            # Get internal context string/markdown
            internal_context_for_llm = "No internal data found."
            if chk_ifnodata != "nodata" and internal_data_source:
                    # Convert value_matrix or use internal_results_markdown
                    internal_context_for_llm = internal_results_markdown #json.dumps() # Example

            # Define the combined prompt (adjust as needed)
            combined_template = """
            User Query: {query}

            Internal Knowledge Base Results:
            {internal_context}

            Web Search Results:
            {web_context}

            Task: 
            Based on the user query, internal results, and web search results, you **must** think step by step to provide a comprehensive answer.
            First, summarize the key findings from internal data relevant to the query.
            Second, summarize the key findings from the web search relevant to the query.
            Finally, provide a synthesized answer combining insights from both sources. 
            You **Must** Use Markdown format.
            Answer in Simplified Chinese.
            """
            combined_prompt = PromptTemplate(input_variables=["query", "internal_context", "web_context"], template=combined_template)
            chain = combined_prompt | llm # Assuming your LLMInitializer sets up LangChain compatible llm

            # Use astream for streaming output from LLM if available
            final_answer_full = ""
            async for chunk in chain.astream({
                "query": message,
                "internal_context": internal_context_for_llm,
                "web_context": web_search_results_markdown
            }):
                # Assuming 'chunk' is a string or has a 'content' attribute based on your LLM setup
                content_chunk = chunk if isinstance(chunk, str) else getattr(chunk, 'content', '')
                if content_chunk:
                    final_answer_full += content_chunk
                    chunk_data = {"type": "final_answer_chunk", "content": content_chunk}
                    yield json.dumps(chunk_data) + "\n"

            # If astream is not available or you need the full response first:
            # final_answer_full = await chain.ainvoke({ ... })
            # final_answer_data["content"] = final_answer_full
            # yield json.dumps(final_answer_data) + "\n"

            logging.info("Step 5: Finished streaming final answer.")

        # else:
        #     # Handle case where only internal search was done (or neither)
        #     if chk_ifnodata != "nodata":
        #             final_answer_data["content"] = internal_results_markdown #internal_results_formatted # Or a summary of it
        #             yield json.dumps(final_answer_data) + "\n"
        #     else:
        #             final_answer_data["content"] = "No relevant information found internally or via web search."
        #             yield json.dumps(final_answer_data) + "\n"
    

    except Exception as e:
        logging.error(f"Error during chat processing: {e}", exc_info=True)
        error_data = {"type": "error", "message": f"An error occurred: {str(e)}"}
        yield json.dumps(error_data) + "\n"
    finally:
        end_time = time()
        total_time = end_time - start_time
        logging.info(f"Total processing time: {total_time}")
        done_data = {"type": "done", "computation_time": total_time}
        yield json.dumps(done_data) + "\n"

# 啟動應用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
