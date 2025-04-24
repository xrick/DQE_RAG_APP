import os
from contextlib import asynccontextmanager
from typing import List, Dict, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException #,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# from app.ai_chat_service import AIChatService
from dotenv import load_dotenv
import pandas as pd
import logging
import numpy as np
import re
import html
import json
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
from libs.RAG.Retriever.CustomRetriever import CustomFAISSRetriever;
from libs.RAG.LLM.LLMInitializer import LLMInitializer;
from libs.RAG.Tools.ContentRetriever import GoogleSerperRetriever, CompositiveGoogleSerperSummarizer
from libs.utils.text_processing import format_serper_results_to_markdown
from time import time
from fastapi.responses import StreamingResponse
import asyncio
from langchain.prompts import PromptTemplate

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
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 全局變量
    # global faiss_retriever, faiss_retriever_qsrc, faiss_retriever_module, dfObj, dfObj_aitrial, llm
    global faiss_retriever, dfObj, llm
    try:
        logging.info("開始初始化服務...")

        # 初始化 FAISS 檢索器
        faiss_retriever = CustomFAISSRetriever(
            faiss_index_path=faiss_index_path,
            vector_db_path="nodata",
            model_name=encoding_model_name,
            k=retrieval_num,
        )
        logging.info("faiss_retriever 初始化成功")

        # faiss_retriever_qsrc = CustomFAISSRetriever(
        #     faiss_index_path=faiss_index_path_qsr,
        #     vector_db_path="nodata",
        #     model_name=encoding_model_name,
        #     k=retrieval_num,
        # )
        # logging.info("faiss_retriever_qsrc 初始化成功")

        # faiss_retriever_module = CustomFAISSRetriever(
        #     faiss_index_path=faiss_index_path_module,
        #     vector_db_path="nodata",
        #     model_name=encoding_model_name,
        #     k=retrieval_num,
        # )
        # logging.info("faiss_retriever_module 初始化成功")

        # 加載 CSV 數據
        dfObj = pd.read_csv(datasrc_deqlearn, encoding='utf-8-sig')
        logging.info("dfObj 初始化成功")

        # dfObj_aitrial = pd.read_csv(datasrc_deqaitrial, encoding='utf-8-sig')
        # logging.info("dfObj_aitrial 初始化成功")

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
LLM_MODEL='deepseek-r1:7b'#'phi4:latest';
# llm=None;

# stackexchange = StackExchangeAPIWrapper(result_separator='\n\n')
# googleserper = http.client.HTTPSConnection("google.serper.dev")
# SERPER_KEY = "3026422e86bf19796a06b5f15433e9c0cd9bd88a"

# 加載環境變數
load_dotenv()
encoding_model_name = os.getenv("SENTENCE_MODEL")
faiss_index_path = os.getenv("FAISS_INDEX_PATH")
faiss_index_path_qsr = os.getenv("FAISS_INDEX_QUESRC_INDEX");
faiss_index_path_module = os.getenv("FAISS_INDEX_MODULE_INDEX");
datasrc_deqlearn = os.getenv("DATASRC_DEQ_LEARN")
datasrc_deqaitrial = os.getenv("DATASRC_DEQ_AITRIAL")
retrieval_num = 4;

logging.info(f"Encoding model: {encoding_model_name}");
logging.info(f"FAISS index path: {faiss_index_path}");
# logging.info(f"Vector DB path: {vector_db_path}");

#所需欄位定義
required_columns = [
        '模块', '严重度', '问题现象描述', '原因分析', 
        '改善对策', '经验萃取', '评审后优化', '评分'
]

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


def search_similar_questions(retriever, question):
    pos, distances = retriever(question)
    return pos, distances

def generate_markdown_table(headers, value_matrix):
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


def convert_dr_to_list(row):
    global required_columns;
    # 獲取當前行的所有值
    print(f"type of row:{type(row)}")
    row_data = []
    max_length = 1  # 預設長度為1
    # 步驟4：檢查是否有列表值並確定最大長度
    for col in required_columns:
        print(f"row[col]:\n    col:{col}\n    row[col]:{row[col]}\n");
        if isinstance(row[col], list):
            max_length = max(max_length, len(row[col]))
    # 步驟5：處理每一列的值
    for i in range(max_length):
        current_row = []
        for col in required_columns:
            value = row[col]
            value = sanitize_text(replace_chinese_punctuation(value)) #進行資料清理
            if isinstance(value, list):
                # 如果是列表，取對應索引的值，如果索引超出範圍則使用空字符串
                current_row.append(value[i] if i < len(value) else '')
            else:
                # 如果不是列表，則重複使用該值
                current_row.append(value)
        row_data.append(current_row)
    return row_data

def convert_df_to_list(df, pos_list)->List:
    """
    將DataFrame轉換為m*n列表
    Parameters:
    df (pandas.DataFrame): 輸入的DataFrame，包含指定的列
    Returns:
    list: 轉換後的二維列表
    """
    # 步驟1：定義所需的列順序
    logging.info(f"pos_list passed to convert_df_to_list:{pos_list}")
    logging.info(f"df rows:{df.shape}")
    result_list = []
    for idx in pos_list:
        _row = df.iloc[idx]
        result_list.extend(convert_dr_to_list(_row));
        # result_list.extend(convert_dr_to_list(_row, required_columns));
    return result_list

"""======================Generation of Multi-Rows======================"""
async def generate_content(message: str = None, data_frame:pd.DataFrame = None,  data_pos:List=None, search_distances:List=None, history: List[Dict[str, str]] = None, model: str = "deepseek-r1", search_action: int = 2) -> str:
    if message is None:
        raise ValueError("query string is none, please input query string.")
    
    value_matrix = convert_df_to_list(data_frame, data_pos)
    #以下產生關聯度
    # for j in range(len(search_distances)):
    #     dist = search_distances[j]
    #     related_degree = "";
    #     if dist < 5 and dist >= 0:
    #         related_degree = "高<br>"#+str(dist)
    #     elif dist< 11 and dist > 5:
    #         related_degree = "中<br>"#+str(dist)
    #     else:
    #         related_degree = "低<br>"#+str(dist)
    #     value_matrix[j] = [related_degree]+value_matrix[j]

    # 產生markdown table
    # headers=["關聯度","模块", "严重度(A/B/C)", "问题现象描述", "原因分析", "改善对策", "经验萃取", "审后优化", "评分"]
    # headers=["關聯度","模块", "严重度(A/B/C)", "问题现象描述", "原因分析", "改善对策", "经验萃取", "审后优化", "评分"]
    # ret_md_table = generate_markdown_table(headers=headers, value_matrix=value_matrix);
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

def getSubMessages(df_row):
    return {
        "module": replace_chinese_punctuation(str(df_row['模块'])),
        "severity": replace_chinese_punctuation(str(df_row['严重度'])),
        "description": replace_chinese_punctuation(str(df_row['问题现象描述'])),
        "cause": replace_chinese_punctuation(str(df_row['原因分析'])),
        "improve": replace_chinese_punctuation(str(df_row['改善对策'])),
        "experience": replace_chinese_punctuation(str(df_row['经验萃取'])),
        "judge": replace_chinese_punctuation(str(df_row['评审后优化'])),
        "score": replace_chinese_punctuation(str(df_row['评分']))
    }

def merge_and_remove_duplicated_items(list1:List=None, list2:List=None):
    return list(set(list1,list2))

def combine_pos_distance(pos_lists, distance_lists):
    # 合并并去重位置列表
    all_pos = [pos for sublist in pos_lists for pos in sublist if pos > -1]
    combined_pos = sorted(set(all_pos))
    
    # 建立位置-距离映射表
    mappings = []
    for pos_sublist, dist_sublist in zip(pos_lists, distance_lists):
        mappings.append(dict(zip(pos_sublist, dist_sublist)))
    
    # 合并距离列表
    combined_dist = []
    for pos in combined_pos:
        distances = []
        for mapping in mappings:
            if pos in mapping:
                distances.append(mapping[pos])
        avg = round(sum(distances)/len(distances), 3) if distances else 0
        combined_dist.append(avg)
    
    return combined_pos, combined_dist

def sort_list_pair(pos_list:List=None, dist_list:List=None):
    sorted_pairs = sorted(zip(dist_list, pos_list), key=lambda x: x[0])
    dist_list_sorted, pos_list_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # convert to List type
    pos_list_sorted = list(pos_list_sorted)
    dist_list_sorted = list(dist_list_sorted)
    return pos_list_sorted, dist_list_sorted

async def do_google_serper_search(query:str=None)->str:
    google_serper = GoogleSerperRetriever(llm)
    _query = replace_chinese_punctuation(query);
    loop = asyncio.get_event_loop()
    ret_data = await loop.run_in_executor(None, google_serper.perform_query, _query)
    # ret_data = await google_serper.perform_query(query=_query)
    return ret_data

# AI 聊天接口（非流式）
@app.post("/api/ai-chat")
async def api_ai_chat(request: Request):
    """
    edit note:
    2025/04/02:
    """
    # try:
    start_time=time()
    data = await request.json()
    search_action = int(data.get("search_action"))  # 預設為精準搜尋
    search_threshold = float(data.get("search_threshold", 25.0))  # 新增參數接收
    print(f"---------------search_action:{search_action}--------------");
    print(f"---------------search_threshold:{search_threshold}--------------");
    message = data.get("message")
    ret_data = None
    chk_ifnodata = "ragsearch"
    _poses = []
    _dists = []
    if search_action == 1:
        _pos, _distances = search_similar_questions(faiss_retriever, message);
        print(f"_pos:{_pos}\n_distances:{_distances}");
        sorted_poses, sorted_dists = sort_list_pair(_pos[0], _distances[0])
        print(f"sorted_pos:{sorted_poses}\nsorted_distances:{sorted_dists}");
        pos_len = len(sorted_poses)
        dist = -1.0
        # min_dist_idx = -1
        min_dist = 99999
        for i in range(pos_len):
            dist = sorted_dists[i]
            if dist < search_threshold and dist != -1:
                _poses.append(sorted_poses[i])
                _dists.append(sorted_dists[i])
                if min_dist > dist:
                    min_dist = dist
                    # min_dist_idx = sorted_poses[i]
            # submessages = getSubMessages(dfObj.iloc[min_dist_idx])
        if not _poses:
            #"没有任何匹配的资料"
            chk_ifnodata="nodata"
    else:
        if search_action == 3:
            try:
                #web search
                #以下三行移至do_google_serper_search
                # google_serper = GoogleSerperRetriever(llm)
                # query = replace_chinese_punctuation(message);
                # ret_data = google_serper.perform_query(query=query)
                ret_data = await do_google_serper_search(query=message)
                print(f"轉換為markdown前..........\n查詢內容：{message}\n回傳結果:\n{ret_data}")
                ret_data = format_serper_results_to_markdown(serper_data=ret_data)
                print(f"\n轉換為markdown後..........\n查詢內容：{message}\n回傳結果:\n{ret_data}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
    
    # call generate function
    ai_response = None;
    if search_action != 99:
        if chk_ifnodata != "nodata":
            if search_action == 1:
                ai_response = await generate_content(message=message, data_frame=dfObj, data_pos=_poses, search_distances=_dists, search_action=search_action);
            # elif search_action == 2:
            #     ai_response = await generate_content(message=message, data_frame=dfObj_aitrial, data_pos=_poses, search_distances=_dists, search_action=search_action)
            else:
                ai_response = {
                    'primary_msg':ret_data,
                    'status_code':200
                }
        else:
            ai_response = {
                'primary_msg':"nodata",
                'status_code':200
            }
    # 返回 AI 的回應
    end_time=time()
    totaltime = end_time-start_time
    logging.info(f"查詢耗時:{totaltime}")
    ai_response["computation_time"]=totaltime
    return {"response": ai_response}; 
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    

# 錯誤處理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

'''
###############################################################
'''
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
    search_threshold = float(data.get("search_threshold", 25.0))

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
    web_search_results = None
    final_answer = ""

    try:
        # === Step 1 & 2: Internal Search and Yield ===
        logging.info("Step 1: Performing internal search...")
        _poses = []
        _dists = []
        chk_ifnodata = "ragsearch"
        internal_data_source = None

        # (Include your existing logic for search_action 1 and 2 here to populate _poses, _dists, chk_ifnodata)
        
        logging.info(f"***search_action***:{search_action}")
        if search_action == 1 or search_action == 3:
            _pos, _distances = search_similar_questions(faiss_retriever, message);
            # ... (rest of your thresholding logic for action 1) ...
            print(f"_pos:{_pos}\n_distances:{_distances}");
            sorted_poses, sorted_dists = sort_list_pair(_pos[0], _distances[0])
            print(f"sorted_pos:{sorted_poses}\nsorted_distances:{sorted_dists}");
            pos_len = len(sorted_poses)
            dist = -1.0
            min_dist = 99999
            for i in range(pos_len):
                dist = sorted_dists[i]
                if dist < search_threshold and dist != -1:
                    _poses.append(sorted_poses[i])
                    _dists.append(sorted_dists[i])
                    if min_dist > dist:
                        min_dist = dist
            message = replace_chinese_punctuation(message);
            if not _poses: chk_ifnodata="nodata"
            else: internal_data_source = dfObj

        # Add handling if search_action is not 1 or 2 but requires internal data later?

        if chk_ifnodata != "nodata" and internal_data_source is not None:
            value_matrix = convert_df_to_list(internal_data_source, _poses)
            headers=["模块", "严重度(A/B/C)", "问题现象描述", "原因分析", "改善对策", "经验萃取", "审后优化", "评分"]
            internal_results_markdown = generate_markdown_table(headers=headers, value_matrix=value_matrix);

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
        final_answer_data = {"type": "final_answer", "content": ""}
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
            if chk_ifnodata != "nodata" and internal_data_source is not None:
                    # Convert value_matrix or use internal_results_markdown
                    internal_context_for_llm = json.dumps(convert_df_to_list(internal_data_source, _poses)) # Example

            # Define the combined prompt (adjust as needed)
            combined_template = """
            User Query: {query}

            Internal Knowledge Base Results:
            {internal_context}

            Web Search Results:
            {web_context}

            Task: Based on the user query, internal results, and web search results, provide a comprehensive answer.
            First, summarize the key findings from internal data relevant to the query.
            Second, summarize the key findings from the web search relevant to the query.
            Finally, provide a synthesized answer combining insights from both sources. Use Markdown format.
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
