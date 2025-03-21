import os
import dspy
from typing import List, Dict, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# from app.ai_chat_service import AIChatService
from dotenv import load_dotenv
import pandas as pd
# from openai import AsyncOpenAI
# import asyncio
# from libs.base_classes import AssistantRequest, AssistantResponse
import logging
import numpy as np
from libs.RAG.Retriever.CustomRetriever import CustomFAISSRetriever;
from langchain_ollama.llms import OllamaLLM
# from langchain_community.utilities import StackExchangeAPIWrapper
import http.client
import re
import html
import json
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
import re

###setup debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

##########################################
# 初始化FastAPI應用及其它程式起始需要初始的程式碼
##########################################
app = FastAPI()

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
LLM_MODEL='phi4:latest';
llm=None;

# stackexchange = StackExchangeAPIWrapper(result_separator='\n\n')
googleserper = http.client.HTTPSConnection("google.serper.dev")
SERPER_KEY = "3026422e86bf19796a06b5f15433e9c0cd9bd88a"

# 加載環境變數
load_dotenv()
encoding_model_name = os.getenv("SENTENCE_MODEL")
faiss_index_path = os.getenv("FAISS_INDEX_PATH")
faiss_index_path_qsr = os.getenv("FAISS_INDEX_QUESRC_INDEX");
faiss_index_path_module = os.getenv("FAISS_INDEX_MODULE_INDEX");
datasrc_deqlearn = os.getenv("DATASRC_DEQ_LEARN");
datasrc_deqaitrial = os.getenv("DATASRC_DEQ_AITRIAL");

# vector_db_path = os.getenv("FAISS_DB_PATH")
logging.info(f"Encoding model: {encoding_model_name}");
logging.info(f"FAISS index path: {faiss_index_path}");
# logging.info(f"Vector DB path: {vector_db_path}");

required_columns = [
        '模块', '严重度', '问题现象描述', '原因分析', 
        '改善对策', '经验萃取', '评审后优化', '评分'
]
# 定義distance threshold

# regular expression ptterns

################## LLM Initialization ##################

def InitializeLLM_DeepSeekR1():
        local_config = {
            "api_base": "http://localhost:11434/v1",  # 注意需加/v1路徑
            "api_key": "NULL",  # 特殊標記用於跳過驗證
            "model": "deepseek-r1:7b",
            "custom_llm_provider":"deepseek"
        }
        dspy.configure(
            lm=dspy.LM(
                **local_config
            )
        )
        print("DeepSeek-R1 has initialized!")

def InitializeLLM_Phi4():
    global llm;
    if llm == None:
        OllamaLLM(model=LLM_MODEL)
    else:
        print("llm has initialized........");
        return;


###########
# LLM初始化
###########
# InitializeLLM_DeepSeekR1();
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
    #    移除所有 HTML 標籤（包括 span class="highlight"）
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
def format_context_to_md(context):
    """智能Markdown生成器（支持多类型内容识别与自动格式化）
    
    参数:
        context (str): 包含标题、链接、描述的原始文本
        
    返回:
        str: 结构化Markdown文档，含自动生成的目录锚点
    """

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

# def sanitize_text(text):
#     if pd.isna(text):
#         return ""
#     text = str(text).strip()
    
#     # Escape special characters that could break markdown tables
#     escape_chars = {
#         '|': '\\|',
#         '\n': '<br>',
#         '\r': '',
#         '_': '\\_',
#         '*': '\\*'
#     }
#     for char, replacement in escape_chars.items():
#         text = text.replace(char, replacement)
#     # Remove multiple spaces
#     text = ' '.join(text.split())
#     return text


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

"""======================Generation of Single-Rows======================"""
async def generate(message: str = None, submessages: dict = None, history: List[Dict[str, str]] = None, model: str = "deepseekv2", search_action: int = 1) -> str:
    # global stackexchange;
    # global googleserper;
    # global SERPER_KEY;
    # 精準搜尋邏輯
    print("................執行精準搜尋................")
    if message is None:
        raise ValueError("query string is none, please input query string.")
    # try:
    # 清理所有輸入數據
    cleaned_messages = {
        k: sanitize_text(v) for k, v in submessages.items()
    }
    print(f"cleaned_messages:\n{cleaned_messages}\n\n==================================================\n\n")

    headers=["模块", "严重度(A/B/C)", "题现象描述", "原因分析", "改善对策", "经验萃取", "审后优化", "评分"]
    value_matrix = [
        [v for v in cleaned_messages.values()]
    ]
    # print(value_matrix[0])
   
    ret_md_table = generate_markdown_table(headers=headers, value_matrix=value_matrix);
    print(f"ret_md_table:\n{ret_md_table}");

    """make responses and return"""
    ret_dict = {
        'primary_msg':ret_md_table,
        # 'stackexchangemsg': format_response,
        # 'googleserper': gs_responses,
        'status_code':200
    }
    return ret_dict       


    
#     except Exception as e:
#         raise RuntimeError(f"Error : {str(e)}")

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
    
    result_list = []
    for idx in pos_list:
        _row = df.iloc[idx]
        result_list.extend(convert_dr_to_list(_row));
        # result_list.extend(convert_dr_to_list(_row, required_columns));
    return result_list

"""======================Generation of Multi-Rows======================"""
async def generate_multirows(message: str = None, data_frame:pd.DataFrame = None,  data_pos:List=None, search_distances:List=None, history: List[Dict[str, str]] = None, model: str = "deepseek-r1", search_action: int = 2) -> str:
    if message is None:
        raise ValueError("query string is none, please input query string.")
    
    value_matrix = convert_df_to_list(data_frame, data_pos)
    for j in range(len(search_distances)):
        dist = search_distances[j]
        related_degree = "";
        if dist < 0.5 and dist > 0:
            related_degree = "高"
        elif dist< 10 and dist > 0.5:
            related_degree = "中"
        else:
            related_degree = "低"
        value_matrix[j] = [related_degree]+value_matrix[j]

    # 產生markdown table
    # headers=["關聯度","模块", "严重度(A/B/C)", "题现象描述", "原因分析", "改善对策", "经验萃取", "审后优化", "评分"]
    headers=["關聯度","模块", "严重度(A/B/C)", "题现象描述", "原因分析", "改善对策", "经验萃取", "审后优化", "评分"]
    ret_md_table = generate_markdown_table(headers=headers, value_matrix=value_matrix);
    
        
        # ret_md_table[j] = ret_md_table[0][j].insert(0, related_degree)
        
    print(f"ret_md_table:\n{ret_md_table}");

    """make responses and return"""
    ret_dict = {
        'primary_msg':ret_md_table,
        # 'stackexchangemsg': format_response,
        # 'googleserper': gs_responses,
        'status_code':200
    }
    return ret_dict       
    
def search_similar_questions(retriever, question):
    pos, distances = retriever(question)
    return pos, distances

@app.on_event("startup")
async def startup_event():
    # global ai_chat_service
    global faiss_retriever;
    global faiss_retriever_qsrc;
    global faiss_retriever_module;
    global dfObj;
    global dfObj_aitrial;
    try:
        logging.info("start to initialize services.......");
        # ai_chat_service = AIChatService(); # 初始化 KB聊天物件
        # print(f"vector db path: {vector_db_path}");
        faiss_retriever = CustomFAISSRetriever(faiss_index_path=faiss_index_path, vector_db_path="nodata", model_name=encoding_model_name, k=3); # 初始化 faiss 檢索物件
        print("faiss_retriever initializing succeed........");
        faiss_retriever_qsrc = CustomFAISSRetriever(faiss_index_path=faiss_index_path_qsr,
                                                    vector_db_path="nodata",model_name=encoding_model_name,k=4);
        print("faiss_retriever_qsrc initializing succeed........");
        faiss_retriever_module = CustomFAISSRetriever(faiss_index_path=faiss_index_path_module,
                                                    vector_db_path="nodata",model_name=encoding_model_name,k=4);
        print("faiss_retriever_module initializing succeed........");
        dfObj = pd.read_csv(datasrc_deqlearn, encoding='utf-8-sig'); # 初始化 faiss 檢索物件
        print("dfObj initializing succeed........");
        dfObj_aitrial = pd.read_csv(datasrc_deqaitrial, encoding='utf-8-sig');
        print("dfObj_aitrial initializing succeed........");

        # 初始化助手服務管理器
        logging.info("chat, essay-advisor, k9-helper services are initialized!");
    except Exception as e:
        logging.error(f"Failed to run in startup_event: {e}");
        raise RuntimeError(f"Failed to initialize AIChatService: {e}")

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


# AI 聊天接口（非流式）
@app.post("/api/ai-chat")
async def api_ai_chat(request: Request):
    # try:
        # 從請求中提取用戶消息
        # _pos=None;
        # _distances=None;
        submessages=None;
        data = await request.json()
        search_action = data.get("search_action")  # 預設為精準搜尋
        search_threshold = float(data.get("search_threshold", 3.0))  # 新增參數接收
        print(f"---------------search_action:{search_action}--------------");
        message = data.get("message")
        chk_ifnodata = ""
        if search_action == 1:
            _pos, _distances = search_similar_questions(faiss_retriever, message);
            print(f"_pos:{_pos}\n_distances:{_distances}");
            max_pos = np.amax(_pos[0]);
            min_distance = np.amin(_distances[0]);
            # min_distance_index = np.argmin(_distances[0]);
            print(f"most min distance:{min_distance}")
            message = replace_chinese_punctuation(message);
            submessages = getSubMessages(dfObj.iloc[max_pos])
            # if min_distance < 11:
            #     submessages = getSubMessages(dfObj.iloc[max_pos])
            # else:
            #     #"没有任何匹配的资料"
            #     chk_ifnodata="nodata";
        else:
            #進行相似度搜尋
            _pos_qsrc, _distances_qsrc = search_similar_questions(faiss_retriever_qsrc, message);
            _pos_module, _distance_module =  search_similar_questions(faiss_retriever_module,message);
            combined_pos_list=None
            combined_dist_list = None

            if not isinstance(_pos_qsrc, np.ndarray):
                _pos_qsrc.extend(_pos_module);
                _distances_qsrc.extend(_distance_module)
                combined_pos_list, combined_dist_list = combine_pos_distance(_pos_qsrc, _distances_qsrc);
            else:
                _conc_pos_list = np.concatenate([_pos_qsrc, _pos_module], axis=0)
                _conc_dist_list = np.concatenate([_distances_qsrc, _distance_module], axis=0)
                combined_pos_list, combined_dist_list = combine_pos_distance(_conc_pos_list, _conc_dist_list);
            print(f"combined_pos_list:\n{combined_pos_list}\ncombined_dist_list:\n{combined_dist_list}");
            
            message = replace_chinese_punctuation(message);
            if(len(combined_pos_list) > 0):
                submessages = dfObj_aitrial;
            if len(combined_pos_list) < 1:
                chk_ifnodata="nodata";
        
        # call generate function
        ai_response = None;
        if(chk_ifnodata!="nodata"):
            if search_action == 1:
                ai_response = await generate(message=message, submessages=submessages, search_action=search_action);
            else:
                ai_response = await generate_multirows(message=message, data_frame=submessages, data_pos=combined_pos_list, search_distances=combined_dist_list, search_action=search_action)
        else:
            ai_response = {
                'primary_msg':"nodata",
                'status_code':200
            }
        # 返回 AI 的回應
        return {"response": ai_response}; 
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


# AI 聊天接口（流式）
# @app.post("/api/ai-chat-stream")
# async def api_ai_chat_stream(request: Request):
#     try:
#         # 從請求中提取用戶消息
#         data = await request.json()
#         message = data.get("message")
#         history = data.get("history", [])
#         if not message:
#             raise HTTPException(status_code=400, detail="Message is required")
#         # 調用 AIChatService 的流式生成方法
#         async def stream_generator():
#             async for chunk in ai_chat_service.generate_stream(message, history):
#                 yield {"content": chunk}

#         return stream_generator()
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

# 錯誤處理
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


# 啟動應用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
