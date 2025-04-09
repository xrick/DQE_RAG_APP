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
import logging
import numpy as np
from langchain_ollama.llms import OllamaLLM
import http.client
import re
import html
import json
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
import re
from libs.RAG.Retriever.CustomRetriever import CustomFAISSRetriever;
from libs.RAG.LLM.LLMInitializer import LLMInitializer;
from libs.RAG.Tools.ContentRetriever import GoogleSerperRetriever
from libs.utils.text_processing import format_serper_results_to_markdown
# from openai import AsyncOpenAI
# import asyncio
# from libs.base_classes import AssistantRequest, AssistantResponse
# from langchain_community.utilities import StackExchangeAPIWrapper

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
    logging.info(f"pos_list passed to convert_df_to_list:{pos_list}")
    logging.info(f"df rows:{df.shape}")
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
        if dist < 5 and dist >= 0:
            related_degree = "高<br>"+str(dist)
        elif dist< 11 and dist > 5:
            related_degree = "中<br>"+str(dist)
        else:
            related_degree = "低<br>"+str(dist)
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
    


@app.on_event("startup")
async def startup_event():
    # global ai_chat_service
    global faiss_retriever;
    global faiss_retriever_qsrc;
    global faiss_retriever_module;
    global dfObj;
    global dfObj_aitrial;
    global llm;
    try:
        logging.info("start to initialize services.......");
        # ai_chat_service = AIChatService(); # 初始化 KB聊天物件
        # print(f"vector db path: {vector_db_path}");
        faiss_retriever = CustomFAISSRetriever(faiss_index_path=faiss_index_path, vector_db_path="nodata", model_name=encoding_model_name, k=retrieval_num); # 初始化 faiss 檢索物件
        logging.info("faiss_retriever initializing succeed........");
        faiss_retriever_qsrc = CustomFAISSRetriever(faiss_index_path=faiss_index_path_qsr,
                                                    vector_db_path="nodata",model_name=encoding_model_name,k=retrieval_num);
        logging.info("faiss_retriever_qsrc initializing succeed........");
        faiss_retriever_module = CustomFAISSRetriever(faiss_index_path=faiss_index_path_module,
                                                    vector_db_path="nodata",model_name=encoding_model_name,k=retrieval_num);
        logging.info("faiss_retriever_module initializing succeed........");
        dfObj = pd.read_csv(datasrc_deqlearn, encoding='utf-8-sig'); # 初始化 faiss 檢索物件
        logging.info("dfObj initialized succeed........");
        dfObj_aitrial = pd.read_csv(datasrc_deqaitrial, encoding='utf-8-sig');
        logging.info("dfObj_aitrial initialized succeed........");
        llm = LLMInitializer().init_ollama_model()
        logging.info("llm initialized succeed........");

        # 初始化助手服務管理器
        logging.info("Maple-Leaf AI KB Services Initialized...");
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

def sort_list_pair(pos_list:List=None, dist_list:List=None):
    sorted_pairs = sorted(zip(dist_list, pos_list), key=lambda x: x[0])
    dist_list_sorted, pos_list_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # convert to List type
    pos_list_sorted = list(pos_list_sorted)
    dist_list_sorted = list(dist_list_sorted)
    return pos_list_sorted, dist_list_sorted

# AI 聊天接口（非流式）
@app.post("/api/ai-chat")
async def api_ai_chat(request: Request):
    """
    edit note:
    2025/04/02:
    """
    # try:
    data = await request.json()
    search_action = int(data.get("search_action"))  # 預設為精準搜尋
    search_threshold = float(data.get("search_threshold", 15.0))  # 新增參數接收
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
    elif search_action==2:
        #進行相似度搜尋
        _pos_qsrc, _distances_qsrc = search_similar_questions(faiss_retriever_qsrc, message);
        _pos_module, _distance_module =  search_similar_questions(faiss_retriever_module,message);
        message = replace_chinese_punctuation(message);
        if not isinstance(_pos_qsrc, np.ndarray):
            _pos_qsrc.extend(_pos_module);
            _distances_qsrc.extend(_distance_module)
            _poses, _dists = combine_pos_distance(_pos_qsrc, _distances_qsrc);
        else:
            _conc_pos_list = np.concatenate([_pos_qsrc, _pos_module], axis=0)
            _conc_dist_list = np.concatenate([_distances_qsrc, _distance_module], axis=0)
            _poses, _dists = combine_pos_distance(_conc_pos_list, _conc_dist_list);
        print(f"combined_pos_list:\n{_poses}\ncombined_dist_list:\n{_dists}");
        message = replace_chinese_punctuation(message);
        if not _poses:
            search_action = 99
            chk_ifnodata="nodata";
    else:
        if search_action == 3:
            try:
                # 
                google_serper = GoogleSerperRetriever(llm)
                query = replace_chinese_punctuation(message);
                # summary = wiki_summarizer.summarize(query).replace("Abstract:","摘要：").replace("Summarization Content:","内容概要：").replace("a. Most Important Points:","一、重点说明：").replace("b. Extended Content:","二、延伸内容：")
                # filtered_summary = summary[summary.find("</think>")+8:]
                ret_data = google_serper.perform_query(query=query)
                print(f"轉換為markdown前..........\n查詢內容：{query}\n回傳結果:\n{ret_data}")
                ret_data = format_serper_results_to_markdown(serper_data=ret_data)
                print(f"\n轉換為markdown後..........\n查詢內容：{query}\n回傳結果:\n{ret_data}")
                
            except Exception as e:
                print(f"Error: {str(e)}")
    
    # call generate function
    ai_response = None;
    if search_action != 99:
        if chk_ifnodata != "nodata":
            if search_action == 1:
                ai_response = await generate_multirows(message=message, data_frame=dfObj, data_pos=_poses, search_distances=_dists, search_action=search_action);
            elif search_action == 2:
                ai_response = await generate_multirows(message=message, data_frame=dfObj_aitrial, data_pos=_poses, search_distances=_dists, search_action=search_action)
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


# 啟動應用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
