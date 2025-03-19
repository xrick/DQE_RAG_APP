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
# import http.client
import re
import html
import json
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style

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
datasrc_deqlearn = os.getenv("DATASRC_DEQ_LEARN");
datasrc_deqaitrial = os.getenv("DATASRC_DEQ_AITRIAL");

# vector_db_path = os.getenv("FAISS_DB_PATH")
logging.info(f"Encoding model: {encoding_model_name}");
logging.info(f"FAISS index path: {faiss_index_path}");
# logging.info(f"Vector DB path: {vector_db_path}");

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
InitializeLLM_DeepSeekR1();
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
    print(value_matrix[0])
   
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

"""======================Generation of Multi-Rows======================"""
async def generate_multirows(message: str = None, submessages: dict = None, history: List[Dict[str, str]] = None, model: str = "deepseekv2", search_action: int = 1) -> str:
    # global stackexchange;
    # global googleserper;
    # global SERPER_KEY;
    print("執行標籤搜尋...")
    if message is None:
        raise ValueError("query string is none, please input query string.")
    # try:
    # 清理所有輸入數據
    cleaned_messages = {
        k: sanitize_text(v) for k, v in submessages.items()
    }
    print(f"cleaned_messages:\n{cleaned_messages}\n\n==================================================\n\n");
     # 首先建立基本的簽名
    signature = "question -> answer"
    llmobj = dspy.Predict(signature)
    # 使用 Template 字符串
    
        # 實現標籤搜尋的邏輯
    # prompt_template = """
    #     Role: 您是一個表格生成專家，擅長將結構化數據轉換為規範的Markdown表格
    #     Task: 根據提供的數據生成多行Markdown表格，需滿足以下格式要求：

    #     | 模組 | 嚴重度 | 問題現象描述 | 原因分析 | 改善對策 | 經驗萃取 | 評審後優化 | 評分 |
    #     |------|:------:|--------------|---------|----------|----------|------------|:----:|
    #     {% for item in items %}
    #     | {{ item.module }} | {{ item.severity }} | {{ item.description }} | {{ item.cause }} | {{ item.improve }} | {{ item.experience }} | {{ item.judge }} | {{ item.score }} |
    #     {% endfor %}

    #     嚴格要求：
    #     1. 必須包含完整的表頭，且順序不可變更
    #     2. 每行數據需嚴格對應表頭欄位
    #     3. 若數值為空則填入「暫無數據」
    #     4. 使用<br>符號處理欄位內的換行
    #     5. 評分欄位必須是1-5的整數，若無則標示「待評分」
    #     6. 嚴格保持表格對齊格式
    #     7. 根據以下JSON數據生成{{ row_count }}行：
    #     {{ data_sample }}
    #     """
    # sample_data = {
    #     "items": [{
    #         "module": cleaned_messages['module'],
    #         "severity": cleaned_messages['severity'],
    #         "description": cleaned_messages['description'],
    #         "cause": cleaned_messages['cause'],
    #         "improve": cleaned_messages['improve'],
    #         "experience": cleaned_messages['experience'],
    #         "judge": cleaned_messages['judge'],
    #         "score": cleaned_messages['score'] or "待評分"
    #     }]
    # }
    # promptPatternStr = prompt_template.format(**cleaned_messages)
    # 
    # promptPatternStr = prompt_template.format(
    #     row_count=len(sample_data['items']),
    #     data_sample=json.dumps(sample_data, ensure_ascii=False),
    #     **{k: v or "暫無數據" for k, v in cleaned_messages.items()}
    # )
    # print(f"*******************\n{promptPatternStr}\n***********************");
    # 創建預測對象並返回結果
    # first_lvl_response = llmobj(question=promptPatternStr)
    
    # responses.append(first_lvl_response.answer);
    
    """ connect to stackexchange"""
    # responses = []
    # raw_ret_msg = stackexchange.run(cleaned_messages['description']);
    # processed_ret_msg = format_qa_content(raw_ret_msg)
    # if(len(processed_ret_msg) > 6):
    #     responses.append("StackExchange上類似的問題：\n"+processed_ret_msg);
    # else:
    #     responses.append("StackExchange上類似的問題：无");
    # format_response = "\n\n".join(responses)

    """google serper api"""
    # headers = {
    #     'X-API-KEY': SERPER_KEY,
    #     'Content-Type': 'application/json'
    # }
    # payload = json.dumps({
    #     "q": message
    # })
    # googleserper.request("POST", "/search", payload, headers)
    # res = googleserper.getresponse()
    # gs_data= res.read().decode("utf-8")
    # # print(data.decode("utf-8"))
    # data_dict = eval(gs_data)
    # gs_content = data_dict["organic"]
    # gs_responses = [];
    # gs_row_count = 1;
    # for c in gs_content:
    #     gs_responses.append(f"{gs_row_count}.標題:{c['title']}<br> 連結:<href>{c['link']}<br> 描述:{c['snippet']}<br>");
    #     gs_row_count += 1;
    
    # if len(gs_responses) < 0:
    #      gs_responses.append("外部資料-google serper:無資料！");

    """make responses and return"""
    ret_dict = {
        'primary_msg':"",#first_lvl_response.answer,
        # 'stackexchangemsg': format_response,
        # 'googleserper': gs_responses,
        'status_code':200
    }
    return ret_dict       
#     except Exception as e:
#         raise RuntimeError(f"Error : {str(e)}")


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
        faiss_retriever = CustomFAISSRetriever(faiss_index_path=faiss_index_path, vector_db_path="nodata", model_name=encoding_model_name, k=1); # 初始化 faiss 檢索物件
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
        "description": replace_chinese_punctuation(str(df_row['问题现象描述'])),
        "module": replace_chinese_punctuation(str(df_row['模块'])),
        "severity": replace_chinese_punctuation(str(df_row['严重度'])),
        "cause": replace_chinese_punctuation(str(df_row['原因分析'])),
        "improve": replace_chinese_punctuation(str(df_row['改善对策'])),
        "experience": replace_chinese_punctuation(str(df_row['经验萃取'])),
        "judge": replace_chinese_punctuation(str(df_row['评审后优化'])),
        "score": replace_chinese_punctuation(str(df_row['评分']))
    }

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
        print(f"---------------search_action:{search_action}--------------");
        message = data.get("message");
        if search_action == 1:
            _pos, _distances = search_similar_questions(faiss_retriever, message);
            print(f"_pos:{_pos}\n_distances:{_distances}");
            max_pos = np.amax(_pos[0]);
            min_distance = np.amin(_distances[0]);
            min_distance_index = np.argmin(_distances[0]);
            print(f"most min distance:{min_distance}")
            message = replace_chinese_punctuation(message);
            if min_distance < 11:
                submessages = getSubMessages(dfObj.iloc[max_pos])
            else:
                #"没有任何匹配的资料"
                submessages="nodata";
        else:
            _pos_qsrc, _distances_qsrc = search_similar_questions(faiss_retriever_qsrc, message);
            _pos_module, _distance_module =  search_similar_questions(faiss_retriever_module,message);
            filtered_pos_qsrc = [];
            for i in range(len(_distances_qsrc[0])):
                _d = _distances_qsrc[0][i];
                if _d < 0.5:
                    filtered_pos_qsrc.append(_pos_qsrc[i])
            filtered_pos_module=[]
            for j in range(len(_distance_module[0])):
                _d = _distance_module[0][j]
                if _d < 0.5:
                    filtered_pos_module.append(_pos_module[j])
            message = replace_chinese_punctuation(message);
            submessages = []
            if len(filtered_pos_qsrc) > 0:
                for item in filtered_pos_qsrc:
                    submessages.append(getSubMessages(dfObj_aitrial.iloc[item]));
            if len(filtered_pos_module) > 0:
                for item in filtered_pos_module:
                    submessages.append(getSubMessages(dfObj_aitrial.iloc[item]));
            print(f"search action:{search_action}, and len of submessages:{len(submessages)}");
            if len(submessages) < 1:
                submessages.append("nodata");
        
        # call generate function
        ai_response = None;
        if(submessages!="nodata"):
            if search_action == 1:
                ai_response = await generate(message=message, submessages=submessages, search_action=search_action);
            else:
                ai_response = await generate_multirows(message=message, submessages=submessages, search_action=search_action)
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
