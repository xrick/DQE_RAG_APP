import os
import dspy
from typing import List, Dict, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.ai_chat_service import AIChatService
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
c
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
vector_db_path = os.getenv("FAISS_DB_PATH")
logging.info(f"Encoding model: {encoding_model_name}");
logging.info(f"FAISS index path: {faiss_index_path}");
logging.info(f"Vector DB path: {vector_db_path}");

################## LLM Initialization ##################

def InitializeLLM_DeepSeekV2():
        local_config = {
            "api_base": "http://localhost:11434/v1",  # 注意需加/v1路徑
            "api_key": "NULL",  # 特殊標記用於跳過驗證
            "model": "deepseek-v2",
            "custom_llm_provider":"deepseek"
        }
        dspy.configure(
            lm=dspy.LM(
                **local_config
            )
        )
        print("DeepSeek-V2 has initialized!")

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

async def generate(message: str = None, submessages: dict = None, history: List[Dict[str, str]] = None, model: str = "deepseekv2") -> str:
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
    global stackexchange;
    global SERPER_KEY;
    global googleserper;
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
    # Rule-6:Please generate the responses using makrdown format
    prompt_template = """
        Role:You are a sentences refinement expert and good at repolish sentences.
        Rules:
            Rule-1: All the data must not be used for training any deep learning model and llm.
            Rule-2: The responses must be expressed in simple chinese
            Rule-3: Generate the responses in a more readably way.
        questions:
        Please refine and repolish the following description and answer sentences based on course_analysis and experience.
        Please strictly follow the following format to generate the responses as markdown sentences.
        
        **A.问题现象描述:**.   
            {description}. 
        **B.回答:**. 
            **1.模块:**. 
                {module}. 
            **2.严重度(A/B/C):**. 
                {severity}. 
            **3.原因分析:**. 
                {cause}.  
            **4.改善对策:**. 
                {improve}. 
            **5.经验萃取:**. 
                {experience}. 
            **6.评审后优化:**. 
                {judge}. 
            **7.评分:**. 
                {score}. 
        """
    # 使用 format 方法安全地插入值
    # promptPatternStr = prompt_template.format(**cleaned_messages)
    promptPatternStr = prompt_template.format(description=cleaned_messages['description'],
                                              module=cleaned_messages['module'],
                                              severity=cleaned_messages['severity'],
                                              cause=cleaned_messages['cause'],
                                              improve=cleaned_messages['improve'],
                                              experience=cleaned_messages['experience'],
                                              judge=cleaned_messages['judge'],
                                              score=cleaned_messages['score'],
                                              )
    # 創建預測對象並返回結果
    first_lvl_response = llmobj(question=promptPatternStr)
    
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
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "q": clean_text['description']
    })
    googleserper.request("POST", "/search", payload, headers)
    res = googleserper.getresponse()
    gs_data= res.read()
    # print(data.decode("utf-8"))
    gs_data = gs_data.decode("utf-8")
    # data_dict = json.load(data.replace("'", '"'))
    data_dict = eval(gs_data)
    gs_content = data_dict["organic"]
    gs_responses = [];
    gs_row_count = 1;
    for c in gs_content:
        gs_responses.append(f"{gs_row_count}.標題:{c['title']}\n 連結:{c['link']}\n 描述:{c['snippet']}\n");

        
    """make responses and return"""
    ret_dict = {
        'primary_msg':first_lvl_response.answer,
        # 'stackexchangemsg': format_response,
        'gooserper': gs_responses,
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
    global faiss_retriever
    global dfObj;
    try:
        logging.info("start to initialize services.......");
        # ai_chat_service = AIChatService(); # 初始化 KB聊天物件
        print(f"vector db path: {vector_db_path}");
        faiss_retriever = CustomFAISSRetriever(faiss_index_path=faiss_index_path, vector_db_path=vector_db_path, model_name=encoding_model_name); # 初始化 faiss 檢索物件
        dfObj = pd.read_csv('./db/raw/deq_learn_refine2_correct.csv', encoding='utf-8-sig'); # 初始化 faiss 檢索物件
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
    

def sanitize_text(text):
    if pd.isna(text):
        return ""
    
    # 基本清理
    text = str(text).strip()
    
    # 移除換行符
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 替換中文標點
    punctuation_map = {
        '，': ',', '。': '.', '：': ':', '；': ';',
        '"': '"', '"': '"', ''': "'", ''': "'",
        '！': '!', '？': '?', '（': '(', '）': ')',
        '【': '[', '】': ']', '、': ',', '「': '"',
        '」': '"', '『': "'", '』': "'"
    }
    for ch, en in punctuation_map.items():
        text = text.replace(ch, en)
    
    # 轉義引號和其他特殊字符
    text = text.replace('"', '\\"').replace("'", "\\'")
    
    return text


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
        '』': "'"
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
        data = await request.json()
        _pos, _distances = search_similar_questions(faiss_retriever, data.get("message"))
        max_pos = np.amax(_pos[0]);
        print(max_pos);
        # for i in _pos[0]:
        # if _distances[0][0] > 0.5:
        #     ai_response = "您好，无法找到相似问题，请重新输入。"
        #     return {"response": ai_response};
        # # for i in _pos[0]:
        # if _distances[0][0] < 0.5:
        message = dfObj.iloc[max_pos].to_string();#data.get("message")
        message = replace_chinese_punctuation(message);
        # print(message);
        _submessages = getSubMessages(dfObj.iloc[max_pos]);
        # _submessages = replace_chinese_punctuation(str(_submessages));

        # history = data.get("history", [])
        # if not message:
        #     raise HTTPException(status_code=400, detail="Message is required")
        # 調用 AIChatService 的生成方法
        
        ai_response = await generate(message=message, submessages=_submessages);
        print(type(ai_response))
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
