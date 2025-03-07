import dspy;
from typing import List, Dict, AsyncGenerator
import openai;


class AIChatService:
    def __init__(self):
        # self.InitializeLLM_Phi4();
        self.InitializeLLM_Llama();

    def InitializeLLM_DeepSeekR1(self):
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
        
    def InitializeLLM_Llama(self):
        lm = dspy.LM('ollama_chat/llama3.2:latest', api_base='http://localhost:11434')
        dspy.configure(lm=lm)

    def InitializeLLM_Phi4(self):
        local_config = {
            "api_base": "http://localhost:11434/",  
            "api_key": "NULL",  # 特殊標記用於跳過驗證
            "model": "phi4:14b",
            "custom_llm_provider":"microsoft"
        }
        dspy.configure(
            lm=dspy.LM(
                **local_config
            )
        )


    async def generate(self, message: str, history: List[Dict[str, str]] = None, model: str = "gpt-4") -> str:
        if message == None:
            raise ValueError("query string is none, please input query string.")
        try:
            promptPatternStr = "question -> answer"
            qa = dspy.Predict(promptPatternStr);
            response = qa(question=message);
            return response.answer;
        except Exception as e:
            raise RuntimeError(f"Error : {e}")


    async def generate_stream(self, message: str, history: List[Dict[str, str]] = None, model: str = "gpt-4") -> AsyncGenerator[str, None]:
        try:
            # 構建消息列表
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": message})
            # 調用流式 ChatCompletion API
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=True,  # 啟用流式回應
            )

            # 流式回應生成器
            async for chunk in response:
                yield chunk["choices"][0]["delta"]["content"]
        except Exception as e:
            raise ValueError(f"Error generating By Hand: {e}");