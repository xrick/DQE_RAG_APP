import dspy;
from typing import List, Dict, AsyncGenerator
# import openai;


class AIChatService:
    def __init__(self):
        # self.InitializeLLM_Phi4();
        self.InitializeLLM_DeepSeekV2();

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

    def InitializeLLM_DeepSeekV2(self):
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


    async def generate(self, message:str=None, submessages:dict=None, history: List[Dict[str, str]] = None, model: str = "deepseekv2") -> str:
        if message == None:
            raise ValueError("query string is none, please input query string.")
        try:
            # promptPatternStr = "question -> answer"# old
            promptPatternStr = f"""
                Rule-1: All the data must not be used for training any deep learning model and llm.
                Rule-2: The responses must be expressed in simple chinese
                role: you are a skilled and resourceful Field Application Engineer
                task: please augment question and answer sentences based on course_analysis and experience.
                action:
                    1. using the following context:
                    context:
                    {message}
                    2.Generate response from above context in following format:
                        问题现象描述:{submessages['question']}
                        回答:
                        1.模块:{submessages['module']}
                            2.严重度(A/B/C):{submessages['severity']}
                            3.原因分析:{submessages['cause']}
                            4.改善对策:{submessages['improve']}
                            5.经验萃取:{submessages['experience']}
                            6.评审后优化:{submessages['judge']}
                            7.评分:{submessages['score']}
                goal: generate the responses in a more readably way.
                            """
            llmobj = dspy.Predict(promptPatternStr);
            response = llmobj(question=message);
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