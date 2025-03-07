# service_manager.py
from openai import AsyncOpenAI
import base64

class ServiceManager:
    async def transcribe_audio(self, audio_path: str) -> str:
        """使用 Whisper API 將音頻轉換為文字"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = await self.llm.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="zh"
                )
                return transcript.text
        except Exception as e:
            logger.error(f"Audio transcription failed: {str(e)}")
            raise e

    async def handle_ai_chat(self, message: str = None, images: list = None, audio_text: str = None) -> str:
        """處理 AI 聊天請求，支持文字、圖片和音頻輸入"""
        try:
            messages = []
            
            # 系統提示詞
            messages.append({
                "role": "system",
                "content": self.prompts[ServiceType.AI_CHAT]
            })

            # 組合用戶輸入
            user_message = []
            
            # 添加文字訊息
            if message:
                user_message.append({"type": "text", "text": message})

            # 添加音頻轉換的文字
            if audio_text:
                user_message.append({"type": "text", "text": f"[Audio Transcript]: {audio_text}"})

            # 添加圖片
            if images:
                for image in images:
                    user_message.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    })

            messages.append({
                "role": "user",
                "content": user_message
            })

            # 調用 GPT-4V API
            response = await self.llm.chat.completions.create(
                model="gpt-4-vision-preview" if images else "gpt-4",
                messages=messages,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"AI Chat error: {str(e)}")
            raise e

###########################################################################################
# service_manager.py
import re
from datetime import datetime
import uuid

class ServiceManager:
    async def handle_ai_chat(self, message: str = None, images: list = None, audio_text: str = None) -> List[MessageContent]:
        try:
            # ... OpenAI API 調用邏輯保持不變 ...

            # 解析 AI 回應，轉換為結構化格式
            raw_response = response.choices[0].message.content
            structured_contents = []

            # 保存用戶的輸入內容
            if message:
                structured_contents.append(MessageContent(
                    type="text",
                    content=message
                ))

            if images:
                for idx, image in enumerate(images):
                    structured_contents.append(MessageContent(
                        type="image",
                        content=image,
                        metadata={
                            "index": idx,
                            "mime_type": "image/jpeg"
                        }
                    ))

            if audio_text:
                structured_contents.append(MessageContent(
                    type="audio_transcript",
                    content=audio_text
                ))

            # 解析 AI 回應中的特殊格式
            # 例如：代碼塊、表格等
            segments = self._parse_response(raw_response)
            structured_contents.extend(segments)

            return structured_contents

    def _parse_response(self, raw_response: str) -> List[MessageContent]:
        """解析回應文本，識別特殊格式"""
        segments = []
        
        # 使用正則表達式識別代碼塊
        code_pattern = r"```(.*?)```"
        text_parts = re.split(code_pattern, raw_response)
        
        for i, part in enumerate(text_parts):
            if i % 2 == 0:  # 普通文本
                if part.strip():
                    segments.append(MessageContent(
                        type="text",
                        content=part.strip()
                    ))
            else:  # 代碼塊
                segments.append(MessageContent(
                    type="code",
                    content=part.strip(),
                    metadata={
                        "language": self._detect_language(part)
                    }
                ))
        
        return segments

    def _detect_language(self, code: str) -> str:
        """檢測代碼語言"""
        # 簡單的語言檢測邏輯
        return "python"  # 這裡可以實現更複雜的語言檢測