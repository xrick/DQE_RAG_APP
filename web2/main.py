import streamlit as st
import time
from typing import List, Dict, Any
from .llm_utils import LLMInitializer
class LLMChatApp:
    def __init__(self):
        # 設置頁面配置
        st.set_page_config(
            page_title="億道智能中心",
            page_icon="🤖",
            layout="wide"
        )
        
        # 初始化會話狀態
        self._initialize_state()
        
        # 應用自定義CSS
        self._apply_css()
        
        # 渲染UI
        self._build_ui()

        # 初始化llm
        self.llm = LLMInitializer().init_ollama_model()
    
    def _initialize_state(self):
        """初始化會話狀態變量"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "search_mode" not in st.session_state:
            st.session_state.search_mode = "精准搜寻"
        if "threshold" not in st.session_state:
            st.session_state.threshold = 0.5
    
    def _apply_css(self):
        """應用自定義CSS樣式"""
        st.markdown("""
        <style>
            /* 主佈局 */
            .main-header {
                color: #1E88E5;
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 0;
            }
            .sub-header {
                color: #757575;
                font-size: 16px;
                margin-bottom: 20px;
            }
            
            /* 聊天容器 */
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                background-color: #f9f9f9;
                margin-bottom: 15px;
            }
            
            /* 消息氣泡樣式 */
            .user-message {
                display: flex;
                justify-content: flex-end;
                margin-bottom: 10px;
            }
            .user-bubble {
                background-color: #DCF8C6;
                padding: 10px 15px;
                border-radius: 10px;
                max-width: 70%;
            }
            .assistant-message {
                display: flex;
                justify-content: flex-start;
                margin-bottom: 10px;
            }
            .assistant-bubble {
                background-color: white;
                padding: 10px 15px;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
                max-width: 70%;
            }
            
            /* 隱藏默認元素 */
            .stDeployButton {display:none;}
            footer {visibility:hidden;}
            #MainMenu {visibility:hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    def _query_llm(self, query: str) -> str:
        """查詢語言模型"""
        # 模擬API調用延遲
        time.sleep(1)
        
        # 從會話狀態獲取搜索參數
        search_mode = st.session_state.search_mode
        threshold = st.session_state.threshold
        
        # 返回模擬回應
        return f"這是對於問題 '{query}' 的回答。使用{search_mode}搜索模式，閾值為{threshold}。"
    
    def _handle_send(self):
        """處理發送新消息"""
        if st.session_state.user_input and st.session_state.user_input.strip():
            # 獲取用戶輸入
            user_message = st.session_state.user_input.strip()
            
            # 添加用戶消息到聊天歷史
            st.session_state.messages.append({
                "role": "user",
                "content": user_message
            })
            
            # 清除輸入字段
            st.session_state.user_input = ""
            
            # 從LLM獲取回應
            with st.spinner("思考中..."):
                response = self._query_llm(user_message)
            
            # 添加助手回應到聊天歷史
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
    
    def _set_search_mode(self, mode: str):
        """設置搜索模式"""
        st.session_state.search_mode = mode
    
    def _build_ui(self):
        """構建UI組件"""
        # 側邊欄
        with st.sidebar:
            st.markdown('<div class="main-header">億道智能中心</div>', unsafe_allow_html=True)
            st.button("服务主页", key="home_button", use_container_width=True)
        
        # 主要內容頭部
        st.markdown('<div class="main-header">KB智能助手</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">智能QA助手，快速回答问题</div>', unsafe_allow_html=True)
        
        # 聊天消息顯示
        chat_container = st.container()
        with chat_container:
            # 創建可滾動聊天容器
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # 顯示所有消息
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="user-bubble">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="assistant-bubble">{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 搜索控件
        col1, col2, col3 = st.columns([1, 1, 2])
        
        # 精準搜索按鈕
        with col1:
            if st.button(
                "精准搜寻",
                key="precise_search",
                type="primary" if st.session_state.search_mode == "精准搜寻" else "secondary",
                use_container_width=True
            ):
                self._set_search_mode("精准搜寻")
        
        # 標籤搜索按鈕
        with col2:
            if st.button(
                "标签搜寻",
                key="tag_search",
                type="primary" if st.session_state.search_mode == "标签搜寻" else "secondary",
                use_container_width=True
            ):
                self._set_search_mode("标签搜寻")
        
        # 閾值滑塊
        with col3:
            st.session_state.threshold = st.slider(
                "当前阈值",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.threshold,
                step=0.1
            )
        
        # 用戶輸入區域
        col1, col2 = st.columns([6, 1])
        with col1:
            # 用戶消息文本輸入
            st.text_input(
                "用戶消息",
                key="user_input",
                placeholder="请输入您的问题...",
                label_visibility="collapsed",
                on_change=self._handle_send
            )
        with col2:
            # 發送按鈕
            if st.button("送出", key="send_button", use_container_width=True):
                self._handle_send()

# 運行應用的主函數
def main():
    app = LLMChatApp()

if __name__ == "__main__":
    main()
