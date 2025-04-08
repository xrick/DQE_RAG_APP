// ai-chat.js
(function () {
    console.log("ai-chat.js 加載完成！");
    console.log("it has invoke！");
    // 定義初始化聊天介面的函數
    function initializeChatInterface() {
        const maxRetries = 10;
        let retryCount = 0;
    
        function tryInitialize() {
            console.log("嘗試初始化聊天介面...");
            // const chatContainer = document.getElementById("chat-container");
            // const messageInput = document.getElementById("message-input");
            // const sendButton = document.getElementById("send-button");
            const chatContainer = document.getElementById("preview");
            const messageInput = document.getElementById("editor");
            const sendButton = document.getElementById("submit");
            

            if (!chatContainer || !messageInput || !sendButton) {
                if (retryCount < maxRetries) {
                    retryCount++;
                    console.log(`重試 ${retryCount}/${maxRetries}...`);
                    setTimeout(tryInitialize, 100);
                    return;
                }
                console.error("初始化失敗：無法找到必要元素");
                return;
            }
    
            // 綁定發送按鈕事件
            sendButton.addEventListener("click", function() {
                const message = messageInput.value.trim();
                if (message) {
                    sendMessage(message);
                }
            });
    
            // 綁定輸入框事件
            messageInput.addEventListener("keydown", function(event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    const message = messageInput.value.trim();
                    if (message) {
                        sendMessage(message);
                    }
                }
            });
    
            console.log("聊天介面初始化完成！");
        }
    
        tryInitialize();
    }
    

    // 將函數掛載到全局作用域
    window.initializeChatInterface = initializeChatInterface;

    // 發送消息的輔助函數
    function sendMessage(message) {
        const chatContainer = document.getElementById("preview");
        const messageInput = document.getElementById("editor");

        if (!chatContainer || !messageInput) {
            console.error("聊天框或輸入框未找到！");
            return;
        }

        // 顯示用戶消息到聊天框
        appendMessage("user", message);

        // 清空輸入框
        messageInput.value = "";

        // 模擬向後端發送請求
        console.log("正在向後端發送請求...");
        fetch("/api/ai-chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message }),
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then((data) => {
            console.log("後端回應：", data);
            if (data && data.response) {
                // appendMessage("ai", `${JSON.stringify(data.response)}`);
                appendMessage("ai:", data.response['primary_msg'],'markdown')
                // appendMessage("similar questions:", data.response['googleserper'],'markdown')
                // appendMessage("similar questions:", data.response['googleserper'],'puretxt')
            } else {
                appendMessage("error", "後端未返回有效回應！");
            }
        })
        .catch((error) => {
            console.error("請求失敗詳情:", error);
            appendMessage("error", `連接失敗: ${error.message}`);
        });
    }

    // 添加消息到聊天框的輔助函數
    // 配置 marked.js 使用 highlight.js
    // for markdown table
    // marked.setOptions({
    //     highlight: function(code, lang) {
    //         if (lang && hljs.getLanguage(lang)) {
    //             return hljs.highlight(code, { language: lang }).value;
    //         }
    //         return hljs.highlightAuto(code).value;
    //     },
    //     breaks: true,
    //     gfm: true,
    //     tables: true,  // 啟用表格支持
    //     pedantic: false,
    //     sanitize: false
    // });

    marked.setOptions({
        gfm: true,
        breaks: true,
        tables: true,
        pedantic: false,
        sanitize: false,
        smartLists: true,
        smartypants: false,
        xhtml: false
    });

    function appendMessage(role, text, act) {
        const chatContainer = document.getElementById("preview");
        
        if (!chatContainer) {
            console.error("找不到聊天容器！");
            return;
        }
    
        const messageWrapper = document.createElement("div");
        messageWrapper.className = "w-full flex mb-4 " + 
            (role === "user" ? "justify-end" : "justify-start");
    
        const messageDiv = document.createElement("div");
        messageDiv.className = `max-w-[100%] p-4 rounded-lg ${
            role === "user" 
                ? "bg-blue-500 text-white ml-auto rounded-br-none" 
                : "bg-gray-100 text-gray-800 mr-auto rounded-bl-none"
        }`;
    
        const contentDiv = document.createElement("div");
        contentDiv.className = "markdown-content prose " + 
            (role === "user" ? "prose-invert" : "");
        
        try {
            if(act === 'markdown') {
                // 使用 marked 解析 Markdown
                contentDiv.innerHTML = marked.parse(text);
            } else {
                contentDiv.textContent = text;
            }
            
            messageDiv.appendChild(contentDiv);
            messageWrapper.appendChild(messageDiv);
            chatContainer.appendChild(messageWrapper);
            
            // 滾動到底部
            chatContainer.scrollTop = chatContainer.scrollHeight;
        } catch (error) {
            console.error("渲染訊息時發生錯誤：", error);
            contentDiv.textContent = text;
        }
    }

})();
