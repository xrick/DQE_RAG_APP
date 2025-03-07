// ai-chat.js
(function () {
    console.log("ai-chat.js 加載完成！");

    // 定義初始化聊天介面的函數
    function initializeChatInterface() {
        const maxRetries = 10;
        let retryCount = 0;
    
        function tryInitialize() {
            console.log("嘗試初始化聊天介面...");
            const chatContainer = document.getElementById("chat-container");
            const messageInput = document.getElementById("message-input");
            const sendButton = document.getElementById("send-button");
    
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
    
    // function initializeChatInterface() {
    //     console.log("初始化聊天介面...");

    //     // 確保 DOM 元素存在
    //     const chatContainer = document.getElementById("chat-container");
    //     const messageInput = document.getElementById("message-input");
    //     const sendButton = document.getElementById("send-button");

    //     if (!chatContainer || !messageInput || !sendButton) {
    //         console.error("聊天介面的必要 DOM 元素未找到，稍後重試...");
    //         return;
    //     }

    //     // 綁定發送按鈕點擊事件
    //     sendButton.addEventListener("click", function () {
    //         const message = messageInput.value.trim();
    //         if (message) {
    //             console.log("用戶輸入的訊息：", message);
    //             sendMessage(message);
    //         }
    //     });

    //     // 綁定輸入框的 [Enter] 鍵事件
    //     messageInput.addEventListener("keydown", function (event) {
    //         const message = messageInput.value.trim();
    //         if (event.key === "Enter") {
    //             event.preventDefault(); // 防止回車換行
    //             sendMessage(message);
    //         }
    //     });

    // }

    // 將函數掛載到全局作用域
    window.initializeChatInterface = initializeChatInterface;

    // 發送消息的輔助函數
    function sendMessage(message) {
        const chatContainer = document.getElementById("chat-container");
        const messageInput = document.getElementById("message-input");

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
                    appendMessage("ai", data.response);
                } else {
                    appendMessage("error", "後端未返回有效回應！");
                }
            })
            .catch((error) => {
                console.error("請求失敗：", error);
                appendMessage("error", "無法連接到後端，請稍後再試！");
            });
    }

    // 添加消息到聊天框的輔助函數
    // 配置 marked.js 使用 highlight.js
    marked.setOptions({
        highlight: function(code, lang) {
            if (lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return hljs.highlightAuto(code).value;
        },
        breaks: true,
        gfm: true
    });

    function appendMessage(role, text) {
        const chatContainer = document.getElementById("chat-container");
        if (!chatContainer) {
            console.error("找不到聊天容器！");
            return;
        }

        const messageWrapper = document.createElement("div");
        messageWrapper.className = "w-full flex mb-4 " + 
            (role === "user" ? "justify-end" : "justify-start");

        const messageDiv = document.createElement("div");
        messageDiv.className = `max-w-[80%] p-4 rounded-lg ${
            role === "user" 
                ? "bg-blue-500 text-white ml-auto rounded-br-none" 
                : "bg-gray-100 text-gray-800 mr-auto rounded-bl-none"
        }`;

        const contentDiv = document.createElement("div");
        contentDiv.className = "markdown-content prose " + 
            (role === "user" ? "prose-invert" : "");
        
        // 使用 marked 解析 Markdown
        contentDiv.innerHTML = marked.parse(text);

        messageDiv.appendChild(contentDiv);
        messageWrapper.appendChild(messageDiv);
        chatContainer.appendChild(messageWrapper);
        
        // 對新添加的代碼塊應用語法高亮
        messageDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    
    

    // function appendMessage(role, text) {
    //     const chatContainer = document.getElementById("chat-container");
    //     if (!chatContainer) {
    //         console.error("找不到聊天容器！");
    //         return;
    //     }

    //     const messageDiv = document.createElement("div");
    //     messageDiv.classList.add("p-3", "max-w-xs", "rounded-lg", "shadow", "text-sm");

    //     if (role === "user") {
    //         messageDiv.classList.add(
    //             "bg-blue-500",
    //             "text-white",
    //             "self-end",
    //             "rounded-br-none",
    //             "text-right"
    //         );
    //         messageDiv.textContent = text;
    //     } else if (role === "ai") {
    //         messageDiv.classList.add(
    //             "bg-gray-200",
    //             "text-gray-800",
    //             "self-start",
    //             "rounded-bl-none",
    //             "text-left"
    //         );
    //         messageDiv.textContent = text;
    //     } else if (role === "error") {
    //         messageDiv.classList.add(
    //             "bg-red-100",
    //             "text-red-500",
    //             "text-center",
    //             "font-semibold"
    //         );
    //         messageDiv.textContent = text;
    //     }

    //     chatContainer.appendChild(messageDiv);

    //     // 滾動到底部
    //     chatContainer.scrollTop = chatContainer.scrollHeight;
    // }
})();
