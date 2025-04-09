document.addEventListener("DOMContentLoaded", function () {
    const rootEle = document.getElementById('root')
    const chatSection = document.getElementById('chat-section');
    const chatContainer = document.getElementById("chat-container");
    const messageInput = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    // 確認必要的 DOM 元素是否存在
    // if (!chatContainer || !messageInput || !sendButton) {
    //     console.error("必要的 DOM 元素未找到：#chat-container, #message-input 或 #send-button");
    //     return;
    // }
    if(!rootEle){
        console.error("rootEle is not found");
    }
    
    // const subElements = rootEle.children;
    // // Loop through each sub-element
    // for (let i = 0; i < subElements.length; i++) {
    //     console.log(subElements[i]);
    // }

    if(!chatSection){
        console.error("chatSection is not found");
    }
    if(!chatContainer){
        console.error("chatContainer is not found");
    }
    if(!messageInput){
        console.error("messageInput is not found");
    }
    if(!sendButton){
        console.error("sendButton is not found");
    }else{
        console.log("sendButton is found")
    }
    /* 
    output all the elements
    */
   // Get all HTML elements
    const allElements = document.getElementsByTagName("*");

    // Loop through each element
    for (let i = 0; i < allElements.length; i++) {
        console.log(allElements[i]);
    }
    // 定義 initializeChatInterface 函數
    // 初始化聊天介面
    function initializeChatInterface() {
        console.log("初始化聊天介面...");
        // 綁定發送按鈕點擊事件
        sendButton.addEventListener("click", function () {
            const message = messageInput.value.trim();
            if (message) {
                console.log("用戶輸入的訊息：", message);
                sendMessage(message);
            }
        });
    }

    // 調用初始化函數
    initializeChatInterface();

    // 測試按鈕是否被綁定
    if (!sendButton) {
        console.error("找不到發送按鈕！");
        return;
    }

    // sendButton.addEventListener("click", sendMessage);

    function sendMessage() {
        const userMessage = messageInput.value.trim();

        if (userMessage === "") {
            console.log("請輸入消息！");
            return;
        }

        // 測試輸入的消息
        console.log("用戶消息：", userMessage);

        // 顯示用戶消息到聊天框
        appendMessage("user", userMessage);

        // 清空輸入框
        messageInput.value = "";

        // 模擬向後端發送請求
        console.log("正在向後端發送請求...");
        fetch("/api/ai-chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message: userMessage }),
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

    function appendMessage(role, text) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("p-3", "max-w-xs", "rounded-lg", "shadow", "text-sm");

        if (role === "user") {
            messageDiv.classList.add(
                "bg-blue-500",
                "text-white",
                "self-end",
                "rounded-br-none",
                "text-right"
            );
            messageDiv.textContent = text;
        } else if (role === "ai") {
            messageDiv.classList.add(
                "bg-gray-200",
                "text-gray-800",
                "self-start",
                "rounded-bl-none",
                "text-left"
            );
            messageDiv.textContent = text;
        } else if (role === "error") {
            messageDiv.classList.add(
                "bg-red-100",
                "text-red-500",
                "text-center",
                "font-semibold"
            );
            messageDiv.textContent = text;
        }

        chatContainer.appendChild(messageDiv);

        // 滾動到底部
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
});
