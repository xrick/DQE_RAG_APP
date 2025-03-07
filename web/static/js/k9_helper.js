// k9_helper.js
(function () {
    console.log("k9_helper.js 加載完成！");

    function initializeK9HelperInterface() {
        console.log("初始化K9商城小幫手介面...");

        const chatContainer = document.getElementById("chat-container");
        const messageInput = document.getElementById("message-input");
        const sendButton = document.getElementById("send-button");

        if (!chatContainer || !messageInput || !sendButton) {
            console.error("K9商城小幫手介面的必要 DOM 元素未找到，稍後重試...");
            return;
        }

        sendButton.addEventListener("click", function () {
            const message = messageInput.value.trim();
            if (message) {
                console.log("用戶輸入的訊息：", message);
                sendMessage(message);
            }
        });

        messageInput.addEventListener("keydown", function (event) {
            const message = messageInput.value.trim();
            if (event.key === "Enter") {
                event.preventDefault();
                sendMessage(message);
            }
        });
    }

    window.initializeK9HelperInterface = initializeK9HelperInterface;

    function sendMessage(message) {
        const chatContainer = document.getElementById("chat-container");
        const messageInput = document.getElementById("message-input");

        if (!chatContainer || !messageInput) {
            console.error("聊天框或輸入框未找到！");
            return;
        }

        appendMessage("user", message);
        messageInput.value = "";

        console.log("正在向後端發送請求...");
        fetch("/api/k9-helper", {
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
                appendMessage("error", "無法連接到商城助手服務，請稍後再試！");
            });
    }

    function appendMessage(role, text) {
        const chatContainer = document.getElementById("chat-container");
        if (!chatContainer) {
            console.error("找不到聊天容器！");
            return;
        }

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
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
})();
