document.addEventListener("DOMContentLoaded", function () {
    const chatContainer = document.getElementById("chat-container");

    // 使用事件委派處理按鈕點擊和回車鍵
    chatContainer.addEventListener("click", function (event) {
        if (event.target.id === "send-button") {
            sendMessage();
        }
    });

    chatContainer.addEventListener("keydown", function (event) {
        if (event.target.id === "user-message" && event.key === "Enter") {
            event.preventDefault(); // 阻止默認表單提交行為
            sendMessage();
        }
    });

    document.addEventListener("DOMContentLoaded", function () {
        const sendButton = document.getElementById("send-button");
    
        if (!sendButton) {
            console.error("找不到發送按鈕！");
            return;
        }
    
        sendButton.addEventListener("click", function () {
            console.log("發送按鈕被點擊！");
        });
    });

    // 發送消息的函數
    function sendMessage() {
        const userInput = document.getElementById("user-message");
        const messagesDiv = document.getElementById("messages");
        const userMessage = userInput.value.trim();

        if (userMessage === "") {
            alert("請輸入消息！");
            return;
        }

        // 發送消息到後端
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
                // 顯示用戶消息
                const userMsgDiv = document.createElement("div");
                userMsgDiv.textContent = `你: ${userMessage}`;
                messagesDiv.appendChild(userMsgDiv);

                // 顯示 AI 回覆
                const aiMsgDiv = document.createElement("div");
                aiMsgDiv.textContent = `AI: ${data.response}`;
                messagesDiv.appendChild(aiMsgDiv);

                // 清空輸入框
                userInput.value = "";
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("發送消息時出錯，請稍後再試！");
            });
    }
});
