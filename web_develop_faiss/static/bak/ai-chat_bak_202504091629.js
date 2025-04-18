// ai-chat.js
// 在 ai-chat.js 開頭添加
let loadingAnimationTimer = null;
let currentLoadingText = '';
const loadingText = '搜尋中......';

(function () {
    console.log("ai-chat.js 加載完成！");

    // 添加搜尋模式變數和控制邏輯
    let search_action = 1; // 預設為精準搜尋
    // 定義初始化聊天介面的函數
    let search_threshold = 25;
    function initializeSearchButtons() {
        const preciseButton = document.getElementById('precise-search');
        const tagButton = document.getElementById('tag-search');
        const webButton = document.getElementById('web-search');
        
        preciseButton.addEventListener('click', () => {
            search_action = 1;
            preciseButton.classList.add('active');
            tagButton.classList.remove('active');
            webButton.classList.remove('active');
        });
        
        tagButton.addEventListener('click', () => {
            search_action = 2;
            tagButton.classList.add('active');
            preciseButton.classList.remove('active');
            webButton.classList.remove('active');
        });

        webButton.addEventListener('click', () => {
            search_action = 3;
            webButton.classList.add('active');
            preciseButton.classList.remove('active');
            tagButton.classList.remove('active');
        });
    }

    // 新增滑動條事件監聽
    const thresholdSlider = document.getElementById('search-threshold');
    const thresholdLabel = document.getElementById('search-threshold-label');
    thresholdSlider.addEventListener('input', function(e) {
        search_threshold = parseFloat(e.target.value);
        thresholdLabel.textContent = `当前阈值: ${search_threshold.toFixed(1)}`;
    });

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
            initializeSearchButtons();
    
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

    // 添加動畫控制函數
    function startLoadingAnimation() {
        if (loadingAnimationTimer) {
            clearInterval(loadingAnimationTimer);
        }
        
        currentLoadingText = '';
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loading-animation';
        loadingDiv.className = 'w-full flex mb-4 justify-start';
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'max-w-[100%] p-4 rounded-lg bg-gray-100 text-gray-800 mr-auto rounded-bl-none loading-message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'markdown-content prose';
        
        messageDiv.appendChild(contentDiv);
        loadingDiv.appendChild(messageDiv);
        
        const chatContainer = document.getElementById('preview');
        chatContainer.appendChild(loadingDiv);
        
        let charIndex = 0;
        loadingAnimationTimer = setInterval(() => {
            if (charIndex < loadingText.length) {
                currentLoadingText += loadingText[charIndex];
                charIndex++;
            } else {
                currentLoadingText = '';
                charIndex = 0;
            }
            contentDiv.textContent = currentLoadingText;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 200); // 每200毫秒更新一次
    }

    function stopLoadingAnimation() {
        if (loadingAnimationTimer) {
            clearInterval(loadingAnimationTimer);
            loadingAnimationTimer = null;
        }
        const loadingElement = document.getElementById('loading-animation');
        if (loadingElement) {
            loadingElement.remove();
        }
    }

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

        // 啟動載入動畫
        startLoadingAnimation();

        // 模擬向後端發送請求
        console.log("正在向後端發送請求...");
        fetch("/api/ai-chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ 
                message,
                search_action, // 添加搜尋模式參數
                search_threshold
            }),
        })
        .then((response) => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then((data) => {
            // 停止載入動畫
            stopLoadingAnimation();
            console.log("後端回應：", data);
            if (data && data.response) {
                // appendMessage("ai", `${JSON.stringify(data.response)}`);
                // console.log(data.response['primary_msg'])
                appendMessage("DQE-AI:", data.response['primary_msg'],'markdown')
                appendMessage("總共耗時:", data.response['totaltime'],'markdown')
                // appendMessage("similar questions:", data.response['googleserper'],'markdown')
                // appendMessage("similar questions:", data.response['googleserper'],'puretxt')
            } else {
                appendMessage("error", "後端未返回有效回應！");
            }
        })
        .catch((error) => {
            // 停止載入動畫
            stopLoadingAnimation();
            console.error("請求失敗詳情:", error);
            appendMessage("error", `連接失敗: ${error.message}`);
        });
    }


    marked.setOptions({
        gfm: true,
        breaks: true,
        tables: true,
        pedantic: false,
        sanitize: false,
        smartLists: true,
        smartypants: false,
        xhtml: false,
        headerIds: false,
        mangle: false
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
        // messageDiv.className = `message ${role}`;
        messageDiv.className = `max-w-[100%] p-4 rounded-lg ${
            role === "user" 
                ? "bg-blue-500 text-white ml-auto rounded-br-none" 
                : "bg-gray-100 text-gray-800 mr-auto rounded-bl-none"
        }`;
        
        const contentDiv = document.createElement("div");
        contentDiv.className = `markdown-content prose  $(role === "user" ? "prose-invert" : "")`;
        
        try {
                console.log("this text var is %s",text);
                if(text!="nodata"){
                    if(act === 'markdown') {
                        // 設置marked選項，確保表格正確渲染
                        marked.setOptions({
                            gfm: true,
                            breaks: true,
                            tables: true,
                            pedantic: false,
                            sanitize: false,
                            smartLists: true,
                            smartypants: false,
                            xhtml: false,
                            headerIds: false,
                            mangle: false
                        });
                        // 使用 marked 解析 Markdown
                        // const cleanContent = text.trim();
                        const cleanContent = text.trim()
                        .replace(/\n{3,}/g, '\n\n') // Normalize multiple newlines
                        .replace(/\|{2,}/g, '|');   // Fix multiple pipe characters
                        contentDiv.innerHTML = marked.parse(cleanContent);
                        // 為表格添加額外的樣式
                        const tables = contentDiv.querySelectorAll('table');
                        tables.forEach(table => {
                            table.className = 'min-w-full divide-y divide-gray-200 border';
                            const headers = table.querySelectorAll('th');
                            headers.forEach(header => {
                                header.className = 'px-3 py-2 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
                            });
                            const cells = table.querySelectorAll('td');
                            cells.forEach(cell => {
                                cell.className = 'px-3 py-2 whitespace-normal text-sm text-gray-500';
                            });
                        });
                    } else {
                        contentDiv.textContent = text;
                    }
                }else{
                    contentDiv.textContent = "未有相似的資料，請重新輸入查詢。"
                }
            
                messageDiv.appendChild(contentDiv);
                messageWrapper.appendChild(messageDiv);
                chatContainer.appendChild(messageWrapper);
                // 滾動到底部
                chatContainer.scrollTop = chatContainer.scrollHeight;

            // 重新應用程式碼高亮
            // document.querySelectorAll('pre code').forEach((block) => {
            //     hljs.highlightBlock(block);
            // });
        } catch (error) {
            console.error("渲染訊息時發生錯誤：", error);
            contentDiv.textContent = text;
        }
    }

})();
