// ai-chat.js
(function () {
    console.log("ai-chat.js 加載完成！");
    console.log("it has invoke！");

    // 搜尋模式控制變數
    let search_action = 1; // 預設為精準搜尋

    // 欄位校驗函數 (新增)
    function validateTableColumns(data) {
        const requiredColumns = [
            '模块', '严重度(A/B/C)', '问题现象描述', 
            '原因分析', '改善对策', '经验萃取',
            '评审后优化', '评分'
        ];
        return data.every(row => 
            requiredColumns.every(col => col in row)
        );
    }

    // 改進版表格渲染邏輯 (修改)
    function renderMarkdownTable(data) {
        if (!validateTableColumns(data)) {
            console.error('欄位結構異常，缺少必要字段');
            return '```markdown\n[錯誤] 數據格式不符合規範\n```';
        }

        const processedData = data.map(row => ({
            模块: row.模块,
            严重度: row['严重度(A/B/C)'],
            现象描述: row.问题现象描述.replace(/\n/g, '<br>'),
            原因分析: row.原因分析,
            对策: row.改善对策.replace(/\n/g, '<br>'),
            经验: row.经验萃取,
            优化: row.评审后优化 || '无',
            评分: isNaN(row.评分) ? 'No Data' : row.评分
        }));

        return `
| 模块 | 严重度(A/B/C) | 问题现象描述 | 原因分析 | 改善对策 | 经验萃取 | 评审后优化 | 评分 |
|------|:------------:|--------------|---------|---------|---------|-----------|:----:|
${processedData.map(row => 
    `| ${row.模块} ` +
    `| ${row.严重度} ` +
    `| ${row.现象描述} ` +
    `| ${row.原因分析} ` +
    `| ${row.对策} ` +
    `| ${row.经验} ` +
    `| ${row.优化} ` +
    `| ${row.评分} |`
).join('\n')}
        `;
    }

    // 初始化搜尋按鈕 (保留原始邏輯)
    function initializeSearchButtons() {
        const preciseButton = document.getElementById('precise-search');
        const tagButton = document.getElementById('tag-search');
        
        preciseButton.addEventListener('click', () => {
            search_action = 1;
            preciseButton.classList.add('active');
            tagButton.classList.remove('active');
        });
        
        tagButton.addEventListener('click', () => {
            search_action = 2;
            tagButton.classList.add('active');
            preciseButton.classList.remove('active');
        });
    }

    // 初始化聊天介面 (整合校驗邏輯)
    function initializeChatInterface() {
        const maxRetries = 10;
        let retryCount = 0;
    
        function tryInitialize() {
            console.log("嘗試初始化聊天介面...");
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
    
            // 事件綁定 (保留原始邏輯)
            sendButton.addEventListener("click", function() {
                const message = messageInput.value.trim();
                if (message) {
                    sendMessage(message);
                }
            });
    
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

    // 發送消息函數 (整合數據處理)
    function sendMessage(message) {
        const chatContainer = document.getElementById("preview");
        const messageInput = document.getElementById("editor");

        if (!chatContainer || !messageInput) {
            console.error("聊天框或輸入框未找到！");
            return;
        }

        appendMessage("user", message);

        messageInput.value = "";

        fetch("/api/ai-chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ 
                message,
                search_action
            }),
        })
        .then((response) => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then((data) => {
            if (data?.response) {
                // 新增表格渲染調用
                const tableMarkdown = renderMarkdownTable(data.response.data || []);
                appendMessage("ai", tableMarkdown, 'markdown');
            } else {
                appendMessage("error", "後端未返回有效回應！");
            }
        })
        .catch((error) => {
            console.error("請求失敗詳情:", error);
            appendMessage("error", `連接失敗: ${error.message}`);
        });
    }

    // Markdown 解析配置 (保留原始設定)
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

    // 消息渲染函數 (優化錯誤處理)
    function appendMessage(role, text, act) {
        const chatContainer = document.getElementById("preview");
        if (!chatContainer) {
            console.error("找不到聊天容器！");
            return;
        }

        const messageWrapper = document.createElement("div");
        messageWrapper.className = `w-full flex mb-4 ${role === "user" ? "justify-end" : "justify-start"}`;
    
        const messageDiv = document.createElement("div");
        messageDiv.className = `max-w-[100%] p-4 rounded-lg ${
            role === "user" 
                ? "bg-blue-500 text-white ml-auto rounded-br-none" 
                : "bg-gray-100 text-gray-800 mr-auto rounded-bl-none"
        }`;
        
        const contentDiv = document.createElement("div");
        contentDiv.className = `markdown-content prose ${role === "user" ? "prose-invert" : ""}`;

        try {
            if(act === 'markdown') {
                contentDiv.innerHTML = marked.parse(text.trim());
            } else {
                contentDiv.textContent = text;
            }
            messageDiv.appendChild(contentDiv);
            messageWrapper.appendChild(messageDiv);
            chatContainer.appendChild(messageWrapper);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            // 代碼高亮
            document.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightBlock(block);
            });
        } catch (error) {
            console.error("渲染訊息時發生錯誤：", error);
            contentDiv.textContent = text;
        }
    }

    // 全局暴露初始化函數
    window.initializeChatInterface = initializeChatInterface;

    // 延遲初始化
    setTimeout(() => {
        initializeChatInterface();
        initializeSearchButtons();
    }, 500);
})();
