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
    // Modify loading animation functions if needed
    let loadingElement = null; // Keep track of the loading element

    function startLoadingAnimation(initialText = '搜尋中......') {
        stopLoadingAnimation(); // Ensure previous one is stopped

        const chatContainer = document.getElementById('preview');
        if (!chatContainer) return;

        loadingElement = document.createElement('div');
        loadingElement.id = 'loading-animation';
        // Simplified loading - just a text message
        loadingElement.className = 'w-full flex mb-4 justify-start';
        const messageDiv = document.createElement('div');
        messageDiv.className = 'max-w-[100%] p-4 rounded-lg bg-gray-100 text-gray-800 mr-auto rounded-bl-none loading-message';
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content markdown-content prose'; // Use same class
        contentDiv.textContent = initialText;
        messageDiv.appendChild(contentDiv);
        loadingElement.appendChild(messageDiv);
        chatContainer.appendChild(loadingElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function updateLoadingAnimation(newText) {
        if (loadingElement) {
            const contentDiv = loadingElement.querySelector('.message-content');
            if (contentDiv) {
                contentDiv.textContent = newText;
                const chatContainer = document.getElementById('preview');
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    }

    function stopLoadingAnimation() {
        if (loadingElement) {
            loadingElement.remove();
            loadingElement = null;
        }
    }

    // 添加動畫控制函數
    // function startLoadingAnimation() {
    //     if (loadingAnimationTimer) {
    //         clearInterval(loadingAnimationTimer);
    //     }
        
    //     currentLoadingText = '';
    //     const loadingDiv = document.createElement('div');
    //     loadingDiv.id = 'loading-animation';
    //     loadingDiv.className = 'w-full flex mb-4 justify-start';
        
    //     const messageDiv = document.createElement('div');
    //     messageDiv.className = 'max-w-[100%] p-4 rounded-lg bg-gray-100 text-gray-800 mr-auto rounded-bl-none loading-message';
        
    //     const contentDiv = document.createElement('div');
    //     contentDiv.className = 'markdown-content prose';
        
    //     messageDiv.appendChild(contentDiv);
    //     loadingDiv.appendChild(messageDiv);
        
    //     const chatContainer = document.getElementById('preview');
    //     chatContainer.appendChild(loadingDiv);
        
    //     let charIndex = 0;
    //     loadingAnimationTimer = setInterval(() => {
    //         if (charIndex < loadingText.length) {
    //             currentLoadingText += loadingText[charIndex];
    //             charIndex++;
    //         } else {
    //             currentLoadingText = '';
    //             charIndex = 0;
    //         }
    //         contentDiv.textContent = currentLoadingText;
    //         chatContainer.scrollTop = chatContainer.scrollHeight;
    //     }, 200); // 每200毫秒更新一次
    // }

    // function stopLoadingAnimation() {
    //     if (loadingAnimationTimer) {
    //         clearInterval(loadingAnimationTimer);
    //         loadingAnimationTimer = null;
    //     }
    //     const loadingElement = document.getElementById('loading-animation');
    //     if (loadingElement) {
    //         loadingElement.remove();
    //     }
    // }

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
        startLoadingAnimation("Initiating search...");

        // 模擬向後端發送請求
        console.log("正在向後端發送請求...");

        fetch("/api/ai-chat-stream", { // <--- CHANGE URL
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message,
                search_action,
                search_threshold
            }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            if (!response.body) {
                throw new Error('Response body is missing');
            }
            // Handle the streaming response
            return handleStream(response.body); // NEW function to process stream
        })
        .catch((error) => {
            stopLoadingAnimation(); // Stop animation on fetch error
            console.error("Fetch or stream setup failed:", error);
            appendMessage("error", `Connection or setup failed: ${error.message}`);
        });

        // fetch("/api/ai-chat", {
        //     method: "POST",
        //     headers: {
        //         "Content-Type": "application/json",
        //     },
        //     body: JSON.stringify({ 
        //         message,
        //         search_action, // 添加搜尋模式參數
        //         search_threshold
        //     }),
        // })
        // .then((response) => {
        //     if (!response.ok) {
        //         throw new Error(`HTTP error! status: ${response.status}`);
        //     }
        //     return response.json();
        // })
        // .then((data) => {
        //     // 停止載入動畫
        //     stopLoadingAnimation();
        //     console.log("後端回應：", data);
        //     if (data && data.response) {
        //         // appendMessage("ai", `${JSON.stringify(data.response)}`);
        //         // console.log(data.response['primary_msg'])
        //         appendMessage("DQE-AI:", data.response['primary_msg'],'markdown')
        //         appendMessage("總共耗時:", data.response['totaltime'],'markdown')
        //         // appendMessage("similar questions:", data.response['googleserper'],'markdown')
        //         // appendMessage("similar questions:", data.response['googleserper'],'puretxt')
        //     } else {
        //         appendMessage("error", "後端未返回有效回應！");
        //     }
        // })
        // .catch((error) => {
        //     // 停止載入動畫
        //     stopLoadingAnimation();
        //     console.error("請求失敗詳情:", error);
        //     appendMessage("error", `連接失敗: ${error.message}`);
        // });
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

    async function handleStream(stream) {
        const reader = stream.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let internalResultsDisplayed = false;
        let finalAnswerContent = ''; // Accumulate final answer chunks
    
        // Create placeholders in the UI if needed
        const internalResultsDivId = `internal-${Date.now()}`;
        const finalAnswerDivId = `final-${Date.now()}`;
    
        appendMessage("placeholder", "", 'html', internalResultsDivId); // Placeholder for internal results
        appendMessage("placeholder", "", 'html', finalAnswerDivId); // Placeholder for final answer
    
        // try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log("Stream finished.");
                    // Process any remaining buffer content
                    if (buffer.trim()) {
                         processChunkData(buffer.trim(), internalResultsDivId, finalAnswerDivId);
                    }
                    break; // Exit loop when stream is done
                }
    
                buffer += decoder.decode(value, { stream: true });
                console.log("Received raw chunk:", buffer); // Debugging
    
                // Process buffer line by line (assuming newline delimited JSON)
                let lines = buffer.split('\n');
                buffer = lines.pop(); // Keep the potentially incomplete last line in buffer
    
                for (const line of lines) {
                    if (line.trim() === '') continue;
                    try {
                        const jsonData = JSON.parse(line.trim());
                        console.log("Parsed JSON:", jsonData); // Debugging
                        // Process the structured data from the backend
                        const { type, content, message: statusMessage, computation_time } = jsonData;
    
                        if (type === "internal_results") {
                            updateMessageContent(internalResultsDivId, content || "No internal data found.", 'markdown');
                            internalResultsDisplayed = true;
                            // Maybe hide initial part of loading animation or change text
                            updateLoadingAnimation("Web search in progress...");
                        } else if (type === "status") {
                            updateLoadingAnimation(statusMessage || "Processing..."); // Update loading text
                        } else if (type === "final_answer_chunk") {
                            if (!internalResultsDisplayed) {
                                // Ensure internal results (even if 'nodata') are shown first
                                updateMessageContent(internalResultsDivId, "Processing internal data...", 'text'); // Placeholder
                            }
                            finalAnswerContent += content;
                            updateMessageContent(finalAnswerDivId, finalAnswerContent, 'markdown');
                            stopLoadingAnimation(); // Stop loading animation once final answer starts
                        } else if (type === "final_answer") { // If LLM doesn't stream chunks
                             if (!internalResultsDisplayed) {
                                updateMessageContent(internalResultsDivId, "Processing internal data...", 'text');
                            }
                            finalAnswerContent = content;
                            updateMessageContent(finalAnswerDivId, finalAnswerContent, 'markdown');
                            stopLoadingAnimation();
                        } else if (type === "error") {
                            updateMessageContent(finalAnswerDivId, `Error: ${statusMessage}`, 'text');
                            stopLoadingAnimation();
                        } else if (type === "done") {
                            stopLoadingAnimation();
                            appendMessage("info", `Total time: ${computation_time.toFixed(2)} seconds`, 'text');
                            // Final cleanup if needed
                            return; // End processing
                        }
    
                    } catch (e) {
                        console.error("Failed to parse JSON line:", line, e);
                        // Handle parse error - maybe display an error message or log it
                    }
                }
            }
            reader.releaseLock();
            stopLoadingAnimation(); // Ensure animation stops
        // } catch (error) {
        //     console.error("Error reading stream:", error);
        //     appendMessage("error", `Stream reading failed: ${error.message}`);
        //     stopLoadingAnimation();
        // } finally {
        //      reader.releaseLock();
        //      stopLoadingAnimation(); // Ensure animation stops
        // }
    }

    function appendMessage(role, text, format = 'text', elementId = null) {
        const chatContainer = document.getElementById("preview");
        if (!chatContainer) {
            console.error("找不到聊天容器！");
            return;
        }
    
        const messageWrapper = document.createElement("div");
        // Use provided ID or generate one
        messageWrapper.id = elementId || `${role}-${Date.now()}`;
        messageWrapper.className = "w-full flex mb-4 " +
            (role === "user" ? "justify-end" : "justify-start");
    
        // Don't display placeholder role
        if (role === "placeholder") {
             messageWrapper.style.display = 'none'; // Hide initially
        }
    
    
        const messageDiv = document.createElement("div");
        messageDiv.className = `max-w-[100%] p-4 rounded-lg ${
            role === "user"
            ? "bg-blue-500 text-white ml-auto rounded-br-none"
            : "bg-gray-100 text-gray-800 mr-auto rounded-bl-none"
            }`;
    
        const contentDiv = document.createElement("div");
        contentDiv.className = `message-content markdown-content prose ${role === "user" ? "prose-invert" : ""}`;
    
        if (text) { // Only set content if text is provided
           renderContent(contentDiv, text, format);
           if (role === "placeholder") messageWrapper.style.display = 'flex'; // Show if it has content
        }
    
    
        messageDiv.appendChild(contentDiv);
        messageWrapper.appendChild(messageDiv);
        chatContainer.appendChild(messageWrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    
        return messageWrapper.id; // Return the ID for potential updates
    }

    function updateMessageContent(elementId, text, format = 'text') {
        const messageWrapper = document.getElementById(elementId);
        if (messageWrapper) {
           const contentDiv = messageWrapper.querySelector('.message-content');
           if (contentDiv) {
               renderContent(contentDiv, text, format);
               messageWrapper.style.display = 'flex'; // Ensure it's visible
               // Adjust scroll if necessary
               const chatContainer = document.getElementById("preview");
               chatContainer.scrollTop = chatContainer.scrollHeight;
           }
        } else {
            console.warn(`Element with ID ${elementId} not found for updating.`);
            // Optionally append as a new message if update target is missing
            // appendMessage('ai', text, format);
        }
   }
   
   // Helper function to render content (extracted logic from appendMessage)
   function renderContent(contentDiv, text, format) {
        try {
            if (text === "nodata" && format !== 'text') { // Handle 'nodata' specifically unless it's meant as raw text
               contentDiv.textContent = "未有相似的资料，请重新输入查询。";
            } else if (format === 'markdown' || format === 'html') { // Treat html similar to markdown for now
               // Ensure marked options are set
               marked.setOptions({
                   gfm: true, breaks: true, tables: true, pedantic: false, sanitize: false, // Be careful with sanitize: false
                   smartLists: true, smartypants: false, xhtml: false, headerIds: false, mangle: false
               });
                const cleanContent = text.trim()
                                     .replace(/\n{3,}/g, '\n\n')
                                     .replace(/\|{2,}/g, '|');
               contentDiv.innerHTML = marked.parse(cleanContent);
               // Apply table styles (your existing logic)
               const tables = contentDiv.querySelectorAll('table');
               tables.forEach(table => { /* ... apply styles ... */ });
            } else { // Default to text
               contentDiv.textContent = text;
            }
        } catch (error) {
            console.error("渲染訊息時發生錯誤：", error);
            contentDiv.textContent = text; // Fallback to raw text on error
        }
   }

    // function appendMessage(role, text, act) {
    //     const chatContainer = document.getElementById("preview");
    //     if (!chatContainer) {
    //         console.error("找不到聊天容器！");
    //         return;
    //     }
    //     const messageWrapper = document.createElement("div");
        
    //     messageWrapper.className = "w-full flex mb-4 " + 
    //         (role === "user" ? "justify-end" : "justify-start");
    
    //     const messageDiv = document.createElement("div");
    //     // messageDiv.className = `message ${role}`;
    //     messageDiv.className = `max-w-[100%] p-4 rounded-lg ${
    //         role === "user" 
    //             ? "bg-blue-500 text-white ml-auto rounded-br-none" 
    //             : "bg-gray-100 text-gray-800 mr-auto rounded-bl-none"
    //     }`;
        
    //     const contentDiv = document.createElement("div");
    //     contentDiv.className = `markdown-content prose  $(role === "user" ? "prose-invert" : "")`;
        
    //     try {
    //             console.log("this text var is %s",text);
    //             if(text!="nodata"){
    //                 if(act === 'markdown') {
    //                     // 設置marked選項，確保表格正確渲染
    //                     marked.setOptions({
    //                         gfm: true,
    //                         breaks: true,
    //                         tables: true,
    //                         pedantic: false,
    //                         sanitize: false,
    //                         smartLists: true,
    //                         smartypants: false,
    //                         xhtml: false,
    //                         headerIds: false,
    //                         mangle: false
    //                     });
    //                     // 使用 marked 解析 Markdown
    //                     // const cleanContent = text.trim();
    //                     const cleanContent = text.trim()
    //                     .replace(/\n{3,}/g, '\n\n') // Normalize multiple newlines
    //                     .replace(/\|{2,}/g, '|');   // Fix multiple pipe characters
    //                     contentDiv.innerHTML = marked.parse(cleanContent);
    //                     // 為表格添加額外的樣式
    //                     const tables = contentDiv.querySelectorAll('table');
    //                     tables.forEach(table => {
    //                         table.className = 'min-w-full divide-y divide-gray-200 border';
    //                         const headers = table.querySelectorAll('th');
    //                         headers.forEach(header => {
    //                             header.className = 'px-3 py-2 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
    //                         });
    //                         const cells = table.querySelectorAll('td');
    //                         cells.forEach(cell => {
    //                             cell.className = 'px-3 py-2 whitespace-normal text-sm text-gray-500';
    //                         });
    //                     });
    //                 } else {
    //                     contentDiv.textContent = text;
    //                 }
    //             }else{
    //                 contentDiv.textContent = "未有相似的資料，請重新輸入查詢。"
    //             }
            
    //             messageDiv.appendChild(contentDiv);
    //             messageWrapper.appendChild(messageDiv);
    //             chatContainer.appendChild(messageWrapper);
    //             // 滾動到底部
    //             chatContainer.scrollTop = chatContainer.scrollHeight;

    //         // 重新應用程式碼高亮
    //         // document.querySelectorAll('pre code').forEach((block) => {
    //         //     hljs.highlightBlock(block);
    //         // });
    //     } catch (error) {
    //         console.error("渲染訊息時發生錯誤：", error);
    //         contentDiv.textContent = text;
    //     }
    // }

})();
