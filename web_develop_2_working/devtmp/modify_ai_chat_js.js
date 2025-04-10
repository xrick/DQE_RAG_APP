

// Inside the sendMessage function:
// Replace the existing fetch().then().then().catch() block

// Start loading animation (maybe update its text later)
startLoadingAnimation("Initiating search..."); // Optional: pass initial message

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

    try {
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
    } catch (error) {
        console.error("Error reading stream:", error);
        appendMessage("error", `Stream reading failed: ${error.message}`);
        stopLoadingAnimation();
    } finally {
         reader.releaseLock();
         stopLoadingAnimation(); // Ensure animation stops
    }
}

// Modify appendMessage and add updateMessageContent
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

