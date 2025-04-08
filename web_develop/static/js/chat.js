// 聊天功能主類
class ChatInterface {
    constructor() {
      this.messageContainer = document.getElementById('chat-messages');
      this.messageInput = document.getElementById('message-input');
      this.sendButton = document.getElementById('send-button');
      this.isProcessing = false;
  
      this.init();
    }
  
    init() {
      // 綁定事件監聽器
      this.sendButton.addEventListener('click', () => this.sendMessage());
      this.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }
  
    // 發送訊息
    async sendMessage() {
      if (this.isProcessing || !this.messageInput.value.trim()) return;
  
      const userMessage = this.messageInput.value.trim();
      this.addMessage(userMessage, 'user');
      this.messageInput.value = '';
      this.isProcessing = true;
  
      // 顯示打字指示器
      this.showTypingIndicator();
  
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: userMessage }),
        });
  
        if (!response.ok) throw new Error('網路請求失敗');
  
        const data = await response.json();
        // 移除打字指示器
        this.removeTypingIndicator();
        // 添加 AI 回應
        this.addMessage(data.response, 'ai');
      } catch (error) {
        console.error('Error:', error);
        this.addMessage('抱歉，發生了一些錯誤。請稍後再試。', 'ai');
        this.removeTypingIndicator();
      }
  
      this.isProcessing = false;
    }
  
    // 添加訊息到聊天界面
    addMessage(message, type) {
      const messageDiv = document.createElement('div');
      messageDiv.className = `flex items-start space-x-3 ${type === 'user' ? 'justify-end' : ''}`;
  
      const bubbleClass = type === 'user' ? 'user-message' : 'ai-message';
      
      messageDiv.innerHTML = `
        ${type === 'ai' ? `
          <div class="flex-shrink-0">
            <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
              <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
            </div>
          </div>
        ` : ''}
        <div class="flex-1">
          <div class="message-bubble ${bubbleClass}">
            <p>${this.escapeHtml(message)}</p>
          </div>
        </div>
      `;
  
      this.messageContainer.appendChild(messageDiv);
      this.scrollToBottom();
    }
  
    // 顯示打字指示器
    showTypingIndicator() {
      const indicatorDiv = document.createElement('div');
      indicatorDiv.id = 'typing-indicator';
      indicatorDiv.className = 'flex items-start space-x-3';
      indicatorDiv.innerHTML = `
        <div class="flex-shrink-0">
          <div class="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
            </svg>
          </div>
        </div>
        <div class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      `;
      this.messageContainer.appendChild(indicatorDiv);
      this.scrollToBottom();
    }
  
    // 移除打字指示器
    removeTypingIndicator() {
      const indicator = document.getElementById('typing-indicator');
      if (indicator) {
        indicator.remove();
      }
    }
  
    // 滾動到底部
    scrollToBottom() {
      this.messageContainer.scrollTop = this.messageContainer.scrollHeight;
    }
  
    // HTML 轉義
    escapeHtml(unsafe) {
      return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }
  }
  
  // 當 DOM 載入完成後初始化聊天介面
  document.addEventListener('DOMContentLoaded', () => {
    window.chatInterface = new ChatInterface();
  });