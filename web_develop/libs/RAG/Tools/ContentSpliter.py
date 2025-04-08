from typing import List, Dict

class ContentProcessor:
    def __init__(self):
        # HTML 符號替換映射
        self.html_replacements = {
            '\n': '<br>',
            '-': '&ndash;',
            '"': '&quot;',
            "'": '&apos;',
            '(': '&#40;',
            ')': '&#41;',
        }
    
    def split_pages(self, content: str) -> List[Dict[str, str]]:
        """
        將內容按頁面分割並進行格式化
        
        Args:
            content: 原始內容字符串
        
        Returns:
            包含每個頁面信息的字典列表
        """
        try:
            # 使用 \n\n 分割頁面
            pages = content.split('\n\n')
            result = []
            
            for page in pages:
                if page.startswith('Page:'):
                    # 分離頁面標題和內容
                    parts = page.split('\nSummary: ', 1)
                    if len(parts) == 2:
                        title = parts[0].replace('Page: ', '')
                        content = parts[1]
                        
                        # 處理HTML符號
                        processed_content = self.replace_symbols(content)
                        
                        result.append({
                            'title': title,
                            'content': processed_content
                        })
            
            return result
        except Exception as e:
            print(f"分割頁面時發生錯誤: {str(e)}")
            return []
    
    def replace_symbols(self, text: str) -> str:
        """
        替換文本中的符號為HTML標籤
        
        Args:
            text: 需要處理的文本
        
        Returns:
            處理後的文本
        """
        try:
            # 使用預定義的替換規則
            for symbol, html_tag in self.html_replacements.items():
                text = text.replace(symbol, html_tag)
            
            # 處理特殊的Unicode字符
            text = re.sub(r'\u2060', '', text)  # 移除零寬不換行空格
            
            return text
        except Exception as e:
            print(f"替換符號時發生錯誤: {str(e)}")
            return text