import pandas as pd
import json
import ast


'''
Text Cleaning
'''

def sanitize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # 保留必要的符號
    special_chars = ['|', '-', ':', '<br>']
    for char in special_chars:
        text = text.replace(f' {char} ', char)
    # 移除多餘的空格
    text = ' '.join(text.split())
    return text


'''
df, dr to list
'''
def convert_dr_to_list(row, required_columns):
    # 獲取當前行的所有值
    row_data = []
    max_length = 1  # 預設長度為1
    # 步驟4：檢查是否有列表值並確定最大長度
    for col in required_columns:
        if isinstance(row[col], list):
            max_length = max(max_length, len(row[col]))
    # 步驟5：處理每一列的值
    for i in range(max_length):
        current_row = []
        for col in required_columns:
            value = row[col]
            if isinstance(value, list):
                # 如果是列表，取對應索引的值，如果索引超出範圍則使用空字符串
                current_row.append(value[i] if i < len(value) else '')
            else:
                # 如果不是列表，則重複使用該值
                current_row.append(value)
        row_data.append(current_row)

def convert_df_to_list(df, pos_list):
    """
    將DataFrame轉換為m*n列表
    Parameters:
    df (pandas.DataFrame): 輸入的DataFrame，包含指定的列
    Returns:
    list: 轉換後的二維列表
    """
    # 步驟1：定義所需的列順序
    required_columns = [
        '模块', '严重度', '问题现象描述', '原因分析', 
        '改善对策', '经验萃取', '评审后优化', '评分'
    ]
    # 步驟2：初始化結果列表
    result_list = []
    # 步驟3：遍歷DataFrame的每一行
    # for _, row in df.iterrows():
    for idx in pos_list:
        # 步驟6：將處理後的行數據添加到結果列表
        # result_list.extend(row_data)
        row = df.iloc[idx]
        result_list.extend(convert_dr_to_list(row, required_columns));
    return result_list


def format_googleserper_search_results(text_content):
    """
    用以解析google.serper.dev查詢回傳的文字內容
    使用此函式前，呼叫方需要先讀取回傳資料，程式如下：
    ```python
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
        "q": "SSD不读盘"
        })
        headers = {
        'X-API-KEY': '3026422e86bf19796a06b5f15433e9c0cd9bd88a',
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
    ```
    """
    try:
        # 使用ast.literal_eval安全地評估Python字面量
        # 這對於處理Python字典和列表字符串非常有用
        data = text_content.decode("utf-8")
        data = ast.literal_eval(text_content)
        return data
    except (SyntaxError, ValueError) as e:
        print(f"解析錯誤: {e}")
        
        # 如果ast.literal_eval失敗，嘗試json解析
        try:
            # 將單引號替換為雙引號
            json_str = text_content.replace("'", '"')
            # 處理可能的特殊字符
            json_str = json_str.replace('""delete', '"delete')
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print(f"JSON解析錯誤: {e}")
            return []
        

'''
轉換上面format_googleserper_search_results回傳的dict物件
為markdown格式
'''
def format_serper_results_to_markdown(serper_data):
    """
    將Google Serper API返回的搜索結果轉換為Markdown格式
    
    Args:
        serper_data (dict): Google Serper API返回的搜索結果字典
    
    Returns:
        str: 格式化後的Markdown字符串
    """
    # 檢查必要的鍵是否存在
    if not serper_data or 'organic' not in serper_data:
        return "# 搜索結果\n\n無法獲取相關搜索結果。"
    
    # 提取查詢關鍵詞
    query = serper_data.get('searchParameters', {}).get('q', '未知查詢')
    
    # 構建Markdown標題
    markdown = f"## {query} 搜索結果\n\n"
    
    # 添加摘要部分
    markdown += "### 搜索結果摘要\n\n"
    
    # 提取前三個結果的關鍵信息，用於生成摘要
    top_snippets = [item.get('snippet', '') for item in serper_data['organic'][:3] if 'snippet' in item]
    
    # 從摘要中提取可能的原因和解決方案
    causes = []
    solutions = []
    
    for snippet in top_snippets:
        if '原因' in snippet or '問題' in snippet or '故障' in snippet:
            causes.append(snippet)
        if '解決' in snippet or '修復' in snippet or '方法' in snippet:
            solutions.append(snippet)
    
    # 添加可能的原因
    if causes:
        markdown += "#### 可能的原因\n"
        for cause in causes[:3]:  # 限制顯示數量
            markdown += f"- **{cause[:50]}**{'...' if len(cause) > 50 else ''}\n"
        markdown += "\n"
    
    # 添加可能的解決方案
    if solutions:
        markdown += "#### 解決方案\n"
        for i, solution in enumerate(solutions[:5], 1):  # 限制顯示數量
            markdown += f"{i}. **{solution[:50]}**{'...' if len(solution) > 50 else ''}\n"
        markdown += "\n"
    
    # 添加詳細搜索結果表格
    markdown += "### 詳細搜索結果\n\n"
    markdown += "| 標題 | 摘要 |\n"
    markdown += "|------|------|\n"
    
    # 添加每個搜索結果
    for item in serper_data['organic'][:10]:  # 限制顯示前10個結果
        title = item.get('title', '無標題').replace('|', '\\|')  # 轉義表格分隔符
        snippet = item.get('snippet', '無摘要').replace('|', '\\|').replace('\n', ' ')
        markdown += f"| {title} | {snippet} |\n"
    
    # 添加相關搜索
    if 'relatedSearches' in serper_data and serper_data['relatedSearches']:
        markdown += "\n### 相關搜索\n\n"
        for item in serper_data['relatedSearches']:
            markdown += f"- {item.get('query', '')}\n"
    
    return markdown



