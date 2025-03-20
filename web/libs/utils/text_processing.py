import pandas as pd


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



