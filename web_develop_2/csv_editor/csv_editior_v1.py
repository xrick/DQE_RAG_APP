import streamlit as st
import pandas as pd
import io
from pathlib import Path
from streamlit.components.v1 import html

# 設置頁面配置
st.set_page_config(layout="wide", page_title="CSV編輯器")

# 初始化 session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'show_message' not in st.session_state:
    st.session_state.show_message = {'type': None, 'message': None}

# 自定義CSS和JavaScript
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        margin-bottom: 10px;
    }
    .stDataFrame {
        height: calc(100vh - 200px) !important;
    }
    div[data-testid="column"] {
        height: 100%;
    }
    /* Alert styles */
    .alert {
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .alert-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def show_alert(message_type, message):
    """顯示提示訊息"""
    if message_type == 'success':
        st.success(message)
    elif message_type == 'error':
        st.error(message)

def load_csv(uploaded_file):
    """載入CSV文件"""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        if 'selected' not in df.columns:
            df['selected'] = False
        return df
    except Exception as e:
        show_alert('error', f"載入文件時發生錯誤: {str(e)}")
        return None

def save_dataframe(df, filepath):
    """保存數據框到CSV"""
    try:
        save_df = df.copy()
        if 'selected' in save_df.columns:
            save_df = save_df.drop(columns=['selected'])
        save_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        return True
    except Exception as e:
        show_alert('error', f"保存文件時發生錯誤: {str(e)}")
        return False

# 主要應用布局
st.title("CSV文件編輯器")

# 側邊欄 - 文件上傳
with st.sidebar:
    st.header("文件操作")
    uploaded_file = st.file_uploader("選擇CSV文件", type=['csv'])
    
    if uploaded_file is not None:
        if st.session_state.current_file != uploaded_file.name:
            st.session_state.df = load_csv(uploaded_file)
            st.session_state.current_file = uploaded_file.name
            show_alert('success', f'成功載入文件: {uploaded_file.name}')

    # 功能按鈕
    if st.session_state.df is not None:
        if st.button("儲存", key="save"):
            if uploaded_file is not None:
                # 創建臨時文件並保存
                temp_path = Path("temp.csv")
                if save_dataframe(st.session_state.df, temp_path):
                    with open(temp_path, 'rb') as f:
                        csv_data = f.read()
                    temp_path.unlink()  # 刪除臨時文件
                    
                    st.download_button(
                        label="下載已保存的文件",
                        data=csv_data,
                        file_name=uploaded_file.name,
                        mime='text/csv'
                    )
                    show_alert('success', '文件已成功保存！')

        if st.button("另存新檔", key="save_as"):
            if st.session_state.df is not None:
                save_df = st.session_state.df.copy()
                if 'selected' in save_df.columns:
                    save_df = save_df.drop(columns=['selected'])
                
                csv_data = save_df.to_csv(index=False, encoding='utf-8-sig')
                new_filename = f"new_{uploaded_file.name}" if uploaded_file else "new_file.csv"
                
                st.download_button(
                    label="下載另存新檔",
                    data=csv_data,
                    file_name=new_filename,
                    mime='text/csv'
                )

        if st.button("刪除選中行", key="delete_selected"):
            if st.session_state.df is not None:
                selected_rows = st.session_state.df['selected']
                if selected_rows.any():
                    st.session_state.df = st.session_state.df[~selected_rows].reset_index(drop=True)
                    show_alert('success', '已刪除選中的行')
                else:
                    show_alert('error', '請先選擇要刪除的行')

# 主要內容區域 - 數據編輯器
if st.session_state.df is not None:
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        height=600,
        column_config={
            "selected": st.column_config.CheckboxColumn(
                "選擇",
                help="選擇要刪除的行",
                default=False,
            )
        },
        key="data_editor"
    )
    
    # 更新session state中的數據框
    st.session_state.df = edited_df.copy()
else:
    st.info("請上傳CSV文件開始編輯")

# 顯示數據統計信息
if st.session_state.df is not None:
    with st.expander("數據統計信息"):
        st.write(f"總行數: {len(st.session_state.df)}")
        st.write(f"總列數: {len(st.session_state.df.columns)}")
        st.write("列名:", list(st.session_state.df.columns))
