import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List
import json
from datetime import datetime
from pathlib import Path

# 設置頁面配置
st.set_page_config(layout="wide", page_title="CSV編輯器")

# 自定義CSS樣式
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

@dataclass
class DataChange:
    """記錄數據變更的類"""
    timestamp: str
    row_index: int
    column_name: str
    old_value: any
    new_value: any

class DataManager:
    """數據管理器類"""
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._changes: List[DataChange] = []
        self._original_data: Optional[pd.DataFrame] = None
        self._file_name: Optional[str] = None

    def load_data(self, file) -> bool:
        """載入CSV文件"""
        try:
            self._data = pd.read_csv(file, encoding='utf-8-sig')
            self._original_data = self._data.copy()
            self._file_name = file.name
            if 'selected' not in self._data.columns:
                self._data['selected'] = False
            return True
        except Exception as e:
            st.error(f"載入文件時發生錯誤: {str(e)}")
            return False

    def update_data(self, new_df: pd.DataFrame):
        """更新整個數據框"""
        if self._data is not None:
            old_df = self._data.copy()
            self._data = new_df.copy()
            # 記錄變更
            for col in self._data.columns:
                if col != 'selected':  # 忽略選擇列的變更
                    changed_mask = old_df[col] != new_df[col]
                    for idx in new_df[changed_mask].index:
                        self._changes.append(DataChange(
                            timestamp=datetime.now().isoformat(),
                            row_index=idx,
                            column_name=col,
                            old_value=old_df.at[idx, col],
                            new_value=new_df.at[idx, col]
                        ))

    def get_data(self) -> Optional[pd.DataFrame]:
        """獲取當前數據"""
        return self._data.copy() if self._data is not None else None

    def save_data(self, path: str = None) -> tuple[bool, bytes]:
        """保存數據到CSV文件並返回數據內容"""
        try:
            if self._data is not None:
                save_df = self._data.copy()
                if 'selected' in save_df.columns:
                    save_df = save_df.drop(columns=['selected'])
                
                # 將數據轉換為CSV字節
                csv_buffer = save_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                return True, csv_buffer
            return False, b''
        except Exception as e:
            st.error(f"保存文件時發生錯誤: {str(e)}")
            return False, b''

    def delete_selected_rows(self) -> bool:
        """刪除選中的行"""
        try:
            if self._data is not None and 'selected' in self._data.columns:
                selected_rows = self._data['selected']
                if selected_rows.any():
                    self._data = self._data[~selected_rows].reset_index(drop=True)
                    return True
            return False
        except Exception as e:
            st.error(f"刪除行時發生錯誤: {str(e)}")
            return False

    def has_changes(self) -> bool:
        """檢查是否有未保存的變更"""
        return len(self._changes) > 0

# 初始化 session state
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

def main():
    st.title("CSV文件編輯器")

    # 側邊欄 - 文件上傳
    with st.sidebar:
        st.header("文件操作")
        uploaded_file = st.file_uploader("選擇CSV文件", type=['csv'])
        
        if uploaded_file is not None:
            if (st.session_state.data_manager._file_name != uploaded_file.name or 
                st.session_state.data_manager._data is None):
                if st.session_state.data_manager.load_data(uploaded_file):
                    st.success(f'成功載入文件: {uploaded_file.name}')

        # 功能按鈕
        if st.session_state.data_manager.get_data() is not None:
            if st.button("儲存", key="save"):
                success, csv_data = st.session_state.data_manager.save_data()
                if success:
                    st.download_button(
                        label="下載已保存的文件",
                        data=csv_data,
                        file_name=uploaded_file.name if uploaded_file else "data.csv",
                        mime='text/csv'
                    )
                    st.success('文件已成功保存！')

            if st.button("另存新檔", key="save_as"):
                success, csv_data = st.session_state.data_manager.save_data()
                if success:
                    new_filename = f"new_{uploaded_file.name}" if uploaded_file else "new_file.csv"
                    st.download_button(
                        label="下載另存新檔",
                        data=csv_data,
                        file_name=new_filename,
                        mime='text/csv'
                    )

            if st.button("刪除選中行", key="delete_selected"):
                if st.session_state.data_manager.delete_selected_rows():
                    st.success('已刪除選中的行')
                else:
                    st.error('請先選擇要刪除的行')

    # 主要內容區域 - 數據編輯器
    if st.session_state.data_manager.get_data() is not None:
        edited_df = st.data_editor(
            st.session_state.data_manager.get_data(),
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
        
        # 更新數據管理器中的數據
        if edited_df is not None:
            st.session_state.data_manager.update_data(edited_df)

        # 顯示數據統計信息
        with st.expander("數據統計信息"):
            current_data = st.session_state.data_manager.get_data()
            st.write(f"總行數: {len(current_data)}")
            st.write(f"總列數: {len(current_data.columns)}")
            st.write("列名:", list(current_data.columns))
            if st.session_state.data_manager.has_changes():
                st.write("存在未保存的更改")
    else:
        st.info("請上傳CSV文件開始編輯")

if __name__ == "__main__":
    main()
