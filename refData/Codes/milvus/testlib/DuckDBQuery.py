from .DatabaseQuery import DatabaseQuery

# class RelationalQuery(DatabaseQuery):
#     """關聯式資料庫查詢基底類別"""
#     pass

# DuckDB 查詢類別
import duckdb
import pandas as pd
import polars as pl

class DuckDBQuery:
    def __init__(self, db_path=":memory:"):
        self.conn = duckdb.connect(database=db_path)

    def query(self, sql, params=None, return_type='pandas'):
        """
        執行 SQL 查詢，回傳 pandas 或 polars DataFrame。
        """
        result = self.conn.execute(sql, params or ())
        if return_type == 'pandas':
            return result.df()
        elif return_type == 'polars':
            return result.pl()
        else:
            raise ValueError("return_type 僅支援 'pandas' 或 'polars'")

    def register_df(self, name, df):
        """
        註冊 pandas 或 polars DataFrame 為 DuckDB 查詢用表格。
        """
        if isinstance(df, pd.DataFrame):
            self.conn.register(name, df)
        elif isinstance(df, pl.DataFrame):
            # DuckDB 直接支援 polars DataFrame 註冊（需 pyarrow 支援）
            self.conn.register(name, df)
        else:
            raise TypeError("只支援 pandas.DataFrame 或 polars.DataFrame")

    def close(self):
        self.conn.close()

# 使用範例
# if __name__ == '__main__':
#     duckdbq = DuckDBQuery()

#     # pandas 範例
#     df_pandas = pd.DataFrame({'id': [1, 2, 3], 'value': ['a', 'b', 'c']})
#     duckdbq.register_df('pandas_table', df_pandas)
#     result_pd = duckdbq.query('SELECT * FROM pandas_table', return_type='pandas')
#     print('Pandas 查詢結果:')
#     print(result_pd)

#     # polars 範例
#     df_polars = pl.DataFrame({'id': [4, 5, 6], 'value': ['x', 'y', 'z']})
#     duckdbq.register_df('polars_table', df_polars)
#     result_pl = duckdbq.query('SELECT * FROM polars_table', return_type='polars')
#     print('Polars 查詢結果:')
#     print(result_pl)
