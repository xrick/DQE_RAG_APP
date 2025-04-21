
from abc import ABC, abstractmethod

class DatabaseQuery(ABC):
    """所有資料庫查詢操作的抽象基底類別"""
    
    @abstractmethod
    def query(self, *args, **kwargs):
        pass





# class NoSQLQuery(DatabaseQuery):
#     """NoSQL 資料庫查詢基底類別"""
#     pass

# class RedisQuery(NoSQLQuery):
#     """Redis 查詢實作（鍵值、集合等）"""
#     def query(self, key: str):
#         # 查詢 Redis 鍵
#         pass

# class VectorDBQuery(DatabaseQuery):
#     """向量資料庫查詢基底類別"""
#     pass

# class MilvusQuery(VectorDBQuery):
#     """Milvus 向量查詢實作"""
#     def query(self, vector, top_k: int):
#         # 執行向量相似度查詢
#         pass

# class RedisVectorQuery(VectorDBQuery):
#     """Redis 向量查詢實作（如 Redis-Search 向量模組）"""
#     def query(self, vector, top_k: int):
#         # 執行向量查詢
#         pass
