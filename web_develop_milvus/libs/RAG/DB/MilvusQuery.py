
import sys
import os
# sys.path.append(os.path.abspath('../'))
from .DatabaseQuery import DatabaseQuery
from pymilvus import MilvusClient
from pymilvus.client.types import LoadState
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from typing import List, Any, Dict
import polars as pl
import logging

# from pymilvus import (
#     MilvusClient,
#     connections,
#     utility,
#     FieldSchema, CollectionSchema, DataType,
#     Collection,
# )
# from pymilvus.client.types import LoadState
# from utils.logUtils import debugInfo

###setup debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Milvus 查詢類別
class MilvusQuery(DatabaseQuery):
    def __init__(self, uri="http://localhost:19530", token="root:Milvus", database:str=None, collection:str=None, embedding_model_path:str=None):
        self.client = MilvusClient(uri=uri, token=token)
        self.client.use_database(db_name=database)
        self.used_db = database
        self.used_collection = collection
        self.embedding_model="jina-embeddings-v2-base-zh"
        self.embedding_model_path=embedding_model_path#"../../../Embedding_Models/jina-embeddings-v2-base-zh"
        print(f"embedding path:{embedding_model_path} in MilvusQuery")
        self.ef=SentenceTransformerEmbeddingFunction(self.embedding_model_path, trust_remote_code=True)

    def create_collection(self, collection_name, dimension, schema=None):
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)
        if schema:
            self.client.create_collection(collection_name=collection_name, schema=schema)
        else:
            self.client.create_collection(collection_name=collection_name, dimension=dimension)

    def load_collection(self, collection_name:str=None):
        exists = self.client.has_collection(collection_name=collection_name)
        if exists:
            self.used_collection = collection_name
            self.client.load_collection(collection_name=collection_name)
        else:
            raise Exception("collection not found",)

    def insert(self, collection_name, data):
        return self.client.insert(collection_name=collection_name, data=data)

    def search(self, collection_name, query_vectors, limit=5, output_fields=None, search_params=None, metric_type:str="COSINE"):
        return self.client.search(
            collection_name=collection_name,
            data=query_vectors,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params or {"metric_type": metric_type}
        )

    def query(self, query:str=None, collection_name:str=None, output_fields:List=None, limit:int=None)->Any:
        _load_state = self.client.get_load_state(collection_name= collection_name)
        if _load_state['state'] != LoadState.Loaded:
            # debugInfo(f"the collection:{collection_name} is not loaded yet")    
            self.client.load_collection(collection_name = collection_name)
        encoded_query = self.ef.encode_documents([query])
        query_vec = encoded_query[0]
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vec],
            output_fields= output_fields, #example:['description'],
            search_params= {"metric_type": "COSINE"},
            limit=limit
        )
        # debugInfo(f"results:searched rows:{len(results[0])    }\n{results}")
        return results

    def delete(self, collection_name, ids=None, filter=None):
        return self.client.delete(
            collection_name=collection_name,
            ids=ids,
            filter=filter
        )
    
    def extract_entity_values_to_list(self, raw_entity_list:List[Dict[str, Any]]=None, keys:List[str]=None, threshold:float=None)-> List[List[Any]]:
        """
        Bug:
        根據 2024 年 5 月的 GitHub 回報，當你設定 metric_type 為 COSINE 時，Milvus 回傳的 distance 值與一般認知的 cosine distance 定義相反：

        完全相同的向量會得到 -1

        完全不相關得到 0

        完全相反得到 1
        也就是說，目前 Milvus 回傳的 distance 其實是「cosine similarity 的負值」或是直接回傳 similarity 而非 distance，導致排序與預期顛倒。

        這會造成你遇到的情況：「距離值較大」的結果反而更相關，因為在這個實作下，越小的 distance 其實代表越不相似，越大的 distance 反而是越相似。
        這與 cosine distance 應該「越小越相似」的定義是相反的。

        解決建議
        暫時解法：你可以在應用層將 distance 乘以 -1 或重新排序，讓結果符合「越小越相似」的直覺。

        關注官方修正：這個問題已被官方回報，建議持續關注 Milvus 的 issue 更新。

        檢查 embedding 模型：如果你用的 embedding 模型對非英文文本（如「印度」）支援不佳，也可能導致語意不準確，建議確認 embedding 輸出的品質。

        補充說明
        Milvus 官方文件也明確指出, cosine similarity 介於 [-1, 1]，而 cosine distance 應為 
        1 - cosine similarity, 但目前實作與文件描述不符，這正是你遇到問題的根本原因。
        """
        # 初始化結果列表
        result_list = []
        # estimeated_threshold = -1 * threshold 
        # 遍歷輸入列表
        for item_dict in raw_entity_list:
            # 步驟 5 & 7: 獲取 entity 和 distance，並進行健壯性檢查和條件判斷
            entity = item_dict.get('entity')
            distance = item_dict.get('distance')

            # 檢查 entity 是否是字典，distance 是否是數值，以及 distance 是否 <= threshold
            if isinstance(entity, dict) and isinstance(distance, (int, float)) and distance >= threshold:
                print(f"distance:{distance},  threshold:{threshold}")
                # 如果所有條件都滿足，則提取 entity 中的值
                # 使用 .get(key, None) 處理可能缺失的鍵
                inner_list = [entity.get(key, None) for key in keys]

                # 步驟 5: 將滿足條件的 inner_list 添加到結果列表
                result_list.append(inner_list)
            # else:
                # 不滿足條件的項目將被自動跳過，無需在此處添加 append
        # 返回過濾後的結果列表
        return result_list
    
    def get_entity_list(self, raw_result:List=None):
        return [item_dict['entity'] for item_dict in raw_result[0] if 'entity' in item_dict]


    def convert_entities_to_polars_df(self, entity_list: List[Dict[str, Any]]) -> pl.DataFrame:
        """
        將包含實體字典的列表轉換為 Polars DataFrame。

        這個函數接收一個列表，其中每個元素都是一個字典，
        代表一個實體的屬性。它使用 Polars 函式庫將此列表
        轉換為一個結構化的 DataFrame。

        Args:
            entity_list: 一個列表，其中每個元素都是一個字典。
                        預期每個字典包含以下鍵 (或其他鍵，Polars 會自動處理):
                        "problemtype", "severity", "module", "causeAnalysis",
                        "description", "improve", "experience"。

        Returns:
            一個 Polars DataFrame，其中:
            - 每行對應 entity_list 中的一個字典。
            - 列名對應字典的鍵。
            - Polars 會自動推斷數據類型。

        Raises:
            ImportError: 如果 polars 函式庫未安裝。
            TypeError: 如果輸入的 entity_list 不是列表或列表內元素不是字典
                    (雖然 Polars 可能會拋出自己的錯誤)。
        """
        # 步驟 4 (核心邏輯): 使用 pl.DataFrame() 建構函數直接轉換
        # Polars 的 DataFrame 建構函數可以直接處理字典列表
        try:
            df = pl.DataFrame(entity_list)
            return df
        except Exception as e:
            # 可以添加更具體的錯誤處理，但基本轉換失敗會在這裡捕獲
            print(f"轉換為 Polars DataFrame 時發生錯誤: {e}")
            # 根據需求決定是返回 None，空 DataFrame 還是重新拋出錯誤
            # 這裡選擇重新拋出，讓調用者知道發生了問題
            raise

    # 精度驗證
    # print(f"results:searched rows:{len(results[0])}\n{results}")
    # distances = [hit['distance'] for hit in results[0]]
    #     return self.client.query(
    #         collection_name=collection_name,
    #         # filter=filter,
    #         # ids=ids,
    #         output_fields=output_fields,
    #         limit=limit,
    #         partition_names=partition_names
    #     )

    # def query(self, collection_name, filter=None, ids=None, output_fields=None, limit=None, partition_names=None):
    #     return self.client.query(
    #         collection_name=collection_name,
    #         filter=filter,
    #         ids=ids,
    #         output_fields=output_fields,
    #         limit=limit,
    #         partition_names=partition_names
    #     )