
import sys
import os
sys.path.append(os.path.abspath('../'))
from .DatabaseQuery import DatabaseQuery
from pymilvus import MilvusClient
from pymilvus.client.types import LoadState
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from typing import List, Any, Dict
import polars as pl

# from pymilvus import (
#     MilvusClient,
#     connections,
#     utility,
#     FieldSchema, CollectionSchema, DataType,
#     Collection,
# )
# from pymilvus.client.types import LoadState
from utils.logUtils import debugInfo


# Milvus 查詢類別
class MilvusQuery(DatabaseQuery):
    def __init__(self, uri="http://localhost:19530", token="root:Milvus", database:str=None, collection:str=None, embedding_model_path:str=None):
        self.client = MilvusClient(uri=uri, token=token)
        self.client.use_database(db_name=database)
        self.used_db = database
        self.used_collection = collection
        self.embedding_model="jina-embeddings-v2-base-zh"
        self.embedding_model_path=embedding_model_path#"../../../Embedding_Models/jina-embeddings-v2-base-zh"
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
            debugInfo(f"the collection:{collection_name} is not loaded yet")    
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
    
    def extract_entity_values_to_list(raw_entity_list: List[Dict[str, Any]], keys_order:List[str],threshold:float)-> List[List[Any]]:
        """
        將包含 'entity' 字典的原始列表轉換為值的列表的列表，
        並根據 'distance' 值進行過濾。

        遍歷輸入列表中的每個字典，檢查其 'distance' 值。如果 'distance'
        小於或等於給定的 threshold，則提取其 'entity' 鍵對應的嵌套字典中的值，
        並按照預定義的順序將這些值組成一個內部列表。最終返回包含所有
        滿足條件的內部列表的外部列表。
        Args:
            raw_entity_list: 一個列表，其中每個元素都是一個字典。
                            預期每個字典包含 'distance' (數值類型) 和 'entity' 鍵。
                            'entity' 的值應為另一個包含以下鍵的字典：
                            "problemtype", "severity", "module", "causeAnalysis",
                            "description", "improve", "experience"。
            threshold:       一個浮點數。只有當 item_dict['distance'] <= threshold 時，
                            對應的 'entity' 數據才會被處理和包含在結果中。
        Returns:
            一個列表的列表。每個內部列表包含按以下順序排列的值：
            ["problemtype", "severity", "module", "causeAnalysis",
            "description", "improve", "experience"]。
            僅包含那些原始 'distance' 值小於或等於 threshold 的項目。
            如果 'entity' 字典中缺少某個鍵，則該位置的值為 None。
            如果輸入列表的某個元素缺少 'entity' 或 'distance' 鍵，
            或者它們的類型不正確，則該元素將被跳過。

        Raises:
            TypeError: 如果輸入的 raw_entity_list 不是列表。
                    (由 Python 迭代機制隱式處理)
        """
        # 初始化結果列表
        result_list = []

        # 遍歷輸入列表
        for item_dict in raw_entity_list:
            # 步驟 5 & 7: 獲取 entity 和 distance，並進行健壯性檢查和條件判斷
            entity = item_dict.get('entity')
            distance = item_dict.get('distance')

            # 檢查 entity 是否是字典，distance 是否是數值，以及 distance 是否 <= threshold
            if isinstance(entity, dict) and isinstance(distance, (int, float)) and distance <= threshold:
                print(f"distance:{distance},  threshold:{threshold}")
                # 如果所有條件都滿足，則提取 entity 中的值
                # 使用 .get(key, None) 處理可能缺失的鍵
                inner_list = [entity.get(key, None) for key in keys_order]

                # 步驟 5: 將滿足條件的 inner_list 添加到結果列表
                result_list.append(inner_list)
            # else:
                # 不滿足條件的項目將被自動跳過，無需在此處添加 append

        # 返回過濾後的結果列表
        return result_list
    
    def get_entity_list(raw_result:List=None):
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