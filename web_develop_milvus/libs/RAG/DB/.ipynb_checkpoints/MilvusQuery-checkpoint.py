
import sys
import os
sys.path.append(os.path.abspath('../'))
from .DatabaseQuery import DatabaseQuery
from pymilvus import MilvusClient
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from typing import List, Any

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
    def __init__(self, uri="http://localhost:19530", token="root:Milvus", database:str=None, collection:str=None):
        self.client = MilvusClient(uri=uri, token=token)
        self.used_db = database
        self.used_collection = collection
        self.embedding_model="jina-embeddings-v2-base-zh"
        self.embedding_model_path="../../../Embedding_Models/jina-embeddings-v2-base-zh"
        self.ef=SentenceTransformerEmbeddingFunction(self.embedding_model_path, trust_remote_code=True)

    def create_collection(self, collection_name, dimension, schema=None):
        if self.client.has_collection(collection_name=collection_name):
            self.client.drop_collection(collection_name=collection_name)
        if schema:
            self.client.create_collection(collection_name=collection_name, schema=schema)
        else:
            self.client.create_collection(collection_name=collection_name, dimension=dimension)

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
        encoded_query = self.ef.encode_documents([query])
        query_vec = encoded_query[0]
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vec],
            output_fields= output_fields, #example:['description'],
            search_params= {"metric_type": "COSINE"},
            limit=limit
        )
        debugInfo(f"results:searched rows:{len(results[0])}\n{results}")
        return results
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

    def query(self, collection_name, filter=None, ids=None, output_fields=None, limit=None, partition_names=None):
        return self.client.query(
            collection_name=collection_name,
            filter=filter,
            ids=ids,
            output_fields=output_fields,
            limit=limit,
            partition_names=partition_names
        )

    def delete(self, collection_name, ids=None, filter=None):
        return self.client.delete(
            collection_name=collection_name,
            ids=ids,
            filter=filter
        )