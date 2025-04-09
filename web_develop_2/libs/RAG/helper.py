from Retriever.CustomRetriever import CustomFAISSRetriever

def get_retriever(faiss_index_path, vector_db_path, k=2):
    return CustomFAISSRetriever(faiss_index_path, vector_db_path, k=k)