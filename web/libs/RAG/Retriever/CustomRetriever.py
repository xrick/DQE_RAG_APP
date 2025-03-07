import dspy
import faiss
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS

class CustomFAISSRetriever(dspy.Retrieve):
    def load_index(self, idx_path=None):
        try:
            index = faiss.read_index(idx_path)
            print(f"成功載入FAISS索引，包含 {index.ntotal} 個向量")
            return index
        except Exception as e:
            print(f"索引載入失敗: {str(e)}")
            return None
        
    def load_local_db(self, local_db_path=None, embeddings=None):
        try:
            db = LangchainFAISS.load_local(
                folder_path=local_db_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"載入成功，共 {db.index.ntotal} 筆技術問答")
            return db
        except Exception as e:
            print(f"向量庫載入異常: {str(e)}")
            return None
            
    def __init__(self, faiss_index_path, vector_db_path, k=2):
        super().__init__()
        self.k = k
        # 使用同一個模型名稱
        self.model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        
        # 初始化 embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name
        )
        
        # 載入 FAISS 索引
        self.index = self.load_index(faiss_index_path)
        
        # 載入向量庫
        # self.vector_db = self.load_local_db(vector_db_path, self.embeddings)
        
        # 使用相同的模型進行查詢編碼
        self.model = SentenceTransformer(self.model_name)
    
    def __call__(self, query):
        # 編碼查詢
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=False,
            show_progress_bar=False  # 對單一查詢關閉進度條
        )
        query_embedding = query_embedding.reshape(-1,1).T
        # query_embedding = query_embedding.cpu().numpy()
        query_embedding = query_embedding.astype(np.float32)
        
        # 搜索向量庫
        # docs = self.vector_db.similarity_search_with_score(query_embedding, k=self.k)
        distance,pos = self.index.search(query_embedding, k=self.k)
        print(distance)
        print(pos)
        # return the pos for retrieving data from answers
        return pos;