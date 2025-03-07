"""######################
1. 提取並向量化 qa.csv
首先，我們需要從 qa.csv 中提取問題和答案，然後使用FAISS將其向量化。
"""######################
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 讀取CSV文件
df = pd.read_csv('qa.csv', header=None, names=['question', 'answer'])

# 初始化句子嵌入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 向量化問題
question_embeddings = model.encode(df['question'].tolist(), convert_to_numpy=True)

# FAISS索引設置
embedding_dim = question_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)  # 使用L2距離度量
index.add(question_embeddings.astype(np.float32))  # 添加嵌入到索引中

# 保存FAISS索引
faiss.write_index(index, 'faiss_index.index')


"""######################
2. 構建多跳RAG應用
接下來，我們使用DSPy來構建多跳RAG應用，這將利用向量化的QA數據。
"""######################
import dspy
# 加載FAISS索引
index = faiss.read_index('faiss_index.index')

# 定義檢索器
class FaissRetriever(dspy.Retriever):
    def __init__(self, index):
        self.index = index

    def retrieve(self, query, k=3):
        query_embedding = model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k)
        return indices[0]  # 返回最接近的問題索引

# 定義回答生成器
def answer_generator(indices):
    answers = df['answer'].iloc[indices].tolist()
    return " ".join(answers)

# 整合檢索器和生成器
retriever = FaissRetriever(index)

def rag_qa_system(query):
    indices = retriever.retrieve(query)
    answer = answer_generator(indices)
    return answer

# 測試系統
query = "你的問題在這裡"
print(rag_qa_system(query))

"""######################
3. 完整實現的詳細說明
步驟1：我們首先從CSV文件中讀取問題和答案，然後使用句子嵌入模型將問題向量化。這些嵌入被添加到FAISS索引中，以便進行快速檢索。

步驟2：我們創建了一個自定義檢索器類 FaissRetriever，它使用FAISS索引來查找最接近的問題。檢索到的問題索引將用於生成最終答案。

步驟3：最終，我們整合了檢索器和回答生成器，並提供了一個簡單的接口 rag_qa_system，用於處理用戶查詢並返回相應的答案。

這樣，您就可以使用DSPy和FAISS構建一個有效的問答檢索增強生成應用，能夠高效地處理用戶查詢並返回相關答案。
"""######################