from pymilvus import model,MilvusClient, DataType, FieldSchema, CollectionSchema
import pandas as pd

dbName="./dqe_milvus_data.db"
collectionName="quality_issues"
collectionDesc="quailty_issues_collection"
csv_src = "deq_learn_refine2_correct.csv"

ef = model.DefaultEmbeddingFunction()  # 确保已安装 pymilvus[model]
client = MilvusClient(db_name=dbName)  # 必须包含.db后缀

# 定义字段结构（关键修正点）
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="module", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="severity", dtype=DataType.VARCHAR, max_length=1),
    FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="causeAnalysis", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="improve", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="experience", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="judge", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="score", dtype=DataType.INT16),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=ef.dim)
]

# 创建集合（新API规范）
# collection_name = "quality_issues"
if client.has_collection(collectionName):
    client.drop_collection(collectionName)

client.create_collection(
    collection_name=collectionName,
    schema=CollectionSchema(fields, description=collectionDesc),  # 单一路径传递schema
    # 不再需要单独传递fields参数
)

df = pd.read_csv(csv_src)
data = [{
    "module": row["模块"],
    "severity": str(row["严重度"]),
    "description": row["问题现象描述"],
    "description": row["原因分析"],
    "description": row["改善对策"],
    "description": row["经验萃取"],
    "description": row["评审后优化"],
    "description": row["评分"],
    "vector": ef.encode_documents([row["问题现象描述"]])[0]
} for _, row in df.iterrows()]

client.insert(collectionName, data)

print(f"成功插入 {len(df)} 条数据，向量维度={ef.dim}")