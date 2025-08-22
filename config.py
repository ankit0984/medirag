import os
from dotenv import load_dotenv

load_dotenv()

db_endpoint=os.getenv("ASTRADB_ENDPOINT")
db_token=os.getenv("ASTRA_TOKEN")
db_collection_name=os.getenv("DB_COLLECTION_NAME")
db_collection_names=os.getenv("DB_COLLECTION_NAMES")
Model_name = "sentence-transformers/all-MiniLM-L6-v2"
Chunk_Size = 1000
Chunk_Overlap = 100
gemini_key=os.getenv("GEMINI_KEY")
gemini_model=os.getenv("GEMINI_MODEL")
astra_namespace=os.getenv("ASTRA_NAMESPACE")
