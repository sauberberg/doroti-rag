from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

DB_DIR = "chroma_db"  # Путь к папке базы (если у тебя другая — подправь!)

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

# Попробуй найти 5 любых фрагментов (результаты могут быть случайные)
docs = vector_db.similarity_search(" ", k=5)
for doc in docs:
    print(doc.page_content)
