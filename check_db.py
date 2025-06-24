from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

DB_DIR = './chroma_db'
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

docs = vector_db.similarity_search(" ", k=5)
for doc in docs:
    print(doc.page_content)
