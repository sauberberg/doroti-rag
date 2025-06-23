from fastapi import FastAPI, Query
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

DB_DIR = './chroma_db'
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

app = FastAPI()

@app.get("/search")
def search(query: str = Query(..., min_length=3)):
    docs = vector_db.similarity_search(query, k=3)
    return {"results": [
        {"content": doc.page_content, "meta": doc.metadata} for doc in docs
    ]}
