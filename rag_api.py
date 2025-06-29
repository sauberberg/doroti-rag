from fastapi import FastAPI, Query, Body, HTTPException
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# ─── Инициализация хранилища ────────────────────────────────────────────────────
DB_DIR = "./chroma_db"
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

app = FastAPI()


def _run_search(query: str, k: int = 3):
    """
    Общая логика поиска, чтобы не дублировать код
    """
    if not query or len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="`query` must be at least 3 characters long.")

    docs = vector_db.similarity_search(query, k=k)
    return {"results": [
        {"content": doc.page_content, "meta": doc.metadata} for doc in docs
    ]}


# ─── GET-эндпоинт (как было) ────────────────────────────────────────────────────
@app.get("/search")
async def search_get(query: str = Query(..., min_length=3)):
    """
    Старый способ: ?query=...
    """
    return _run_search(query)


# ─── Новый POST-эндпоинт ────────────────────────────────────────────────────────
@app.post("/search")
async def search_post(payload: dict = Body(...)):
    """
    Новый способ: тело запроса { "query": "..." }
    """
    query = payload.get("query")
    return _run_search(query)
