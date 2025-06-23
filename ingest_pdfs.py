from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import glob

PDF_DIR = './pdf_lessons/*.pdf'
DB_DIR = './chroma_db'

# Загружаем все PDF-файлы
all_docs = []
for path in glob.glob(PDF_DIR):
    loader = PyPDFLoader(path)
    docs = loader.load()
    all_docs.extend(docs)

# Делим на куски для поиска
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
docs_chunks = splitter.split_documents(all_docs)

# Делаем embedding и сохраняем в chroma_db
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(
    docs_chunks,
    embedding,
    persist_directory=DB_DIR
)

print(f"Индекс построен! Кусков текста: {len(docs_chunks)}")
