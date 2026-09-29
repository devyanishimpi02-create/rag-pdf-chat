# Chat with your PDF - simple RAG app
# Usage: python app.py yourfile.pdf
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

pdf_path = sys.argv[1]

# 1. Load the PDF and split it into chunks
docs = PyPDFLoader(pdf_path).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)

# 2. Create embeddings and store them in ChromaDB
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, emb, persist_directory="chroma_db")
retriever = db.as_retriever(search_kwargs={"k": 4})

# 3. LLM (needs GROQ_API_KEY set as an environment variable)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# 4. Ask questions: retrieve relevant chunks, then answer using only them
while True:
    q = input("\nAsk a question (or type exit): ")
    if q.strip().lower() == "exit":
        break
    hits = retriever.invoke(q)
    context = "\n\n".join(d.page_content for d in hits)
    prompt = (
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {q}"
    )
    print("\n" + llm.invoke(prompt).content)
    print("Sources: pages", sorted({d.metadata.get("page", 0) + 1 for d in hits}))
