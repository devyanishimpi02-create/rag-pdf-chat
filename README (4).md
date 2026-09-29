# Chat with PDF - RAG Question-Answering App

Ask questions about any PDF. Built with LangChain, ChromaDB, Sentence Transformers and Llama 3 (Groq API).

## How it works
1. Load the PDF and split it into overlapping chunks
2. Convert chunks to vector embeddings (all-MiniLM-L6-v2)
3. Store embeddings in ChromaDB
4. For each question, retrieve the top 4 similar chunks
5. Send the chunks + question to the LLM, which answers only from that context

## Run
    pip install -r requirements.txt
    set GROQ_API_KEY=your_key      (Windows)   |   export GROQ_API_KEY=your_key   (Mac/Linux)
    python app.py yourfile.pdf
