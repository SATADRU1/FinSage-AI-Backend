import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from models import Models

load_dotenv()

# Initialize the models
models = Models()
embeddings = models.embeddings_ollama
llm = models.model_ollama

# Define constants
data_folder = "data"
chunk_size = 1000
chunk_overlap = 50
check_interval = 10

# Chroma vector store
vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./db/chroma_langchain_db",  # Where to save data locally
)
# Ingest a file


def ingest_file(file_path: str):
    print(f"Ingesting file: {file_path}")
    # Skip non-PDF files
    if not file_path.lower().endswith('.pdf'):
        print(f"Skipping non-PDF file: {file_path}")
        return
    print(f"Starting to ingest file: {file_path}")
    # Editor wallah - next generain physicswallah
    loader = PyPDFLoader(file_path=file_path)
    loaded_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=[
            "\n", ","]
    )

    docs = text_splitter.split_documents(loaded_documents)
    print(f"Loaded {len(docs)} documents from {file_path}")
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)
    print(f"Added {len(docs)} documents to the vector store")
    print(f"Finished ingesting file: {file_path}")


def main():
    while True:
        files = os.listdir(data_folder)
        for file in files:
            if not file.startswith("ingested_"):
                file_path = os.path.join(data_folder, file)
                ingest_file(file_path)
                new_file = "ingested_" + file
                new_file_path = os.path.join(data_folder, new_file)
                os.rename(file_path, new_file_path)
        time.sleep(check_interval)


if __name__ == "__main__":
    main()
