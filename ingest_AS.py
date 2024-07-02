import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

def file_log(logentry, filename):
    with open(filename, "a") as file:
        file.write(logentry + "\n")
    print(logentry)

def load_ingested_files():
    ingested_files = set()
    if os.path.exists("file_done_ingested.log"):
        with open("file_done_ingested.log", "r") as file:
            ingested_files = set(file.read().splitlines())
    return ingested_files

def load_single_document(file_path: str) -> Document:
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
            return loader.load()[0]
        else:
            file_log(f"{file_path} document type is undefined.", "file_done_ingested.log")
            return None
    except Exception as ex:
        file_log(f"{file_path} loading error: \n{ex}", "file_done_ingested.log")
        return None 

def load_documents(source_dir: str) -> list[Document]:
    ingested_files = load_ingested_files()
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                if source_file_path not in ingested_files:
                    paths.append(source_file_path)
                else:
                    file_log(f"{source_file_path} already ingested, skipping.", "duplicate_not_ingested.log")

    if not paths:
        logging.info("No new documents to ingest.")
        return []

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = max(round(len(paths) / n_workers), 1)  # Ensure chunksize is at least 1
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        for i in range(0, len(paths), chunksize):
            filepaths = paths[i : (i + chunksize)]
            future = executor.submit(load_document_batch, filepaths)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                contents = future.result()
                docs.extend([doc for doc in contents if doc is not None])
            except Exception as ex:
                file_log(f"Exception: {ex}", "file_done_ingested.log")
                
    # Log successfully loaded documents
    for doc in docs:
        file_log(doc.metadata["source"], "file_done_ingested.log")
                
    return docs

def load_document_batch(filepaths):
    logging.info("Loading document batch")
    with ThreadPoolExecutor(len(filepaths)) as exe:
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        return [future.result() for future in futures]

def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
           file_extension = os.path.splitext(doc.metadata["source"])[1]
           if file_extension == ".py":
               python_docs.append(doc)
           else:
               text_docs.append(doc)
    return text_docs, python_docs

def batch_insert_documents(texts, embeddings):
    BATCH_SIZE = 20000
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : (i + BATCH_SIZE)]
        db = Chroma.from_documents(
            batch_texts,
            embeddings,
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )

@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    if not documents:
        logging.info("No new documents to process.")
        return

    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
    )

    batch_insert_documents(texts, embeddings)
    
    # Need to create this .txt to use in another script
    with open("ingest_complete.txt", "w") as f:
        f.write("Ingest process completed")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()