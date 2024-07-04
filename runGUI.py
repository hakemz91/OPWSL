import logging
import utils
import torch
import time
import threading
import sys
import streamlit as st
from auto_gptq import AutoGPTQForCausalLM
from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from constants import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

# Suppress useless text
logging.getLogger("transformers").setLevel(logging.CRITICAL)

def beep():
    sys.stdout.write('\a')  # Writes the ASCII bell character

def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename, resume_download=False)
            max_ctx_size = 4096
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)

        else:
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif device_type.lower() == "cuda":
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    generation_config = GenerationConfig.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=4096,
        temperature=0.2,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm

def initialize_qa_system(device_type):
    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = db.as_retriever()
    
    template = """<s>[INST] You are a helpful assistant and you will answer all of user questions correctly 
    using the most related sentence in the context as long as you have knowledge about it. If you do not have knowledge about it, inform user that you do not know. 
    Double check for your mistake before answering. Provide detailed answer. Do not repeat what you have answered before.

    Context: {history} \n {context}
    User: {question} [/INST]"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    
    return qa

def main():
    st.set_page_config(page_title="QA System", layout="wide")
    st.title("QA System")

    # Sidebar for settings
    st.sidebar.title("Settings")
    device_type = st.sidebar.selectbox(
        "Device Type",
        ["cuda" if torch.cuda.is_available() else "cpu", "cpu", "cuda", "mps"],
        index=0
    )
    show_sources = st.sidebar.checkbox("Show Sources", value=False)
    save_qa = st.sidebar.checkbox("Save Q&A", value=False)

    # Initialize session state
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = initialize_qa_system(device_type)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_sources' not in st.session_state:
        st.session_state.current_sources = []

    # New Chat button
    if st.sidebar.button("New Chat"):
        st.session_state.chat_history = []
        st.session_state.current_sources = []
        st.experimental_rerun()

    # Main chat interface
    st.header("Chat")
    for q, a in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Assistant:** {a}")
        st.write("---")

    user_input = st.text_input("Enter your question:")

    if st.button("Ask"):
        if user_input:
            progress_bar = st.progress(0)
            start_time = time.time()
            
            res = st.session_state.qa_system(user_input)
            answer, docs = res["result"], res["source_documents"]
            
            time_taken = time.time() - start_time
            progress_bar.progress(100)

            st.session_state.chat_history.append((user_input, answer))
            st.session_state.current_sources = docs
            
            st.write("**Answer:**")
            st.write(answer)
            st.write(f"Time taken: {time_taken:.2f} seconds")

            if save_qa:
                utils.log_to_csv_and_txt(user_input, answer)

            # Trigger the beep in a separate thread
            threading.Thread(target=beep).start()

            st.experimental_rerun()

    # Display sources if the toggle is on
    if show_sources and st.session_state.current_sources:
        st.subheader("Source Documents")
        for i, doc in enumerate(st.session_state.current_sources):
            st.write(f"**Source {i+1}:** {doc.metadata['source']}")
            st.text(doc.page_content)
            st.write("---")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()