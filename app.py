import gradio as gr
import os
import pdfplumber
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

llm = ChatGroq(
    api_key=os.getenv("API_KEY"),
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

vector_store = None  

def process_pdf(pdf_file):
    pdf_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text()
    
    chunks = chunk_text(pdf_text)
    
    return FAISS.from_texts(chunks, embedding_model)

def process_pdf(pdf_file):
    pdf_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text()

    chunks = chunk_text(pdf_text)

    vector_store = FAISS.from_texts(chunks, embedding_model)

    retrieval_qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

    return retrieval_qa

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_response(query, retrieval_qa):
    response = retrieval_qa.run(query)
    return response

def chatbot_interface(pdf_file, user_query):
    retrieval_qa = process_pdf(pdf_file)
    return get_response(user_query, retrieval_qa)

with gr.Blocks() as app:
    gr.Markdown("# RAG Chatbot by Sakthimakesh")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF Document", type="filepath")
        query_input = gr.Textbox(label="Ask your question:", placeholder="Type your query here...")
        submit_button = gr.Button("Submit")

    output = gr.Textbox(label="Response")

    submit_button.click(fn=chatbot_interface, inputs=[pdf_input, query_input], outputs=output)

app.launch()
