# app.py
import streamlit as st
from utils import process_pdf, get_qa_chain
import tempfile
import os

st.title("📄 PDF Q&A with LangChain")

openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")

if uploaded_file and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    os.environ["OPENAI_API_KEY"] = openai_api_key

    st.success("✅ PDF uploaded and processed!")
    vectorstore = process_pdf(tmp_path)
    qa_chain = get_qa_chain(vectorstore)

    question = st.text_input("💬 Ask a question about the PDF")

    if question:
        response = qa_chain.run(question)
        st.write("🧠 Answer:", response)
