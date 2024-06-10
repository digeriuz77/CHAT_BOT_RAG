# app.py
import os
import csv
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["streamlit"]["openai_api_key"]

# Custom prompt template for rephrasing follow-up questions
custom_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow-Up Input: {question}
Standalone question:
"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def get_pdf_text(docs):
    try:
        text = ""
        for pdf in docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
                text += "\n"
        return text
    except Exception as e:
        st.error(f"Failed to read PDF text: {e}")
        return ""

def get_chunks(raw_text):
    try:
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_text(raw_text)
        return chunks
    except Exception as e:
        st.error(f"Failed to split text into chunks: {e}")
        return []

def get_vectorstore(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
        vectorstore = faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )
    return conversation_chain

def save_dataset(question, answer):
    if "dataset" not in st.session_state:
        st.session_state.dataset = []
    st.session_state.dataset.append({"question": question, "answer": answer})
    with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(st.session_state.dataset)

def load_dataset():
    if "dataset" not in st.session_state:
        st.session_state.dataset = []
    if os.path.exists("dataset.csv"):
        with open("dataset.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            st.session_state.dataset = [row for row in reader]

def handle_question(question, openai_api_key):
    try:
        if st.session_state.conversation:
            response = st.session_state.conversation({'question': question})
            if response["answer"]:
                st.session_state.chat_history = response["chat_history"]
                for i, msg in enumerate(st.session_state.chat_history):
                    if i % 2 == 0:
                        st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
                save_dataset(question, response["answer"])
                return

        llm = ChatOpenAI(temperature=0.2, openai_api_key=openai_api_key)
        response = llm.predict(question)
        st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)
        save_dataset(question, response)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.set_page_config(page_title="QnA Bot", page_icon=":robot_face:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    load_dataset()

    st.markdown("<h1 style='text-align: center; color: #075E54;'>QnA Bot</h1>", unsafe_allow_html=True)
    question = st.text_input("Ask a question")

    if question:
        handle_question(question, openai_api_key)
    else:
        st.warning("Type a question to start the conversation.")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    with st.sidebar:
        st.subheader("Upload Documents")
        docs = st.file_uploader("Upload PDF documents", accept_multiple_files=True)

        if docs:
            st.subheader("Uploaded Documents")
            for doc in docs:
                st.write(f"- {doc.name}")

        if st.button("Process Documents"):
            with st.spinner("Processing"):
                if docs:
                    raw_text = get_pdf_text(docs)
                    text_chunks = get_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, openai_api_key)
                    st.success("Documents processed successfully!")
                else:
                    st.warning("No PDF files uploaded. Continuing conversation without searching from PDFs.")

    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

        def convert_to_csv(dataset):
            return pd.DataFrame(dataset).to_csv(index=False).encode('utf-8')

        csv = convert_to_csv(st.session_state.dataset)

        st.download_button(
            label="Download Chat History as CSV",
            data=csv,
            file_name='chat_history.csv',
            mime='text/csv',
        )

if __name__ == '__main__':
    main()
