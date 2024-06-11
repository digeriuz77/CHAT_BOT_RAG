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
from transcriber import Transcription
import whisper

# Load environment variables from .env file
load_dotenv()

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

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

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write("Please upload PDFs and click process")

def upload_audio_file():
    uploaded_file = st.file_uploader("Upload an audio file (.mp3 or .mp4)", type=["mp3", "mp4"])
    if uploaded_file is not None:
        return uploaded_file
    return None

def transcribe_audio(files, model_name, translation):
    transcription = Transcription(files)
    transcription.transcribe(model_name, translation)
    return transcription.output

# Streamlit app layout
def main():
    st.title("PDF and Audio Conversational Assistant")
    st.write("Upload PDF files or audio files to interact with the content using conversational AI.")

    # Load dataset if it exists
    load_dataset()

    # Process uploaded PDFs
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True)
    if st.button("Process PDFs") and uploaded_files:
        pdf_text = get_pdf_text(uploaded_files)
        chunks = get_chunks(pdf_text)
        vectorstore = get_vectorstore(chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)

    # Handle user input
    user_question = st.text_input("Ask your question here:")
    if st.button("Submit"):
        handle_userinput(user_question)

    # Transcription functionality
    uploaded_audio = upload_audio_file()
    if uploaded_audio is not None:
        model_name = st.selectbox("Select Whisper Model", ["tiny", "base", "small", "medium", "large"], index=1)
        translation = st.checkbox("Translate to English", value=False)
        
        if st.button("Transcribe Audio"):
            transcriptions = transcribe_audio([uploaded_audio], model_name, translation)
            for transcription in transcriptions:
                st.write(f"**Transcription of {transcription['name']}:** {transcription['text']}")
                if translation:
                    st.write(f"**Translation:** {transcription['translation']}")
                
                if st.button("Use Transcription as Input"):
                    handle_userinput(transcription['text'])

if __name__ == '__main__':
    main()
