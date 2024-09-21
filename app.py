from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import huggingface_hub

def main():
    load_dotenv()
    st.set_page_config(page_title="QNA Chatbot", page_icon=":robot:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write(css, unsafe_allow_html=True)
    st.header("QNA Chatbot")
    st.write("This is a chatbot that can answer questions about a given PDF file.")
    user_question = st.text_input("Ask a question about your pdf document:")

    if user_question:
        handle_user_question(user_question)

    with st.sidebar:
        st.subheader("Your document")
        pdfs = st.file_uploader("Upload your PDF files here and click on 'Upload'", accept_multiple_files=True)
        if st.button("Upload"):
            with st.spinner('Uploading...'):
                # get pdf text
                raw_data = get_pdf_data(pdfs)
                # create chunks
                chunks = get_chunks(raw_data)
                # create vector store - embedding
                vector_store = get_vector_store(chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)


def handle_user_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def get_pdf_data(pdfs):
    raw_data = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_data += page.extract_text()
    return raw_data

def get_chunks(raw_data):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_data)
    return chunks

def get_vector_store(chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    #llm = huggingface_hub.HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs= {'temperature':0.5, 'max_length':512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain

if __name__ == "__main__":
    main()