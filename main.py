import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from hello import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from gtts import gTTS
import speech_recognition as sr
import pygame
# to retrive info from pdf 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
# speech to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone(device_index=2) as source:
        try:
            st.write("Say something...")
            audio = recognizer.listen(source)
            user_question = recognizer.recognize_google(audio)
            return user_question
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand what you said.")
            return None
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
            return None
        except Exception as e:
            st.write(f"An error occurred: {e}")
            return None
#to convert the retrived info to chunks 
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
#to convert the chunck to vector
def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
#chain convo
def get_conversation_chain(vectorstore):
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    llm=ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
#text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
#user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            bot_response = bot_template.replace("{{MSG}}", message.content)
            st.write(bot_response, unsafe_allow_html=True)
            text_to_speech(message.content)  # Convert bot response to speech
            pygame.mixer.init()
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(5)
            pygame.mixer.music.unload()
#main func
def main():
    load_dotenv()
    st.set_page_config(page_title="ASKMEANYTHING", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ASKMEANYTHING:books:")
    user_question = st.text_input("Ask your queries:")

    if user_question:
        handle_userinput(user_question)

    if st.button("Use Voice Input"):
        new_question = speech_to_text()
        if new_question:
            handle_userinput(new_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)
#run if the file is present , not imported 
if __name__ == '__main__':
    main()
