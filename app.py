import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import pygame

# Initialize the pygame mixer
pygame.mixer.init()

def play_music(music_file):
    try:
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play(loops=-1)  # Play the music file in a loop
    except Exception as e:
        print(f"Failed to play music: {e}")

st.set_page_config(page_title="Chat with MPAC", page_icon="speak.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with MPAC :female-factory-worker:")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, how can I help you?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    play_music('./music/wait.wav')
    with st.spinner(text="We are gathering the necesssary information – Please hang on! This can take up to a minute."):
        # Call to play music
        reader = SimpleDirectoryReader(input_dir="./data/manuals/", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="U bent een expert op het gebied van aansluiting, levering en opwekking van energie voor een energiemaatschappij. Uw taak is om technische vragen te beantwoorden. Ga ervan uit dat alle vragen betrekking hebben op deze diensten. Houd uw antwoorden technisch en feitelijk; verzin geen kenmerken die er niet zijn."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        pygame.mixer.music.stop()  # Stop the music after loading is complete
        return index
    
index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Working on it ..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
