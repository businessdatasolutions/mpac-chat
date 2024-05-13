import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

st.set_page_config(page_title="Chat with MPAC", page_icon="speak.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with MPAC :female-factory-worker:")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, how can I help you?"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="We are gathering the necesssary information – Please hang on! This can take up to a minute."):
        # Call to play music
        reader = SimpleDirectoryReader(input_dir="./data/manuals/", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0, system_prompt="You are an expert in the field of MPAC machines. Your task is to answer technical questions. Assume that all questions relate to the machines supplied by MPAC. Keep your answers technical and factual; do not invent features that do not exist."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index
    
index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0),
    context_prompt=(
        "You are an expert in the field of MPAC machines."
        "Your task is to answer technical questions. Assume that all questions relate to the machines supplied by MPAC."
        "Keep your answers technical and factual; do not invent features that do not exist."
        "Here are the relevant documents for the context:\n"
        "{context_str}"
        "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
    ),
    verbose=False,
)
        # st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

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
