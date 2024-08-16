import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Dark-themed, stylish interface ke liye custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #1e1e1e; /* Background dark color */
        color: #ffffff; /* Text white color */
    }
    .stButton>button {
        background-color: #444444; /* Button ka background color */
        color: #ffffff; /* Button ka text color */
        border-radius: 8px; /* Button ke corners rounded */
        font-weight: bold; /* Button ka text bold */
    }
    .stTextInput>div>div>input {
        background-color: #2b2b2b; /* Text input ka background color */
        color: #ffffff; /* Text input ka text color */
        border-radius: 8px; /* Text input ke corners rounded */
    }
    .stSidebar {
        background-color: #2b2b2b; /* Sidebar ka background color */
        border-radius: 8px; /* Sidebar ke corners rounded */
    }
    .stMarkdown {
        color: #ffffff; /* Markdown text ka color */
    }
    .st-chat-message {
        background-color: #333333; /* Chat messages ka background color */
        border-radius: 12px; /* Chat messages ke corners rounded */
        padding: 10px; /* Chat messages ka padding */
    }
    .st-chat-message-user {
        background-color: #4d4d4d; /* User messages ka background color grey kiya */
        color: #ffffff; /* User messages ka text color white */
        border-radius: 12px; /* User messages ke corners rounded */
        padding: 10px; /* User messages ka padding */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Arxiv aur Wikipedia Tools ko initialize karna
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200) # Arxiv ka wrapper banaya
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper) # Arxiv tool ko initialize kiya

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200) # Wikipedia wrapper banaya
wiki = WikipediaQueryRun(api_wrapper=api_wrapper) # Wikipedia tool ko initialize kiya

search = DuckDuckGoSearchRun(name="Search") # DuckDuckGo search tool ko initialize kiya

st.title("Chat-Mate Search Engine Version") # App ka title diya

# Sidebar ke liye title aur API key input
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password") # API key input box

# Chat history ke liye session state ka use
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm Chat-Mate. How can I help you?"} # Assistant ka pehla message
    ]

# Har message ko display karna
for msg in st.session_state.messages:
    role_class = "st-chat-message-user" if msg["role"] == "user" else "st-chat-message" # Role ke hisaab se class set kiya
    st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True) # Message display kiya

# User input handle karna
if prompt := st.chat_input(placeholder="Search Anything..."): # Chat input box se prompt lena
    st.session_state.messages.append({"role": "user", "content": prompt}) # User ke message ko session state me save kiya
    st.markdown(f"<div class='st-chat-message-user'>{prompt}</div>", unsafe_allow_html=True) # User message ko display kiya

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True) # LLM ko initialize kiya
    tools = [search, arxiv, wiki] # Tools ka list banaya

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True) # Agent ko initialize kiya

    with st.container(): # Response ke liye container banaya
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False) # Callback handler set kiya
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb]) # Agent ko run kiya aur response liya
        st.session_state.messages.append({'role': 'assistant', "content": response}) # Assistant ka response session me save kiya
        st.markdown(f"<div class='st-chat-message'>{response}</div>", unsafe_allow_html=True) # Assistant ka response display kiya
