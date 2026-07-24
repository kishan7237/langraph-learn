import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot_workflow import workflows,llm

CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# st.session_state -> dist->

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type your message...')

if user_input:
    # First add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # this part for normal chatbot without streaming
    response = workflows.invoke({'messages':[HumanMessage(content = user_input)]}, config= CONFIG)
    ai_message = response['messages'][-1].content
    #First add the message to message_history
    st.session_state['message_history'].append({'role':'assistant', 'content':ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)

# ==========================================ui=======================================
# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Chat Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=100
    )
    st.title("🤖 Chat Agent")
    st.divider()
