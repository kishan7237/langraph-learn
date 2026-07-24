from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

llm = ChatOllama(model="qwen3:1.7b")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {'messages': [response]}

checkpoint = InMemorySaver()

graph= StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflows = graph.compile(checkpointer=checkpoint)

