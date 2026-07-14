from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOllama(model="tinyllama:latest")

class ChatState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    messages = state["message"]
    # send to llm
    response = llm.invoke(messages)
    # response store state
    return {'message': [response]}

checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile(checkpointer=checkpointer)

thread_id = 1

while True:
    user_input = input("Type here: ")
    print('User:', user_input)
    if user_input.strip().lower() in ['exit', 'quit', 'bye']:
        break

    config = {'configurable': {'thread_id': thread_id}}

    initial_state ={
        "message":[HumanMessage(content=user_input)]
    }
    final_state = workflow.invoke(initial_state, config = config)

    print('AI:', final_state['message'][-1].content)