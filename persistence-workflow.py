from langchain_ollama import ChatOllama
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOllama(model="tinyllama:latest")

class JokeState(TypedDict):
    topic:str
    joke:str
    explanation:str

def generate_joke(state:JokeState)-> dict:
    prompt = f"""
        You are a comedian.
        
        Generate one short joke about:
        
        Topic: {state['topic']}
    """
    response = llm.invoke(prompt).content

    return {'joke':response}

def generate_explanation(state:JokeState)-> dict:
    prompt = f"""
        Explain why the following joke is funny.
        
        Joke:{state['joke']}
    """
    response = llm.invoke(prompt).content

    return {'explanation': response}

graph = StateGraph(JokeState)

graph.add_node('generate_joke', generate_joke)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_joke')
graph.add_edge('generate_joke', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id":"1"}}
while True:
    user_input = input("\nEnter Joke topic:")
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if not user_input:
        print("Please enter a topic.")
        continue
    initial_state = {'topic':user_input}
    final_state = workflow.invoke(initial_state, config=config1)
    print(list(workflow.get_state_history(config1)))
    print(final_state['joke'])
    print(final_state['explanation'])

