from langchain_ollama import ChatOllama
from typing import TypedDict
from langgraph.graph import StateGraph,START,END

class BlogState(TypedDict):
    title: str
    outline: str
    review: str
    content: str
    attempts: int

model = ChatOllama(model="tinyllama:latest")

MAX_RETRIES = 4

graph = StateGraph(BlogState)

def create_outline(state: BlogState):
    title = state['title']

    prompt = f"create a detailed outline for following title :-  {title}"

    outline = model.invoke(prompt)

    return {'outline': outline}

def review_outline(state: BlogState):
    outline = state['outline']

    prompt = f"""
    Review the following outline.
    Provide the detailed feedback.
    Outline:{outline}
    """

    review = model.invoke(prompt)

    return {'review': review }

def regenerate_outline(state: BlogState):
    outline = state['outline']
    title = state['title']
    review = state['review']

    prompt = f"""
    Regenerate the detailed outline for following 
    title :-  {title}
    outline :-  {outline}
    review :-  {review}
    """
    return {'outline': outline }

def create_blog(state: BlogState):
    outline = state['outline']
    title = state['title']

    prompt = f"""
        Create a detailed blog where blog title :-  {title} 
        followed by outline :-  {outline}
    """
    content = model.invoke(prompt)

    return { 'content': content }

graph.add_node('create_outline', create_outline)
graph.add_node('review_outline', review_outline)
graph.add_node('regenerate_outline' , regenerate_outline)
graph.add_node('create_blog', create_blog)

graph.add_edge(START , 'create_outline')
graph.add_edge('create_outline' , 'review_outline')
graph.add_edge('review_outline', 'regenerate_outline')
graph.add_edge('regenerate_outline' , 'create_blog')
graph.add_edge('create_blog' , END)

workflow = graph.compile()

user_input = input("\nYou:")
initial_state = {'title':user_input}
final_workflow = workflow.invoke(initial_state)
print(final_workflow['content'])