from langgraph.graph import StateGraph, START, END
from typing import TypedDict , Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(model='tinyllama:latest')

class SentimentSchema(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="Sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type:Literal['UX','Performance', 'Bug', 'Support', 'Other'] = Field(description='The category of issue mentioned in review')
    tone:Literal['angry','frustrated','disappointed','calm'] = Field(description='The emotional tone expressed by the user')
    urgency:Literal['Low','Medium','High'] = Field(description='How urgent or critical the issue appears to be')


structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

class ReviewState(TypedDict):
    review: str
    sentiment: Literal['positive','negative']
    diagnosis: dict
    response: str

def find_sentiment(state: ReviewState):
    review = state['review']

    prompt = """
        For the following review find the sentiment.
        Review:'{review}'
    """

    sentiment = structured_model.invoke(prompt).sentiment

    return {'sentiment': sentiment}

def positive_response(state: ReviewState):
    review = state['review']

    prompt = """
    Write a warm thank-you message in response to is review.
    Review:'{review}'
    Also, kindly ask the user to leave the feedback on our website.
    """
    response = model.invoke(prompt).content

    return {'response': response}

def run_diagnosis(state: ReviewState):
    review = state['review']

    prompt = """
        Diagnose this negative review :\n\n'{review}'
        Return issue_type, tone, urgency  
    """

    response = structured_model2.invoke(prompt)

    return {'diagnosis': response.model.dump()}

def negative_response(state: ReviewState):
    issue_type = state['issue_type']
    tone = state['tone']
    urgency = state['urgency']

    prompt = """
        You are a support assistant.
        The user had a '{issue_type}' issue, sounded '{tone}' and marked urgency as '{urgency}'
        write an empathetic, helpful resolution message
    """

    response = model.invoke(prompt)

    return {'response': response}

def check_sentiment(state: ReviewState) -> Literal['positive_response','run_diagnosis']:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

graph  = StateGraph(ReviewState)
graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)

graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment', check_sentiment)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)
graph.add_edge('positive_response', END)


workflow = graph.compile()
user_input = input("\nYou:")
initial_state = {'review':user_input}
final_state = workflow.invoke(initial_state)
print(final_state)
