from typing import TypedDict, Literal, Dict, Any

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# Use a model that supports structured output
model = ChatOllama(model="tinyllama:latest")


# -----------------------------
# Structured Output Schemas
# -----------------------------
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )


class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="Category of issue"
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="Emotional tone"
    )
    urgency: Literal["Low", "Medium", "High"] = Field(
        description="Urgency level"
    )


structured_sentiment = model.with_structured_output(SentimentSchema)
structured_diagnosis = model.with_structured_output(DiagnosisSchema)


# -----------------------------
# State
# -----------------------------
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: Dict[str, Any]
    response: str


# -----------------------------
# Nodes
# -----------------------------
def find_sentiment(state: ReviewState):
    review = state["review"]

    prompt = f"""
    Determine whether the following review is positive or negative.

    Review:
    "{review}"
    """

    sentiment = structured_sentiment.invoke(prompt)

    return {"sentiment": sentiment.sentiment}


def positive_response(state: ReviewState):
    review = state["review"]

    prompt = f"""
    Write a warm thank-you response for this review.

    Review:
    "{review}"

    Also politely ask the customer to leave feedback on our website.
    """

    response = model.invoke(prompt).content

    return {"response": response}


def run_diagnosis(state: ReviewState):
    review = state["review"]

    prompt = f"""
    Analyze the following negative review.

    Review:
    "{review}"

    Return:
    - issue_type
    - tone
    - urgency
    """

    diagnosis = structured_diagnosis.invoke(prompt)

    return {"diagnosis": diagnosis.model_dump()}


def negative_response(state: ReviewState):
    diagnosis = state["diagnosis"]

    prompt = f"""
    You are a customer support assistant.

    The customer reported:

    Issue Type: {diagnosis["issue_type"]}
    Tone: {diagnosis["tone"]}
    Urgency: {diagnosis["urgency"]}

    Write an empathetic and helpful response.
    """

    response = model.invoke(prompt).content

    return {"response": response}


# -----------------------------
# Router
# -----------------------------
def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"


# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment",check_sentiment)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)
graph.add_edge("positive_response", END)


workflow = graph.compile()


# -----------------------------
# Run
# -----------------------------
user_input = input("Review: ")

initial_state = {
    "review": user_input
}

result = workflow.invoke(initial_state)

print("\nFinal State:\n")
print(result)

print("\nGenerated Response:\n")
print(result["response"])