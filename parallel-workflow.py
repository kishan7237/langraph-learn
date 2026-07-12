from typing import TypedDict, Annotated
import operator

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# -----------------------
# Model
# -----------------------

model = ChatOllama(model="tinyllama:latest")

# essay-evaluation-langgraph-agent
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", gt=0, le=10)


structured_model = model.with_structured_output(EvaluationSchema)

# -----------------------
# Essay
# -----------------------

essay = """
Artificial Intelligence (AI) is transforming the world by making machines capable of learning, reasoning, and solving problems like humans. From healthcare and education to agriculture and transportation, AI is improving efficiency and creating new opportunities. India, with its large pool of skilled professionals, strong IT industry, and growing digital infrastructure, is emerging as an important player in the global AI revolution.

One of India's greatest strengths in AI is its talented workforce. Every year, thousands of engineers, data scientists, and software developers graduate from Indian universities and contribute to AI research and development. Indian IT companies and startups are investing heavily in AI-based solutions for businesses, healthcare, finance, and education. The government's "Digital India" initiative and the National AI Strategy have further encouraged the adoption of AI technologies across the country.

AI is making a significant impact in various sectors of India. In healthcare, AI helps doctors diagnose diseases more accurately and quickly. In agriculture, AI-powered tools assist farmers by predicting weather conditions, monitoring crop health, and improving productivity. In education, AI enables personalized learning experiences and smart classrooms. AI is also being used in banking to detect fraud, improve customer service, and automate routine tasks.

India has also become a hub for AI startups and innovation. Many young entrepreneurs are developing AI-based products that solve real-world problems. Research institutions, universities, and technology companies are collaborating to advance AI research and create innovative applications. International companies are also setting up AI research centers in India because of its skilled workforce and favorable business environment.

Despite its progress, India faces several challenges in AI development. These include a shortage of high-quality AI research infrastructure, concerns about data privacy and cybersecurity, and the need to improve digital literacy in rural areas. There is also a need for ethical AI practices to ensure that AI systems are fair, transparent, and beneficial to society.

In conclusion, India has the potential to become a global leader in Artificial Intelligence. With continued investment in education, research, innovation, and responsible AI policies, the country can harness AI to improve the quality of life for its citizens and contribute significantly to global technological advancement. The future of AI in India is bright, and its role in shaping the digital world will continue to grow.
"""

# -----------------------
# LangGraph State
# -----------------------

class EssayState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_score: Annotated[list[int], operator.add]
    average_score: float


graph = StateGraph(EssayState)

# -----------------------
# Nodes
# -----------------------

def evaluate_language(state: EssayState):
    prompt = f"""
    Evaluate the language quality of the following essay.
    Provide detailed feedback and a score out of 10.

    Essay:
    {state["essay"]}
    """

    output = structured_model.invoke(prompt)

    return {
        "language_feedback": output.feedback,
        "individual_score": [output.score],
    }


def evaluate_analysis(state: EssayState):
    prompt = f"""
    Evaluate the depth of analysis of the following essay.
    Provide detailed feedback and a score out of 10.

    Essay:
    {state["essay"]}
    """

    output = structured_model.invoke(prompt)

    return {
        "analysis_feedback": output.feedback,
        "individual_score": [output.score],
    }


def evaluate_thought(state: EssayState):
    prompt = f"""
    Evaluate the clarity of thought of the following essay.
    Provide detailed feedback and a score out of 10.
    Essay:{state["essay"]}
    """

    output = structured_model.invoke(prompt)

    return {
        "clarity_feedback": output.feedback,
        "individual_score": [output.score],
    }


def final_evaluation(state: EssayState):

    prompt = f"""
Summarize the following feedback into one overall evaluation.

Language Feedback:
{state["language_feedback"]}

Analysis Feedback:
{state["analysis_feedback"]}

Clarity Feedback:
{state["clarity_feedback"]}
"""

    overall_feedback = model.invoke(prompt).content

    average_score = (
        sum(state["individual_score"])
        / len(state["individual_score"])
    )

    return {
        "overall_feedback": overall_feedback,
        "average_score": average_score,
    }


# -----------------------
# Build Graph
# -----------------------

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

graph.add_edge("final_evaluation", END)

# -----------------------
# Compile
# -----------------------

workflow = graph.compile()

# -----------------------
# Run
# -----------------------

initial_state = {
    "essay": essay
}

final_state = workflow.invoke(initial_state)

print(final_state)