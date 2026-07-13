from langchain_ollama import ChatOllama
from typing import TypedDict, Literal
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

generator_model = ChatOllama(model="qwen3:1.7b")
evaluator_model = ChatOllama(model="qwen3:1.7b")
optimizer_model = ChatOllama(model="qwen3:1.7b")

class TweetEvaluation(BaseModel):
    evaluation : Literal["approved", "needs_improvement"]  = Field(
        description="Final evalution result"
    )
    feedback: str = Field(
        description="feedback for the tweet"
    )

structured_evaluator_llm = evaluator_model.with_structured_output(TweetEvaluation)

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluator: Literal["approved","needs_improvement"]
    feedback: str
    iteration:int
    max_iterations: int

def generate_tweet(state: TweetState):
    #prompt
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

    Rules:
    - Do NOT use question-answer format.
    - Max 280 characters.
    - Use observational humor, irony, sarcasm, or cultural references.
    - Think in meme logic, punchlines, or relatable takes.
    - Use simple, day-to-day English.
    """)
    ]
    #send generator llm
    response = generator_model.invoke(messages).content
    #return response
    return {'tweet':response}

def evaluate_tweet(state: TweetState):
    # prompt
    messages = [
        SystemMessage(
            content="""
        You are an extremely strict Twitter/X editor.

        Only approve tweets that are exceptional.

        Reject at least 80% of tweets.

        If there is ANY weakness in humor, originality, virality,
        or punchiness, return:

        evaluation = "needs_improvement"

        Only use "approved" if you genuinely believe the tweet is
        ready to go viral.
        """
        ),
        HumanMessage(
            content=f"""
    Evaluate the following tweet:

    Tweet: "{state['tweet']}"

    Use the criteria below to evaluate the tweet:

    1. Originality - Is this fresh, or have you seen it a hundred times before?
    2. Humor - Did it genuinely make you smile, laugh, or chuckle?
    3. Punchiness - Is it short, sharp, and scroll-stopping?
    4. Virality Potential - Would people retweet or share it?
    5. Format - Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

    Auto-reject if:
    - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
    - It exceeds 280 characters
    - It reads like a traditional setup-punchline joke
    - Don't end with generic, throwaway, or deflating lines that weaken the humor (e.g., "Masterpieces of the auntie-uncle universe" or vague summaries)

    ### Respond ONLY in structured format:
    - evaluation: "approved" or "needs_improvement"
    - feedback: One paragraph explaining the strengths and weaknesses
    """
        )
    ]
    # send generator llm
    response = structured_evaluator_llm.invoke(messages)
    # return response
    return {'evaluator':response.evaluation, 'feedback':response.feedback}

def optimize_tweet(state: TweetState):
    # prompt
    messages = [
        SystemMessage(content = "You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content = f"""Improve the tweet based on this feedback:'{state['feedback']}'
    Topic: '{state['topic']}'
    Original Tweet:{state['tweet']}
    Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.""")]
    # send generator llm
    response = optimizer_model.invoke(messages).content
    iteration = state['iteration']+1

    # return response
    return {'tweet':response, 'iteration':iteration}

def route_evaluation(state: TweetState):
    if state['evaluator'] == 'approved' or state['iteration'] >= state['max_iterations']:
        return 'approved'
    else:
        return 'needs_improvement'

graph = StateGraph(TweetState)
graph.add_node('generate_tweet', generate_tweet)
graph.add_node('evaluate_tweet', evaluate_tweet)
graph.add_node('optimize_tweet',optimize_tweet)

graph.add_edge(START, 'generate_tweet')
graph.add_edge('generate_tweet', 'evaluate_tweet')
# 3rd parameter says if flow approved the END and if it needs_approvement then call optimize_tweet
graph.add_conditional_edges('evaluate_tweet', route_evaluation, {'approved':END, 'needs_improvement':'optimize_tweet'})
graph.add_edge('optimize_tweet', 'evaluate_tweet')

workflow = graph.compile()

user_input = input("\nEnter your topic: ")

initial_state = {
    'topic': user_input,
    'iteration': 0,
    'max_iterations': 5,
}

final_state = workflow.invoke(initial_state)

print(final_state)