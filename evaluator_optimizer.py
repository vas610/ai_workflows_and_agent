#!.venv/bin/python3
# Evaluator-Optimizer Workflow
import logging
import os
from typing import Dict, List, Literal

from ollama import ChatResponse, chat
from pydantic import BaseModel, Field

# -----------------------------------------------------#
# Set up logging configuration                         #
# -----------------------------------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------#
# Define Variables                                     #
# -----------------------------------------------------#
MODEL_NAME_1 = "mistral-small"
MODEL_NAME_2 = "mistral-small"


# -----------------------------------------------------#
# Define the data models for each stage using Pydantic #
# -----------------------------------------------------#
class GeneratedJoke(BaseModel):
    "Joke generated and thoughts"

    thoughts: str = Field(
        description="Thought process followed for generating the joke"
    )
    joke: str = Field(description="Joke generated based on user input")


class JokeFeedback(BaseModel):
    "Feedback on the generated joke"

    evaluaton_result: Literal["PASS", "NEEDS IMPROVEMENT", "FAIL"] = Field(
        description="Set if the joke has passed, failed or needs improvement according to the given requirement"
    )
    feedback: str = Field(description="feedback on the joke")


# -----------------------------------------------------#
# Define Prompts                                       #
# -----------------------------------------------------#
evaluator_prompt = """
Evaluate the following joke for:
1. Structure :
    a. Setup : Ensure the setup clearly introduces the context, making the audience ready for the punchline. The joke should align with the user input.
    b. Punchline : The punchline should be both surprising and logically connected to the setup. It needs to balance originality with relatability.
2. Clarity and Relatability :
    a. The joke should be understandable to most of its intended audience, avoiding overly niche references unless it's a targeted group.
    b. Cultural context is crucial; humor can vary widely across different cultures and regions.
3. Delivery : Timing, tone, and body language significantly impact the effectiveness. Enthusiasm and rhythm in delivery enhance the joke's reception.
4. Relevance and Timeliness : Consider whether the joke resonates with the <user input> and (current events or is timeless). Avoid relying on outdated references that might become irrelevant or offensive over time.
5. Engagement : Determine if the joke aims to make people laugh, think, or both. Effective jokes engage their audience on at least one of these levels.

You should be evaluating only and not attemping to rewrite the joke.
Only output "PASS" if all criteria are met and you have no further suggestions for improvements.

Output your evaluation concisely in the following format.

evaluation: PASS, NEEDS_IMPROVEMENT, or FAIL

feedback: [What needs improvement and why.]


user input: {user_input}
joke: {joke}

Return output as JSON
"""

generator_prompt = """
Your goal is to write a job based on <user input>. If there are feedback 
from your previous generations, you should reflect on them to improve your solution



Feedback: {feedback}



Output your answer concisely with the following: 

thoughts: [Your understanding of the given user innput and feedback and how you plan to improve]

response: [Joke that you have generated]

Return as JSON
"""


# -----------------------------------------------------#
# Define functions for calling                         #
# -----------------------------------------------------#
def generate_joke(user_input: str, feedback: str) -> GeneratedJoke:
    """Generate a joke based on user input"""
    response = chat(
        model=MODEL_NAME_1,
        messages=[
            {
                "role": "system",
                "content": generator_prompt.format(
                    feedback=feedback if feedback else None
                ),
            },
            {"role": "user", "content": user_input},
        ],
        options={"temperature": 0.5},
        format=GeneratedJoke.model_json_schema(),
    )
    result = GeneratedJoke.model_validate_json(response.message.content)
    return result


def evaluate_joke(user_input: str, joke: str) -> JokeFeedback:
    """Evaluate a joke and provide feedback"""
    response = chat(
        model=MODEL_NAME_2,
        messages=[
            {
                "role": "system",
                "content": evaluator_prompt.format(user_input=user_input, joke=joke),
            },
        ],
        options={"temperature": 0},
        format=JokeFeedback.model_json_schema(),
    )
    result = JokeFeedback.model_validate_json(response.message.content)
    return result


# -----------------------------------------------------#
# Execute the workflow                                 #
# -----------------------------------------------------#


if __name__ == "__main__":
    # user Input
    user_input = "Write a joke about space travel"
    generated_joke = generate_joke(user_input, None)

    # Generate the first joke
    logger.info(f"Joke: {generated_joke.joke}")
    logger.debug(f"Thoughts: {generated_joke.thoughts}")

    # Get Feedback
    evaluation_feedback = evaluate_joke(user_input, generated_joke)
    logger.info(f"Evaluation: {evaluation_feedback.evaluaton_result}")
    logger.info(f"Feedback: {evaluation_feedback.feedback}")

    # Continue refining the joke until the joke passes the evaluation criteria
    while evaluation_feedback.evaluaton_result != "PASS":
        generated_joke = generate_joke(user_input, evaluation_feedback.feedback)
        logger.info(f"Regenerated Joke: {generated_joke.joke}")
        evaluation_feedback = evaluate_joke(user_input, generated_joke)
        logger.info(f"Evaluation: {evaluation_feedback.evaluaton_result}")
        logger.info(f"Feedback: {evaluation_feedback.feedback}")
