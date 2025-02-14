#!.venv/bin/python3
# Orchestrator-Worker Workflow
from typing import List, Dict
from pydantic import BaseModel, Field
from ollama import ChatResponse, chat
import os
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


MODEL_NAME = "phi4"

# --------------------------------------------------------------#
# Define the data models                                        #
# --------------------------------------------------------------#


class Idea(BaseModel):
    """Idea defined by orchestrator"""

    idea_title: str = Field(description="Title of the idea")
    description: str = Field(description="What should this idea cover")


class OrchestratorPlan(BaseModel):
    """Orchestrator's structure and tasks"""

    topic: str = Field(description="Analysis of the ad topic")
    ideas: List[Idea] = Field(description="List of ideas to write")


class IdeaContent(BaseModel):
    """Script written by a AI worker"""

    idea_title: str = Field(description="Title of the idea")
    content: str = Field(description="Written content for the idea")


class SuggestedEdits(BaseModel):
    """Suggested edits for a section"""

    section_name: str = Field(description="Name of the section")
    suggested_edit: str = Field(description="Suggested edit")


class BestIdea(BaseModel):
    """Identify the best idea"""

    idea_title: str = Field(description="Title of the best idea")
    reason: str = Field(description="Reason for selecting the best idea")


# --------------------------------------------------------------#
#  Define prompts                                               #
# --------------------------------------------------------------#

ORCHESTRATOR_PROMPT = """
You are a distinguished scriptwriter with extensive experience in creating viral social media campaigns and commercials using short-films. 

Your task is to generate creative ideas for a script based on the following topic

TOPIC: {topic}

Please follow these steps to complete the task:

1. Analyze the topic:
   - Consider its key elements and potential angles for a viral campaign.
   - Think about the target audience and what might resonate with them.
   - Reflect on current trends that could be incorporated.
   - Identify potential social media platforms suitable for the campaign.

2. Generate multiple creative ideas for a viral social media campaign or commercial:
   - Ensure each idea is engaging, shareable, and aligned with the topic.
   - Consider the narrative flow and how different elements will work together.
   - Evaluate the viral potential of each idea.

3. Limit the number of ideas to: {num_ideas}


The output should include:

# Analysis
[2-3 sentences summarizing your analysis and structural approach]


# Idea Structure
## Idea 1
- idea_title: [5-10 word summary of the idea]
- description: [2-4 sentences describing the key topics and elements]

Return output as JSON
"""

IDEA_PROMPT = """
Write a script for a viral short-form video ad based on:
Topic: {topic}
Idea for the script: {idea}
Description: {description}

The script should have the following characteristics:
1. Engagement: Capture the audience's attention within the first few seconds.
2. Clarity: Clearly convey the message and call to action.
3. Creativity: Use innovative and captivating elements.
4. Strong visuals: Describe the visuals that will accompany the script.
5. Ending: End with a strong conclusion or call to action.
6. Length: Keep the script concise and impactful. The target length is 30-60 seconds.
7. Storytelling: Use a standard storytelling framework of building up tension and relieving it through the video with conflicts and resolutions

The output should have two attributes:
ideal_title: [5-10 word summary of the idea]
content: [The written content for the idea]

Return output as JSON
"""

FIND_BEST_IDEA_PROMPT = """
Review the ideas generated for a viral social media ad (short-film format) on the following topic:

Topic: {topic}


{ideas_text}


===========================================================


Based on the ideas provided above, please do the following:
1. Review every idea.
2. Identify the best idea based on creativity, engagement, and viral potential.
3. State the reason for selecting the best idea.

Return output as JSON
"""

# --------------------------------------------------------------#
# Implement orchestrator                                        #
# --------------------------------------------------------------#


class AdOrchestrator:
    def __init__(self):
        self.idea_content = {}

    def get_plan(self, topic: str, num_ideas: int = 3) -> OrchestratorPlan:
        """Get orchestrator's structure plan"""
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": ORCHESTRATOR_PROMPT.format(
                        topic=topic,
                        num_ideas=num_ideas,
                    ),
                }
            ],
            options={"temperature": 0.2},
            format=OrchestratorPlan.model_json_schema(),
        )
        result = OrchestratorPlan.model_validate_json(response.message.content)
        return result

    def write_script(self, topic: str, idea: Idea) -> IdeaContent:
        """Worker: Write a specific idea with context from previous ideas"""

        # Create context from previously written sections to avoid repetition of ideas
        previous_ideas = "\n\n".join(
            [
                f"Idea: {idea_title} \n{content.content}\n==========\n"
                for idea_title, content in self.idea_content.items()
            ]
        )

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": IDEA_PROMPT.format(
                        topic=topic,
                        idea=idea.idea_title,
                        description=idea.description,
                        previous_ideas=previous_ideas if previous_ideas else None,
                    ),
                }
            ],
            options={"temperature": 0.5},
            format=IdeaContent.model_json_schema(),
        )
        return IdeaContent.model_validate_json(response.message.content)

    def find_best_idea(self, topic: str) -> BestIdea:
        """Finalize the content of the ad script"""
        ideas_text = "\n\n".join(
            [
                f"=== Idea: {idea_title} ===\nContent: {content.content}\n\n\n"
                for idea_title, content in self.idea_content.items()
            ]
        )
        logger.info(ideas_text)

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": FIND_BEST_IDEA_PROMPT.format(
                        topic=topic, ideas_text=ideas_text
                    ),
                }
            ],
            options={"temperature": 0.2},
            format=BestIdea.model_json_schema(),
        )
        return BestIdea.model_validate_json(response.message.content)

    def generate_ideas(self, topic: str, num_ideas: int = 1) -> Dict:
        """Process to generate various ideas and find the best one"""
        logger.info(f"Writing ideas for viral marketing campaign: {topic}")

        # Get ad ideas structure
        plan = self.get_plan(topic, num_ideas)
        logger.info(f"Ad ideas structure: {plan.model_dump_json(indent=2)}")

        # Write script for each idea
        for idea in plan.ideas:
            logger.info(f"Writing idea: {idea.idea_title}")
            content = self.write_script(topic, idea)
            self.idea_content[idea.idea_title] = content

        # Review and polish
        logger.info("Finding the best idea")
        best_idea = self.find_best_idea(topic)

        return {"plan": plan, "ideas": self.idea_content, "best_idea": best_idea}


# --------------------------------------------------------------#
#  Execute workflow with an example                             #
# --------------------------------------------------------------#

if __name__ == "__main__":
    orchestrator = AdOrchestrator()

    topic = "A new smartphone name z1.0 from the company called Z"
    num_ideas = 3

    result = orchestrator.generate_ideas(topic=topic, num_ideas=num_ideas)
    logger.info("===Ideas===")
    for idea in result.get("ideas"):
        logger.info(f"Idea: {idea} \n")
        logger.info(f"Content: {result.get("ideas")[idea].content}")
        logger.info("=" * 100)
    logger.info("===Best Idea===")
    logger.info(f"Idea: {result.get("best_idea").idea_title}")
    logger.info(f"Reason: {result.get("best_idea").reason}")
    logger.info("=" * 100)