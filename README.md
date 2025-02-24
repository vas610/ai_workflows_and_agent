# ai_workflows_and_agent

This repository contains examples of different AI workflows implemented in Python, showcasing the use of language models (LLMs) for various tasks. The workflows include prompt chaining, parallelization, evaluator-optimizer, and orchestrator-worker patterns.


## Programs

### Prompt Chaining

The `prompt_chaining.py` script demonstrates a basic prompt chaining approach. It takes a user input, determines if it's a ticket booking request, extracts ticket information, and generates a confirmation message.

### Parallelization

The `parallizaton.py` script showcases how to use `asyncio` and LLM to parallelize tasks, such as booking a flight and a hotel room concurrently.

### Evaluator-Optimizer

The `evaluator_optimizer.py` script implements an evaluator-optimizer workflow. It generates a joke, evaluates it, and regenerates it based on the evaluation feedback until the joke passes the evaluation criteria.

### Orchestrator-Worker

The `orchestrator_worker.py` script demonstrates an orchestrator-worker pattern. It generates creative ideas for a script based on a given topic, using an orchestrator to plan and workers to write the script.

### Agent

The `agent.py` program demonstrates a basic AI agent that interacts with the user to book tickets. It uses a loop to continuously take user input, determine the intent, extract relevant information, and respond accordingly.

## Usage

To run these examples, you need to have Python 3.11 or 3.12 installed, along with the `ollama` and `pydantic` libraries. You can install the requirements and run the programs as mentionedbelow:

```bash
pip install uv # Or curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv # Create a virtual environment
source .venv/bin/activate # Activate the virtual environment
uv pip install Ollama==0.4.7 # Install the required package. Please note that Pydantic will be install as a dependent package for Ollama
ollama pull llama3.1
ollama pull phi3.4

prompt_chaining.py
```