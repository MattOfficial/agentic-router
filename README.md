# Agentic Router

**Agentic Router** is a domain-agnostic, zero-shot classification and routing engine designed for modern LLM applications.

Instead of writing rigid `if/else` statements or hardcoding API logic, `Agentic Router` allows you to define complex routing topologies and classification guardrails using simple Python dictionaries and arrays. It dynamically compiles these into robust framework-agnostic prompts to intelligently route user queries to the correct database or pipeline.

## Features
- 🛣️ **Dynamic Topologies**: Route natural language queries across various workflows (e.g., Vector DB vs Graph DB vs SQL) by passing a simple dictionary of rules.
- 🛡️ **Custom Classification**: Inject arbitrary categories and guardrails (e.g., PII detection, Sentiment Analysis) into the agent's decision-making pipeline on the fly.
- 🔌 **Model Agnostic**: Built on top of LlamaIndex. Inject an OpenAI, Gemini, or even a local Ollama model directly into the constructor.

## Installation
```bash
pip install agentic-router
```

## Quick Start

### 1. Dynamic Query Routing
Define your project's unique routing rules, and the Agent will dynamically strictly enforce them:

```python
from agentic_router import AgenticRouter
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")
router = AgenticRouter(llm=llm)

# A Customer Support topology
support_topology = {
    "Tech_Support": "User is asking about a software bug or failure.",
    "Billing": "User is asking about refunds or credit cards."
}

route = router.route_query(
    query="I was charged twice on my visa.", 
    topology=support_topology, 
    fallback="Tech_Support"
)
print(route)  # Output: Billing
```

### 2. Custom Classification & Guardrails
Classify inputs on the fly strictly adhering to your defined arrays:

```python
# A tabletop RPG setting
rpg_vibes = ["Ominous", "Peaceful", "Combat", "Tavern"]
query = "The floorboards creak as three figures step out of the shadows with drawn daggers."

vibe = router.classify_vibe(query=query, valid_vibes=rpg_vibes)
print(vibe)  # Output: Ominous
```
