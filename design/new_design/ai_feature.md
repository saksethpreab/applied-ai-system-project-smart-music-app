# 🎯 Project Goal Summary: Agentic Smart Music Recommender

## The Primary Objective
To evolve a foundational AI prototype into a cohesive, end-to-end applied AI system that curates highly accurate music playlists based on natural language user prompts (e.g., moods, activities, or specific vibes). 

## The Advanced AI Feature (Agentic Workflow)
Instead of relying on a single, unpredictable LLM request, the system implements an **Agentic "Plan-Act-Check" Workflow**. This multi-step reasoning pipeline gives the AI autonomy to:
1. **Analyze:** Deconstruct the user's prompt to identify the ideal tempo, genre, and mood.
2. **Draft:** Generate a preliminary playlist based on those strict parameters and available songs in a csv file.
3. **Self-Correct:** Review the drafted songs against the original prompt to catch hallucinations or mismatched vibes, replacing them before presenting the final output to the user.

## The Academic & Reliability Goals
To fulfill the requirements of the Foundations of AI Engineering final project, this system will also:
* **Demonstrate Reliability:** Incorporate built-in guardrails, error handling (like JSON format enforcement), and evaluation metrics to prove the AI's output is consistent and trustworthy.
* **Provide Transparency:** Output observable intermediate steps (showing the AI's "thought process") to demystify how the recommendations are chosen.
* **Be Fully Reproducible:** Feature clear documentation, setup instructions, and an architectural diagram so anyone can clone the repository, run the code, and get consistent results.