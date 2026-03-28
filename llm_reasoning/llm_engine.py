"""Prompt construction for lightweight LLM-based IDS reasoning."""


class LLMReasoningEngine:
    """Build a structured prompt from model outputs and feature vectors."""

    def generate_prompt(self, features, prediction, confidence):
        return f"""
You are an expert cybersecurity analyst.

Input:
- Predicted Class: {prediction}
- Confidence: {confidence:.2f}
- Feature Sample: {list(features[:10])}

Tasks:
1. Identify attack type
2. Explain reasoning using features
3. Evaluate confidence
4. Assign severity
5. Suggest action

Output format:

Attack:
Reason:
Confidence Analysis:
Severity:
Action:
""".strip()
