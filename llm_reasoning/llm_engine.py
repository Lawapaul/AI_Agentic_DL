"""Prompt construction for lightweight LLM-based IDS reasoning."""


class LLMReasoningEngine:
    """Build a structured prompt from model outputs and feature vectors."""

    def generate_prompt(self, features, prediction, confidence):
        clean_features = [round(float(f), 3) for f in features[:10]]

        return f"""
You are a cybersecurity expert analyzing network traffic.

Input:
- Prediction: {prediction} (0 = Normal, 1 = Attack)
- Confidence: {confidence:.2f}
- Feature Summary: {clean_features}

Think step-by-step and analyze carefully.

Tasks:
1. Identify attack type
2. Explain reasoning using patterns (NOT raw numbers)
3. Evaluate confidence (high/low reliability)
4. Assign severity (Low/Medium/High)
5. Suggest action (Monitor / Block / Alert)

IMPORTANT:
- Do NOT repeat raw numbers
- Explain in simple cybersecurity terms

Output format:

Attack:
Reason:
Confidence Analysis:
Severity:
Action:
""".strip()
