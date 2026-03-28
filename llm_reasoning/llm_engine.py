"""Prompt construction for lightweight LLM-based IDS reasoning."""


class LLMReasoningEngine:
    """Build a structured prompt from model outputs and feature vectors."""

    def generate_prompt(self, features, attack_name, confidence):
        clean_features = [round(float(f), 3) for f in features[:10]]
        confidence_text = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"

        return f"""
You are a cybersecurity expert analyzing network traffic.

Input:
- Predicted Attack Type: {attack_name}
- Confidence Score: {confidence_text} ({confidence:.2f})
- Feature Summary: {clean_features}

Analyze carefully using cybersecurity knowledge.

Tasks:
1. Explain what this attack type means
2. Explain why this traffic is suspicious (use patterns, not raw numbers)
3. Evaluate confidence (High / Medium / Low reliability)
4. Assign severity (Low / Medium / High)
5. Suggest action (Monitor / Block / Alert)

IMPORTANT:
- Do NOT repeat raw numbers
- Do NOT output numeric arrays
- Explain in simple and clear terms

Output format:

Attack:
Reason:
Confidence Analysis:
Severity:
Action:
""".strip()
