"""Prompt construction for lightweight LLM-based IDS reasoning."""


class LLMReasoningEngine:
    """Build a structured prompt from model outputs and feature vectors."""

    def generate_prompt(
        self,
        features,
        attack_name,
        confidence,
        risk_score=None,
        top_features=None,
        memory_similarity=None,
        graph_weight=None,
        decision=None,
    ):
        clean_features = [round(float(f), 3) for f in features[:10]]
        confidence_text = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        feature_names = ", ".join(str(value) for value in (top_features or [])[:5]) or "derived network features"
        risk_text = f"{float(risk_score):.2f}" if risk_score is not None else "unknown"
        memory_text = f"{float(memory_similarity):.2f}" if memory_similarity is not None else "unknown"
        graph_text = f"{float(graph_weight):.2f}" if graph_weight is not None else "unknown"
        decision_text = str(decision) if decision is not None else "Monitor"

        return f"""
You are a cybersecurity expert analyzing network traffic.

Input:
- Predicted Attack Type: {attack_name}
- Confidence Score: {confidence_text} ({confidence:.2f})
- Risk Score: {risk_text}
- Top Features: {feature_names}
- Memory Similarity: {memory_text}
- Graph Correlation Weight: {graph_text}
- Recommended Action: {decision_text}
- Feature Summary: {clean_features}

Analyze carefully using cybersecurity knowledge.

Tasks:
1. Explain what this attack means in plain English
2. Explain why this sample is risky using the top features, memory similarity, and graph correlation
3. Mention whether confidence is high, medium, or low
4. End with a recommended action sentence

IMPORTANT:
- Output exactly 2 to 4 sentences
- Use human-readable cybersecurity language
- Do NOT output numeric arrays, Python lists, logits, or JSON
- Mention the top features by name
- End with: Recommended action: {decision_text}.
""".strip()
