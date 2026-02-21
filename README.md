# 🛡️ Autonomous Explainable Intrusion Detection System

**Deep Learning + Feature Gradients + Graph Correlation + LLM for Network Security**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start

### **Run on Google Colab (Recommended)**

1. Open `IDS_Colab_HuggingFace.ipynb` in [Google Colab](https://colab.research.google.com)
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells
4. Download results

### **Run Locally**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ids-explainable-agent.git
cd ids-explainable-agent

# Setup virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python pipeline.py --samples 5
```

## 📋 Features

- ✅ **Hybrid CNN-LSTM Model** - Selected as best model after comparison experiments
- ✅ **Feature Gradient Explainability** - Gradient-based feature attribution
- ✅ **Graph Correlation Layer** - Structural attack similarity graph
- ✅ **HuggingFace LLM** - Natural language explanations (Flan-T5)
- ✅ **Risk Scoring** - Automated threat assessment
- ✅ **Decision Agent** - Automated response actions

## 🏗️ Architecture

```
Data → Preprocessing → Hybrid CNN-LSTM → Feature Gradients → Graph Correlation → Risk Scorer → LLM → Decision Agent
```

**Pipeline Components:**
1. **Data Loader** - Downloads and preprocesses IDS dataset
2. **CNN Model** - Predicts attack types with confidence scores
3. **Feature Gradient Explainer** - Generates saliency-based feature importance
4. **Graph Correlation Layer** - Builds class-to-class similarity graph from FG profiles
5. **Risk Scorer** - Computes risk scores based on attack severity
6. **LLM Explainer** - Generates human-readable explanations
7. **Decision Agent** - Executes automated responses

## Phase 3 – Graph Correlation Layer

The Graph Correlation Layer adds structural intelligence on top of pointwise model predictions.
Instead of only explaining one sample at a time, it learns class-level relationships by aggregating
Feature Gradient (FG) importance vectors per attack class.

How similarity is computed:
- For each class, we average FG vectors to form an attack profile.
- We compute cosine similarity between every pair of class profiles.
- If similarity is greater than `0.7`, we create an edge in an attack graph.
- Nodes represent attack classes; edge weight is the similarity score.

Why this matters:
- Captures latent attack-family proximity (e.g., classes with similar decision signatures).
- Supports memory retrieval by enabling neighbor-class lookup from graph structure.
- Upgrades the architecture from pure DL prediction to DL + structural reasoning.

## 📊 Dataset

**Source:** [IDS Intrusion CSV](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)
- **Size:** 1M+ network traffic samples
- **Features:** 78 network flow features
- **Classes:** Benign, FTP-BruteForce, SSH-Bruteforce

## 🔧 Requirements

- Python 3.11+
- TensorFlow 2.x
- scikit-learn
- networkx
- transformers (HuggingFace)
- pandas, numpy

## 📖 Usage

### Basic Usage

```python
from pipeline import IDSPipeline

# Create pipeline
pipeline = IDSPipeline(use_ollama=False)  # Uses HuggingFace

# Run on 5 samples
results = pipeline.run_pipeline(num_samples=5)

# Results saved to ids_results_TIMESTAMP.json
```

### Command Line

```bash
# Process 5 samples (use existing model)
python pipeline.py --samples 5

# Retrain model
python pipeline.py --samples 5 --retrain

# Disable LLM
python pipeline.py --samples 5 --no-ollama
```

## 📁 Project Structure

```
ids-explainable-agent/
├── data/
│   └── loader.py              # Dataset loading & preprocessing
├── models/
│   ├── cnn_model.py           # 1D CNN architecture
│   └── trainer.py             # Model training
├── explainability/
│   ├── feature_gradient_explainer.py  # FG explanations
│   ├── graph_correlation.py   # Phase 3 attack correlation graph
│   └── risk_scorer.py         # Risk scoring
├── llm/
│   └── huggingface_client.py  # HuggingFace LLM client
├── agent/
│   └── decision_agent.py      # Automated decision making
├── pipeline.py                # Main pipeline orchestrator
├── requirements.txt           # Python dependencies
└── IDS_Colab_HuggingFace.ipynb  # Google Colab notebook
```

## 🎯 Results

**Model Performance:**
- Accuracy: 99.98%
- Training Time: ~10-15 min (GPU) / ~20-40 min (CPU)

**Sample Output:**
```json
{
  "attack_type": "SSH-Bruteforce",
  "confidence": 0.9876,
  "risk_score": 8.5,
  "severity": "CRITICAL",
  "agent_decision": "BLOCK",
  "llm_explanation": "High-confidence SSH brute force attack detected..."
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Dataset: [Kaggle IDS Intrusion CSV](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)
- LLM: [Google Flan-T5](https://huggingface.co/google/flan-t5-base)
- Explainability: Feature Gradients (TensorFlow GradientTape)

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for Network Security Research**
