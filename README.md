# 🛡️ Autonomous Explainable Intrusion Detection System

**Deep Learning + SHAP + LLM for Network Security**

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

- ✅ **1D CNN Model** - Deep learning for intrusion detection (99%+ accuracy)
- ✅ **SHAP Explainability** - Feature importance analysis
- ✅ **HuggingFace LLM** - Natural language explanations (Flan-T5)
- ✅ **Risk Scoring** - Automated threat assessment
- ✅ **Decision Agent** - Automated response actions

## 🏗️ Architecture

```
Data → Preprocessing → 1D CNN → SHAP → Risk Scorer → LLM → Decision Agent
```

**Pipeline Components:**
1. **Data Loader** - Downloads and preprocesses IDS dataset
2. **CNN Model** - Predicts attack types with confidence scores
3. **SHAP Explainer** - Generates feature importance
4. **Risk Scorer** - Computes risk scores based on attack severity
5. **LLM Explainer** - Generates human-readable explanations
6. **Decision Agent** - Executes automated responses

## 📊 Dataset

**Source:** [IDS Intrusion CSV](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv)
- **Size:** 1M+ network traffic samples
- **Features:** 78 network flow features
- **Classes:** Benign, FTP-BruteForce, SSH-Bruteforce

## 🔧 Requirements

- Python 3.11+
- TensorFlow 2.x
- scikit-learn
- SHAP
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
│   ├── shap_explainer.py      # SHAP explanations
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
- Explainability: [SHAP](https://github.com/slundberg/shap)

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for Network Security Research**
