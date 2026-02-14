#!/bin/bash

# Cleanup Script for IDS Project
# Removes unused files and prepares for deployment

echo "🧹 Cleaning up IDS project..."

cd "$(dirname "$0")"

# Remove old/duplicate Colab notebooks
echo "Removing old Colab notebooks..."
rm -f IDS_Colab.ipynb
rm -f IDS_Complete_Colab.ipynb  
rm -f IDS_Standalone_Colab.ipynb

# Remove test/setup files
echo "Removing test and setup files..."
rm -f test_system.py
rm -f setup.sh
rm -f setup_venv.sh

# Remove system files
echo "Removing system files..."
rm -f .DS_Store
find . -name ".DS_Store" -delete

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Remove Ollama files (replaced with HuggingFace)
echo "Removing Ollama files..."
rm -f llm/ollama_client.py
rm -f llm/prompts.py

# Update .gitignore
echo "Updating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# System
.DS_Store
Thumbs.db

# Project Specific
saved_models/
*.keras
training_history.png
ids_results_*.json

# Data Cache
.cache/
data/*.csv

# Logs
*.log
EOF

echo "✓ Cleanup complete!"
echo ""
echo "📦 Ready for deployment:"
echo "  - Ollama code removed"
echo "  - HuggingFace client ready"
echo "  - Test files removed"
echo "  - Cache cleaned"
echo ""
echo "Next steps:"
echo "  1. git add ."
echo "  2. git commit -m 'Cleaned project for deployment'"
echo "  3. git push"
