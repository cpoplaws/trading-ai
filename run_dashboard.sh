#!/usr/bin/env bash
# Launch the Trading-AI Streamlit Dashboard

echo "🚀 Starting Trading-AI Command Center Dashboard..."
echo ""
echo "📊 Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not installed"
    echo "📦 Installing dashboard dependencies..."
    pip install -r requirements.txt
fi

# Run the dashboard
cd "$(dirname "$0")"
streamlit run src/monitoring/dashboard.py
