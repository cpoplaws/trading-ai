#!/bin/bash
# Start Streamlit Dashboard

echo "Starting Trading AI Dashboard..."
echo "Dashboard will be available at: http://localhost:8501"
echo ""

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run streamlit
streamlit run streamlit_app.py
