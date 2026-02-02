#!/bin/bash

echo "================================================"
echo "🚀 RAG Chatbot - Quick Start Guide"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python $(python3 --version) detected"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo ""
    echo "📝 Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "🔑 Please edit .env and add your Groq API key:"
    echo "   1. Get free key from: https://console.groq.com"
    echo "   2. Open .env file"
    echo "   3. Replace 'your_groq_api_key_here' with your actual key"
    echo "   4. Save the file"
    echo ""
    read -p "Press Enter when you've added your API key..."
fi

echo "✅ .env file exists"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install requirements
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"
echo ""

# Run the app
echo "🚀 Starting Streamlit app..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "📖 Instructions:"
echo "1. Upload a PDF in the sidebar"
echo "2. Click 'Process Documents'"
echo "3. Ask questions in the main area"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""
echo "================================================"
echo ""

streamlit run app.py