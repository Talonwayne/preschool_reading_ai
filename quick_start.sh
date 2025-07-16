#!/bin/bash
# Quick Start Script for Preschool Reading AI - Chained Voice Agent

echo "🎓 Preschool Reading AI - Chained Voice Agent"
echo "============================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OpenAI API key not found in environment variables."
    echo "   You can either:"
    echo "   1. Set it now: export OPENAI_API_KEY='your-key-here'"
    echo "   2. Create a .env file with: OPENAI_API_KEY=your-key-here"
    echo "   3. Continue and add it to the .env file that will be created"
    echo ""
    read -p "Continue setup? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies. Please check the error messages above."
    exit 1
fi

echo ""
echo "🔧 Running setup checks..."
python3 setup.py

echo ""
echo "🚀 Starting the Preschool Reading AI..."
echo "   First, you'll see a text demo of the chained agents."
echo "   Then you can optionally try the voice mode."
echo ""

python3 preschool_voice_agent.py 