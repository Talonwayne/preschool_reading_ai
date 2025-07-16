#!/usr/bin/env python3
"""
Setup script for Preschool Reading AI - Chained Voice Agent Example
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_audio_devices():
    """Check if audio devices are available"""
    try:
        import sounddevice as sd
        
        # Check input devices
        input_devices = sd.query_devices(kind='input')
        if input_devices:
            print(f"✅ Audio input device found: {input_devices['name']}")
        else:
            print("⚠️  No audio input device found")
            
        # Check output devices
        output_devices = sd.query_devices(kind='output')
        if output_devices:
            print(f"✅ Audio output device found: {output_devices['name']}")
        else:
            print("⚠️  No audio output device found")
            
        return True
    except Exception as e:
        print(f"⚠️  Audio device check failed: {e}")
        return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found")
        return True
    else:
        print("⚠️  OpenAI API key not found")
        print("   Please set your API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   or create a .env file with your API key")
        return False

def create_env_file():
    """Create a .env file template"""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# OpenAI API Key\n")
            f.write("OPENAI_API_KEY=your-api-key-here\n")
        print("✅ Created .env file template")
        print("   Please edit .env and add your OpenAI API key")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup function"""
    print("🎓 Preschool Reading AI - Chained Voice Agent Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check audio devices
    check_audio_devices()
    
    # Check OpenAI API key
    key_exists = check_openai_key()
    
    # Create .env file if needed
    if not key_exists:
        create_env_file()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    if not key_exists:
        print("1. Add your OpenAI API key to the .env file")
        print("2. Run: python preschool_voice_agent.py")
    else:
        print("1. Run: python preschool_voice_agent.py")
    
    print("\nFor voice mode, make sure you have:")
    print("- A working microphone")
    print("- Audio speakers or headphones")
    print("- A quiet environment")

if __name__ == "__main__":
    main() 