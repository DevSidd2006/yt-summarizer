#!/usr/bin/env python3
"""
Local Setup Script for YouTube Summarizer
=========================================

This script helps set up the YouTube Summarizer for local development.
It will install dependencies and check if AI features are available.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def install_requirements():
    """Install basic requirements"""
    print("📦 Installing basic requirements...")
    if run_command("pip install -r requirements.txt"):
        print("✅ Basic requirements installed successfully!")
        return True
    else:
        print("❌ Failed to install basic requirements")
        return False

def install_ai_features():
    """Install AI/ML features for enhanced functionality"""
    print("\n🤖 Installing AI/ML features for enhanced functionality...")
    ai_packages = [
        "transformers>=4.35.2",
        "torch>=2.1.1", 
        "pytube>=15.0.0",
        "pydub>=0.25.1",
        "SpeechRecognition>=3.10.0",
        "faster-whisper>=0.10.0"
    ]
    
    success = True
    for package in ai_packages:
        print(f"  Installing {package.split('>=')[0]}...")
        if not run_command(f"pip install {package}"):
            print(f"  ❌ Failed to install {package}")
            success = False
        else:
            print(f"  ✅ {package.split('>=')[0]} installed")
    
    return success

def check_installation():
    """Check if key packages are installed"""
    print("\n🔍 Checking installation...")
    
    packages_to_check = [
        "streamlit",
        "youtube_transcript_api", 
        "googletrans",
        "nltk",
        "pandas",
        "plotly"
    ]
    
    all_good = True
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Missing")
            all_good = False
    
    # Check AI packages
    ai_packages = ["transformers", "torch", "faster_whisper"]
    ai_available = True
    for package in ai_packages:
        try:
            __import__(package)
            print(f"  🤖 {package} (AI feature)")
        except ImportError:
            print(f"  ⚠️  {package} - AI features disabled")
            ai_available = False
    
    return all_good, ai_available

def main():
    print("🚀 YouTube Summarizer - Local Setup")
    print("=" * 50)
    
    # Install basic requirements
    if not install_requirements():
        print("\n❌ Setup failed. Please check your Python environment.")
        return
    
    # Ask about AI features
    while True:
        choice = input("\n🤖 Install AI features for enhanced functionality? (y/n): ").lower()
        if choice in ['y', 'yes']:
            install_ai_features()
            break
        elif choice in ['n', 'no']:
            print("⚠️  Skipping AI features. App will use fallback methods.")
            break
        else:
            print("Please enter 'y' or 'n'")
    
    # Check installation
    basic_ok, ai_ok = check_installation()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"  Basic Features: {'✅ Ready' if basic_ok else '❌ Failed'}")
    print(f"  AI Features: {'✅ Available' if ai_ok else '⚠️  Fallback mode'}")
    
    if basic_ok:
        print("\n🎉 Setup complete! You can now run the app with:")
        print("   python -m streamlit run streamlit_app.py")
        print("   OR")
        print("   Double-click start_app.bat")
    else:
        print("\n❌ Setup incomplete. Please resolve the issues above.")

if __name__ == "__main__":
    main()
