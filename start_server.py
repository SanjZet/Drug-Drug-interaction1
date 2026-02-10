"""
Start Web Server - Drug Interaction Checker
Launches the Flask API server with the web interface
"""

import subprocess
import sys
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import flask
        import flask_cors
        import torch
        import pandas
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-cors"])
        return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'data/trained_model.pt',
        'data/drug_graph.pt',
        'data/drugs.csv',
        'data/interactions.csv'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing required data files:")
        for f in missing:
            print(f"   - {f}")
        print("\n💡 Run 'python run_pipeline.py' first to generate data")
        return False
    
    print("✅ All data files present")
    return True

def main():
    print("\n" + "="*70)
    print("DRUG INTERACTION CHECKER - WEB SERVER".center(70))
    print("="*70 + "\n")
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return
    
    # Check data files
    print("\n🔍 Checking data files...")
    if not check_data_files():
        return
    
    print("\n" + "="*70)
    print("🚀 STARTING WEB SERVER".center(70))
    print("="*70 + "\n")
    
    print("📡 Server will start at: http://localhost:5000")
    print("🌐 Opening browser in 3 seconds...")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    # Wait a moment, then open browser
    time.sleep(3)
    try:
        webbrowser.open('http://localhost:5000')
    except:
        pass
    
    # Start Flask server
    try:
        subprocess.run([sys.executable, "api_server.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

if __name__ == '__main__':
    main()
