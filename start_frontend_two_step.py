#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    try:
        import streamlit
        import requests
        import pandas
        logger.info("✅ Dependencies checked")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        return False

def setup_environment():
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        os.chdir(frontend_dir)
        logger.info(f"📂 Changed directory to: {frontend_dir}")
    else:
        logger.warning("Frontend directory not found, staying in current directory")

def start_frontend():
    try:
        logger.info("🚀 Starting RAG Evaluation Two-Step Workflow frontend...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "app_two_step.py",
            "--server.port", "8501"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Frontend stopped")
    except Exception as e:
        logger.error(f"❌ Start frontend failed: {e}")

def main():
    print("=" * 60)
    print("RAG Evaluation Framework - Two-Step Workflow")
    print("=" * 60)
    
    if not check_dependencies():
        print("Please install required dependencies:")
        print("pip install streamlit requests pandas")
        return
    
    setup_environment()
    start_frontend()

if __name__ == "__main__":
    main() 