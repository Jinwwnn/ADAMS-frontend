#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_poetry():
    """Check if poetry is available"""
    try:
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Found {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    logger.error("❌ Poetry not found. Please install poetry first:")
    logger.error("curl -sSL https://install.python-poetry.org | python3 -")
    return False

def start_frontend_with_poetry():
    """Start frontend application using poetry"""
    try:
        logger.info("🚀 Starting Streamlit application with Poetry...")
        
        # Change to project root directory
        project_root = Path(__file__).parent
        os.chdir(project_root)
        
        # Start streamlit using poetry
        cmd = [
            "poetry", "run", "streamlit", "run",
            "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Frontend application stopped")
    except Exception as e:
        logger.error(f"❌ Start frontend failed: {e}")

def install_dependencies():
    """Install dependencies using poetry"""
    try:
        logger.info("📦 Installing dependencies...")
        subprocess.run(["poetry", "install"], check=True)
        logger.info("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("RAG-LLM-Metric Frontend Application (Poetry)")
    print("=" * 50)
    
    # Check poetry
    if not check_poetry():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Start frontend
    start_frontend_with_poetry()

if __name__ == "__main__":
    main() 