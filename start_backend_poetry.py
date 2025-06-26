#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_poetry():
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

def setup_environment():
    """Setup environment variables"""
    os.environ.setdefault("ANSWER_TYPE", "Model_Answer")
    logger.info("✅ Environment setup completed")

def start_backend_with_poetry():
    """Start backend server using poetry"""
    try:
        logger.info("🚀 Starting backend API server with Poetry...")
        
        # Change to project root directory
        project_root = Path(__file__).parent
        os.chdir(project_root)
        
        # Start uvicorn server using poetry
        cmd = [
            "poetry", "run", "python", "-m", "uvicorn",
            "backend.api.app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped")
    except Exception as e:
        logger.error(f"❌ Start server failed: {e}")

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
    print("RAG-LLM-Metric Backend API Server (Poetry)")
    print("=" * 50)
    
    # Check poetry
    if not check_poetry():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Setup environment
    setup_environment()
    
    # Start server
    start_backend_with_poetry()

if __name__ == "__main__":
    main() 