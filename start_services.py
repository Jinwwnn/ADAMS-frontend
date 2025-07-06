#!/usr/bin/env python3
"""
RAG-LLM-Metric Frontend and Backend Service Startup Script
Starts backend API service and frontend Streamlit application
"""

import subprocess
import time
import os
import sys
import signal
from pathlib import Path

def check_requirements():
    """Check required dependencies"""
    try:
        import streamlit
        import fastapi
        import uvicorn
        print("✅ Basic dependency check passed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: poetry install or pip install streamlit fastapi uvicorn")
        sys.exit(1)

def check_env_file():
    """Check environment variables file"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Please create .env file and add the following variables:")
        print("OPENAI_API_KEY=your_openai_api_key")
        print("HF_TOKEN=your_huggingface_token")
        print("ANSWER_TYPE=gold")
        return False
    else:
        print("✅ .env file found")
        return True

def check_port_in_use(port):
    """Check if port is already in use"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except socket.error:
            return True

def kill_process_on_port(port):
    """Kill process running on specified port"""
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            pid = result.stdout.strip()
            subprocess.run(['kill', pid])
            time.sleep(2)
            print(f"✅ Stopped existing process on port {port}")
            return True
    except Exception as e:
        print(f"⚠️  Could not stop process on port {port}: {e}")
    return False

def start_backend():
    """Start backend API service"""
    # Check if port 8000 is already in use
    if check_port_in_use(8000):
        print("⚠️  Port 8000 is already in use")
        response = input("Stop existing process and restart? (Y/n): ")
        if response.lower() != 'n':
            if not kill_process_on_port(8000):
                print("❌ Failed to stop existing process. Please manually stop it and try again.")
                return None
        else:
            print("ℹ️  Using existing backend service")
            return "existing"
    
    print("🚀 Starting backend API service (port: 8000)...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", 
        "backend.api.app:app", 
        "--host", "localhost", 
        "--port", "8000",
        "--reload"
    ]
    
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return backend_process

def start_frontend():
    """Start frontend Streamlit application"""
    # Check if port 8501 is already in use
    if check_port_in_use(8501):
        print("⚠️  Port 8501 is already in use")
        response = input("Stop existing process and restart? (Y/n): ")
        if response.lower() != 'n':
            if not kill_process_on_port(8501):
                print("❌ Failed to stop existing process. Please manually stop it and try again.")
                return None
        else:
            print("ℹ️  Using existing frontend service")
            return "existing"
    
    print("🎨 Starting frontend Streamlit application (port: 8501)...")
    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]
    
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=Path.cwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return frontend_process

def wait_for_service(url, service_name, max_attempts=30):
    """Wait for service to start"""
    import requests
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} started successfully!")
                if "8000" in url:  # Backend API
                    try:
                        data = response.json()
                        if "message" in data:
                            print(f"   API Message: {data['message']}")
                    except:
                        pass
                return True
        except:
            pass
        
        if attempt < max_attempts - 1:
            print(f"⏳ Waiting for {service_name} to start... ({attempt + 1}/{max_attempts})")
            time.sleep(2)
    
    print(f"❌ {service_name} startup timeout")
    return False

def show_running_services():
    """Show currently running services"""
    try:
        import subprocess
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        backend_running = any('uvicorn' in line and '8000' in line for line in lines)
        frontend_running = any('streamlit' in line and '8501' in line for line in lines)
        
        print("\n📊 Service Status:")
        print(f"   Backend (8000): {'✅ Running' if backend_running else '❌ Stopped'}")
        print(f"   Frontend (8501): {'✅ Running' if frontend_running else '❌ Stopped'}")
        
        return backend_running, frontend_running
    except:
        return False, False

def main():
    """Main function"""
    print("🧠 ADAMS - RAG Evaluation System")
    print("=" * 50)
    
    # Check current service status
    backend_running, frontend_running = show_running_services()
    
    if backend_running and frontend_running:
        print("\n🎉 Both services are already running!")
        print("🌐 Frontend: http://localhost:8501")
        print("🔗 Backend: http://localhost:8000")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Check dependencies
    check_requirements()
    
    # Check environment variables
    if not check_env_file():
        print("\n⚠️  It's recommended to configure environment variables first, but you can continue starting services")
        response = input("Continue starting? (y/N): ")
        if response.lower() != 'y':
            return
    
    processes = []
    
    try:
        # Start backend
        backend_process = start_backend()
        if backend_process is None:
            print("❌ Failed to start backend. Exiting.")
            return
        
        if backend_process != "existing":
            processes.append(("Backend API", backend_process))
            # Wait for backend to start
            time.sleep(5)
        
        if wait_for_service("http://localhost:8000", "Backend API"):
            print("🔗 Backend API: http://localhost:8000")
        else:
            print("❌ Backend API failed to start properly")
            return
        
        # Start frontend
        frontend_process = start_frontend()
        if frontend_process is None:
            print("❌ Failed to start frontend. Exiting.")
            return
            
        if frontend_process != "existing":
            processes.append(("Frontend App", frontend_process))
            # Wait for frontend to start
            time.sleep(5)
        
        if wait_for_service("http://localhost:8501", "Frontend App"):
            print("🌐 Frontend App: http://localhost:8501")
        else:
            print("❌ Frontend App failed to start properly")
            return
        
        print("\n" + "=" * 50)
        print("🎉 All services started successfully!")
        print("🌐 Frontend access: http://localhost:8501")
        print("🔗 Backend API: http://localhost:8000")
        print("📚 API documentation: http://localhost:8000/docs")
        print("\n💡 Quick Test:")
        print("   1. Visit http://localhost:8501")
        print("   2. Click 'Test OpenAI API' button (top-right)")
        print("   3. Upload a dataset and run evaluation")
        print("\nPress Ctrl+C to stop all services")
        print("=" * 50)
        
        # Wait for user interruption
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping services...")
        
        for name, process in processes:
            try:
                print(f"⏹️  Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"🔪 Force terminating {name}...")
                process.kill()
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        print("✅ All services stopped")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        # Just show status and exit
        print("🧠 ADAMS - Service Status Check")
        print("=" * 50)
        show_running_services()
        
        try:
            import requests
            # Test backend
            try:
                response = requests.get("http://localhost:8000", timeout=3)
                if response.status_code == 200:
                    print("🔗 Backend API: ✅ Responding")
                else:
                    print("🔗 Backend API: ⚠️  Running but not responding properly")
            except:
                print("🔗 Backend API: ❌ Not accessible")
            
            # Test frontend
            try:
                response = requests.get("http://localhost:8501", timeout=3)
                if response.status_code == 200:
                    print("🌐 Frontend App: ✅ Responding")
                else:
                    print("🌐 Frontend App: ⚠️  Running but not responding properly")
            except:
                print("🌐 Frontend App: ❌ Not accessible")
                
        except ImportError:
            print("⚠️  Install requests library for full status check: pip install requests")
    else:
        main() 