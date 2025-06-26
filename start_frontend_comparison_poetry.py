#!/usr/bin/env python3
"""
RAG评估结果对比页面启动脚本 (Poetry版本)
"""

import subprocess
import sys
import os


def main():
    print("🚀 启动RAG评估结果对比页面 (Poetry环境)...")
    
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 检查Poetry是否可用
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 未找到Poetry，请先安装Poetry")
        return 1
    
    # 使用Poetry启动Streamlit应用（对比页面）
    try:
        subprocess.run([
            "poetry", "run", "streamlit", "run", 
            "frontend/app_comparison.py",
            "--server.port=8502",
            "--server.address=localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 对比页面已停止")
        return 0


if __name__ == "__main__":
    sys.exit(main()) 