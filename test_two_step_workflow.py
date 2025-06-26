#!/usr/bin/env python3

import pandas as pd
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_dataset_loading():
    """Test loading the user's dataset"""
    dataset_path = Path("src/dataset/train-00000-of-00001.parquet")
    
    if not dataset_path.exists():
        print("❌ Dataset file not found")
        return False
    
    try:
        df = pd.read_parquet(dataset_path)
        print(f"✅ Dataset loaded successfully: {len(df)} samples")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = {'question', 'response', 'documents'}
        if required_columns.issubset(set(df.columns)):
            print("✅ Required columns present")
            print(f"📋 Sample data:")
            print(df.head(2)[['question', 'response']].to_string())
            return True
        else:
            missing = required_columns - set(df.columns)
            print(f"❌ Missing columns: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_backend_imports():
    """Test backend imports"""
    try:
        from api.models import DataAugmentationRequest, DataAugmentationResult
        from api.evaluation_service import evaluation_service
        print("✅ Backend models imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Backend import error: {e}")
        return False

def test_frontend_imports():
    """Test frontend imports"""
    try:
        sys.path.append(str(Path(__file__).parent / "frontend"))
        from backend_client import (
            test_backend_connection,
            start_data_augmentation,
            get_augmentation_result,
            poll_augmentation_progress,
            process_dataset_with_backend,
            get_evaluation_progress
        )
        print("✅ Frontend client imports successful")
        return True
    except ImportError as e:
        print(f"❌ Frontend import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🔬 Testing Two-Step Workflow Setup")
    print("=" * 50)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Backend Imports", test_backend_imports),
        ("Frontend Imports", test_frontend_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! Two-step workflow is ready.")
        print("\n🚀 To start the system:")
        print("   Terminal 1: poetry run python start_backend_poetry.py")
        print("   Terminal 2: poetry run python start_frontend_two_step.py")
        print("\n🌐 Access at: http://localhost:8501")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 