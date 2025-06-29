import requests
import time
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class BackendClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def test_connection(self) -> bool:
        """Test if backend API is available"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Backend connection failed: {e}")
            return False
    
    def get_available_evaluators(self) -> List[Dict]:
        """Get list of available evaluators from backend"""
        try:
            response = self.session.get(f"{self.base_url}/evaluators")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get evaluators: {e}")
            return []
    
    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        """Validate dataset format for HuggingFace format (question, response, documents)"""
        try:
            response = self.session.post(
                f"{self.base_url}/validate-hf-dataset",
                json=dataset
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return {"valid": False, "message": str(e)}
    
    def start_evaluation(self, 
                        dataset: List[Dict], 
                        selected_evaluators: List[str] = None,
                        metric_weights: Dict[str, float] = None,
                        llm_provider: str = "openai",
                        model_name: str = "gpt-4o-mini") -> str:
        """Start evaluation task"""
        try:
            request_data = {
                "dataset": dataset,
                "selected_evaluators": selected_evaluators,
                "metric_weights": metric_weights,
                "llm_provider": llm_provider,
                "model_name": model_name
            }
            
            response = self.session.post(
                f"{self.base_url}/evaluate",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            return result.get("task_id")
        except Exception as e:
            logger.error(f"Failed to start evaluation: {e}")
            raise
    
    def get_evaluation_progress(self, task_id: str) -> Dict:
        """Get evaluation progress"""
        try:
            response = self.session.get(f"{self.base_url}/evaluate/{task_id}/progress")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get progress: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_evaluation_result(self, task_id: str) -> Dict:
        """Get evaluation result"""
        try:
            response = self.session.get(f"{self.base_url}/evaluate/{task_id}/result")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get result: {e}")
            raise
    
    def start_data_annotation(self, dataset: List[Dict]) -> Dict:
        """Process data annotation using original annotation pipeline"""
        try:
            request_data = {
                "dataset": dataset,
                "llm_provider": "openai",
                "model_name": "gpt-4o-mini"
            }
            
            response = self.session.post(
                f"{self.base_url}/annotation",
                json=request_data,
                timeout=120  # Allow more time for annotation processing
            )
            response.raise_for_status()
            result = response.json()
            return result  # Return the complete result directly
        except Exception as e:
            logger.error(f"Failed to process data annotation: {e}")
            raise
    
    def get_augmentation_result(self, task_id: str) -> Dict:
        """Get data augmentation result"""
        try:
            response = self.session.get(f"{self.base_url}/augment/{task_id}/result")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get augmentation result: {e}")
            raise
    
    def save_evaluation_result(self, 
                              name: str,
                              dataset: List[Dict],
                              metrics: List[Dict],
                              final_score: float,
                              llm_judge: str,
                              notes: str = "") -> str:
        """Save evaluation result to history"""
        try:
            request_data = {
                "name": name,
                "dataset": dataset,
                "metrics": metrics,
                "final_score": final_score,
                "llm_judge": llm_judge,
                "notes": notes
            }
            
            response = self.session.post(
                f"{self.base_url}/evaluation/save",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            return result.get("evaluation_id")
        except Exception as e:
            logger.error(f"Failed to save evaluation: {e}")
            raise
    
    def get_evaluation_history(self) -> List[Dict]:
        """Get evaluation history"""
        try:
            response = self.session.get(f"{self.base_url}/evaluation/history")
            response.raise_for_status()
            result = response.json()
            return result.get("history", [])
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

def get_backend_client():
    """Get backend client instance"""
    return BackendClient()

@st.cache_resource
def get_backend_client() -> BackendClient:
    return BackendClient()

def test_backend_connection() -> bool:
    client = get_backend_client()
    return client.test_connection()

def process_dataset_with_backend(dataset: List[Dict], 
                                llm_provider: str,
                                model_name: str = None,
                                selected_evaluators: List[str] = None,
                                metric_weights: Dict[str, float] = None) -> Optional[Dict]:
    client = get_backend_client()
    
    # validate dataset
    validation_result = client.validate_dataset(dataset)
    if not validation_result.get("valid", False):
        st.error(f"validate dataset failed: {validation_result.get('message', 'unknown error')}")
        return None
    
    # start evaluation
    task_id = client.start_evaluation(
        dataset=dataset,
        llm_provider=llm_provider,
        model_name=model_name,
        selected_evaluators=selected_evaluators,
        metric_weights=metric_weights
    )
    
    if not task_id:
        st.error("start evaluation task failed")
        return None
    
    st.info(f"evaluation task started, task id: {task_id}")
    
    # wait for completion
    result = client.wait_for_completion(task_id)
    return result

def get_available_evaluators_from_backend() -> List[Dict]:
    client = get_backend_client()
    return client.get_available_evaluators()

def start_data_augmentation(dataset, llm_provider, model_name=None, base_url=None, mistake_types=None, num_mistakes=3):
    try:
        response = requests.post(
            f"{BACKEND_URL}/augment",
            json={
                "dataset": dataset,
                "llm_provider": llm_provider,
                "model_name": model_name,
                "base_url": base_url,
                "mistake_types": mistake_types,
                "num_mistakes": num_mistakes
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_augmentation_result(task_id):
    try:
        response = requests.get(
            f"{BACKEND_URL}/augment/{task_id}/result",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def validate_hf_dataset(dataset):
    try:
        response = requests.post(
            f"{BACKEND_URL}/validate-hf-dataset",
            json=dataset,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_evaluation_progress(task_id):
    try:
        response = requests.get(
            f"{BACKEND_URL}/evaluate/{task_id}/progress",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def poll_augmentation_progress(task_id, progress_callback=None):
    import time
    
    while True:
        progress = get_evaluation_progress(task_id)
        
        if progress_callback:
            progress_callback(progress)
        
        if progress and progress.get('status') in ['completed', 'error']:
            break
        
        time.sleep(2)
    
    return progress

def save_evaluation_result(evaluation_result, name, dataset_info, llm_provider, model_name=None, evaluation_config=None, notes=None):
    try:
        response = requests.post(
            f"{BACKEND_URL}/evaluation/save",
            json={
                "evaluation_result": evaluation_result,
                "name": name,
                "dataset_info": dataset_info,
                "llm_provider": llm_provider,
                "model_name": model_name,
                "evaluation_config": evaluation_config or {},
                "notes": notes
            },
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_evaluation_history():
    try:
        response = requests.get(
            f"{BACKEND_URL}/evaluation/history",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_evaluation_by_id(evaluation_id):
    try:
        response = requests.get(
            f"{BACKEND_URL}/evaluation/history/{evaluation_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def delete_evaluation(evaluation_id):
    try:
        response = requests.delete(
            f"{BACKEND_URL}/evaluation/history/{evaluation_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def update_evaluation_notes(evaluation_id, notes):
    try:
        response = requests.put(
            f"{BACKEND_URL}/evaluation/history/{evaluation_id}/notes",
            json={"notes": notes},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def compare_evaluations(evaluation_ids):
    try:
        response = requests.post(
            f"{BACKEND_URL}/evaluation/compare",
            json=evaluation_ids,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None 