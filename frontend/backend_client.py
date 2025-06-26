import requests
import time
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import streamlit as st

class BackendAPIClient:
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def test_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception:
            return False
    
    def get_available_evaluators(self) -> List[Dict]:
        try:
            response = self.session.get(f"{self.base_url}/evaluators")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"get evaluators failed: {e}")
            return []
    
    def validate_dataset(self, dataset: List[Dict]) -> Dict:
        try:
            response = self.session.post(
                f"{self.base_url}/validate-dataset",
                json=dataset
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"validate dataset failed: {e}")
            return {"valid": False, "message": str(e)}
    
    def start_evaluation(self, 
                        dataset: List[Dict], 
                        llm_provider: str,
                        model_name: str = None,
                        base_url: str = None,
                        selected_evaluators: List[str] = None,
                        metric_weights: Dict[str, float] = None) -> Optional[str]:
        try:
            request_data = {
                "dataset": dataset,
                "llm_provider": llm_provider,
                "model_name": model_name,
                "base_url": base_url,
                "selected_evaluators": selected_evaluators,
                "metric_weights": metric_weights
            }
            
            # remove None values
            request_data = {k: v for k, v in request_data.items() if v is not None}
            
            response = self.session.post(
                f"{self.base_url}/evaluate",
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("task_id")
            
        except Exception as e:
            st.error(f"start evaluation failed: {e}")
            return None
    
    def get_task_progress(self, task_id: str) -> Dict:
        try:
            response = self.session.get(f"{self.base_url}/evaluate/{task_id}/progress")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"get progress failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_task_result(self, task_id: str) -> Optional[Dict]:
        try:
            response = self.session.get(f"{self.base_url}/evaluate/{task_id}/result")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"get result failed: {e}")
            return None
    
    def wait_for_completion(self, task_id: str, max_wait_time: int = 300) -> Optional[Dict]:
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while time.time() - start_time < max_wait_time:
            progress_info = self.get_task_progress(task_id)
            
            if progress_info["status"] == "completed":
                progress_bar.progress(100)
                status_text.success("evaluation completed!")
                return self.get_task_result(task_id)
            elif progress_info["status"] == "error":
                status_text.error(f"evaluation failed: {progress_info.get('error', 'unknown error')}")
                return None
            elif progress_info["status"] == "running":
                # update progress bar
                if progress_info.get("progress"):
                    progress = progress_info["progress"]
                    if isinstance(progress, dict) and "progress_percentage" in progress:
                        progress_bar.progress(progress["progress_percentage"] / 100)
                        status_text.text(f"evaluating: {progress.get('current_evaluator', 'processing...')}")
                    else:
                        progress_bar.progress(0.5)
                        status_text.text("evaluating...")
                else:
                    progress_bar.progress(0.3)
                    status_text.text("initializing evaluation...")
            
            time.sleep(2)  # check every 2 seconds
        
        status_text.error("evaluation timeout")
        return None

@st.cache_resource
def get_backend_client() -> BackendAPIClient:
    return BackendAPIClient()

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