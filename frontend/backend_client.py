import requests
import time
import json
import pandas as pd
from typing import Dict, List, Any, Optional
import streamlit as st
import logging
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()

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
    
    def start_data_annotation(self, dataset: List[Dict], 
                             llm_provider: str = "openai",
                             model_name: str = "gpt-4o-mini", 
                             base_url: str = None,
                             selected_error_types: List[str] = None,
                             error_probabilities: List[float] = None,
                             include_key_points: bool = True) -> Dict:
        """Process data annotation using original annotation pipeline"""
        try:
            if selected_error_types is None:
                selected_error_types = ["Entity_Error", "Negation", "Missing_Information", "Out_of_Reference", "Numerical_Error"]
            if error_probabilities is None:
                error_probabilities = [0.0, 0.7, 0.3]
                
            request_data = {
                "dataset": dataset,
                "llm_provider": llm_provider,
                "model_name": model_name,
                "base_url": base_url,
                "selected_error_types": selected_error_types,
                "error_probabilities": error_probabilities,
                "include_key_points": include_key_points
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
            # Create proper request format matching backend SaveEvaluationRequest
            evaluation_result = {
                "metrics": metrics,
                "final_score": final_score,
                "processed_dataset": dataset,
                "evaluation_summary": {
                    "total_samples": len(dataset),
                    "llm_judge": llm_judge
                },
                "timestamp": datetime.now().isoformat()
            }
            
            request_data = {
                "evaluation_result": evaluation_result,
                "name": name,
                "dataset_info": {
                    "sample_count": len(dataset),
                    "type": "evaluation_dataset",
                    "columns": list(dataset[0].keys()) if dataset else []
                },
                "llm_provider": llm_judge.lower(),
                "model_name": llm_judge,
                "evaluation_config": {},
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
    
    def get_evaluation_by_id(self, evaluation_id: str) -> Dict:
        """Get evaluation by ID"""
        try:
            response = self.session.get(f"{self.base_url}/evaluation/history/{evaluation_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get evaluation: {e}")
            return {}
    
    def delete_evaluation(self, evaluation_id: str) -> bool:
        """Delete evaluation by ID"""
        try:
            response = self.session.delete(f"{self.base_url}/evaluation/history/{evaluation_id}")
            response.raise_for_status()
            result = response.json()
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to delete evaluation: {e}")
            return False
    
    def update_evaluation_notes(self, evaluation_id: str, notes: str) -> bool:
        """Update evaluation notes"""
        try:
            response = self.session.put(
                f"{self.base_url}/evaluation/history/{evaluation_id}/notes",
                json={"notes": notes}
            )
            response.raise_for_status()
            result = response.json()
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to update notes: {e}")
            return False
    
    def compare_evaluations(self, evaluation_ids: List[str]) -> Dict:
        """Compare multiple evaluations"""
        try:
            response = self.session.post(
                f"{self.base_url}/evaluation/compare",
                json=evaluation_ids
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to compare evaluations: {e}")
            return {}
    
    def start_agent_evaluation(self, dataset: List[Dict], user_criteria: str, 
                              llm_provider: str = "openai", model_name: str = "gpt-4o-mini") -> str:
        """Start Agent-based evaluation with dynamic metric selection"""
        try:
            request_data = {
                "dataset": dataset,
                "user_criteria": user_criteria,
                "llm_provider": llm_provider,
                "model_name": model_name
            }
            
            response = self.session.post(
                f"{self.base_url}/agent-evaluate",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            return result.get("task_id")
        except Exception as e:
            logger.error(f"Failed to start agent evaluation: {e}")
            raise
    
    def get_agent_evaluation_progress(self, task_id: str) -> Dict:
        """Get Agent evaluation progress"""
        try:
            response = self.session.get(f"{self.base_url}/agent-evaluate/{task_id}/progress")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get agent progress: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_agent_evaluation_result(self, task_id: str) -> Dict:
        """Get Agent evaluation result"""
        try:
            response = self.session.get(f"{self.base_url}/agent-evaluate/{task_id}/result")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get agent result: {e}")
            raise
    
    def update_evaluation_weights(self, dataset: List[Dict], metric_weights: Dict[str, float], 
                                 evaluator_results: Dict) -> Dict:
        """Update metric weights and recalculate scores"""
        try:
            request_data = {
                "dataset": dataset,
                "metric_weights": metric_weights,
                "evaluator_results": evaluator_results
            }
            
            response = self.session.post(
                f"{self.base_url}/evaluate/update-weights",
                json=request_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update weights: {e}")
            raise

@st.cache_resource
def get_backend_client() -> BackendClient:
    """Get backend client instance"""
    return BackendClient()

def test_backend_connection() -> bool:
    """Test backend connection"""
    client = get_backend_client()
    return client.test_connection()

def get_available_evaluators_from_backend() -> List[Dict]:
    """Get available evaluators from backend"""
    client = get_backend_client()
    return client.get_available_evaluators() 