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
    
    def get(self, endpoint: str) -> Dict:
        """Generic GET request"""
        try:
            response = self.session.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"GET {endpoint} failed: {e}")
            return None
    
    def post(self, endpoint: str, data: Dict) -> Dict:
        """Generic POST request"""
        try:
            response = self.session.post(f"{self.base_url}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"POST {endpoint} failed: {e}")
            return None
    
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
                timeout=120
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
                              llm_provider: str = "openai", model_name: str = "gpt-4o-mini",
                              agent_model: str = None) -> str:
        """Start Agent-based evaluation with dynamic metric selection"""
        try:
            # Use model_name as agent_model if not specified
            if agent_model is None:
                agent_model = model_name
                
            request_data = {
                "dataset": dataset,
                "user_criteria": user_criteria,
                "llm_provider": llm_provider,
                "model_name": model_name,
                "agent_model": agent_model
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
            result = response.json()
            
            # Log the raw result for debugging
            logger.info(f"Raw agent evaluation result: {result}")
            
            # Handle both success and error cases, extracting available information
            status = result.get("status", "unknown")
            selected_metrics = result.get("selected_metrics", {})
            discussion_summary = result.get("discussion_summary", "")
            chat_history = result.get("chat_history", [])
            evaluation_result = result.get("evaluation_result", None)
            
            if status == "error":
                logger.warning(f"Agent evaluation failed: {result.get('error')}")
                return {
                    "status": "error",
                    "error": result.get("error"),
                    "selected_metrics": selected_metrics,
                    "discussion_summary": discussion_summary,
                    "chat_history": chat_history,
                    "evaluation_result": evaluation_result,
                    "has_evaluation_data": evaluation_result is not None
                }
            else:
                logger.info(f"Agent evaluation successful! Selected metrics: {selected_metrics}")
                return {
                    "status": "success",
                    "selected_metrics": selected_metrics,
                    "discussion_summary": discussion_summary,
                    "chat_history": chat_history,
                    "evaluation_result": evaluation_result,
                    "has_evaluation_data": evaluation_result is not None,
                    "final_score": evaluation_result.get("final_score") if evaluation_result else None,
                    "processed_dataset": evaluation_result.get("processed_dataset") if evaluation_result else None
                }
            
        except Exception as e:
            logger.error(f"Failed to get agent result: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return fallback structure on any error
            fallback_metrics = {
                "FactualAccuracyEvaluator": 0.20,
                "FactualCorrectnessEvaluator": 0.15,
                "KeyPointCompletenessEvaluator": 0.20,
                "KeyPointHallucinationEvaluator": 0.15,
                "ContextRelevanceEvaluator": 0.10,
                "CoherenceEvaluator": 0.10,
                "EngagementEvaluator": 0.10
            }
            logger.warning(f"Using hardcoded fallback metrics due to exception: {fallback_metrics}")
            
            return {
                "status": "error", 
                "error": f"Failed to retrieve agent results: {str(e)}",
                "selected_metrics": fallback_metrics,
                "discussion_summary": "Failed to retrieve agent results due to connection error. Using default metrics.",
                "chat_history": [],
                "evaluation_result": None,
                "has_evaluation_data": False
            }
    
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

    def run_evaluation_pipeline(self, dataset: List[Dict], 
                               llm_provider: str = "openai",
                               model_name: str = "gpt-4o-mini",
                               metrics_config: List[Dict] = None) -> Dict:
        """Run evaluation pipeline with custom metrics configuration"""
        try:
            request_data = {
                "dataset": dataset,
                "llm_provider": llm_provider,
                "model_name": model_name,
                "metrics_config": metrics_config or []
            }
            
            response = self.session.post(
                f"{self.base_url}/evaluate-pipeline",
                json=request_data,
                timeout=300  # Allow more time for evaluation
            )
            response.raise_for_status()
            result = response.json()
            
            # Log the result for debugging
            logger.info(f"Evaluation pipeline result: {result}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to run evaluation pipeline: {e}")
            return {"status": "error", "error": str(e)}

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