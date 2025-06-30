from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class LLMProvider(str, Enum):
    """LLM judge provider"""
    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    LOCAL = "local"

class EvaluationRequest(BaseModel):
    dataset: List[Dict[str, Any]]  
    llm_provider: LLMProvider  
    model_name: Optional[str] = None  
    base_url: Optional[str] = None  
    selected_evaluators: Optional[List[str]] = None  
    metric_weights: Optional[Dict[str, float]] = None  

class EvaluationProgress(BaseModel):
    total_samples: int
    processed_samples: int
    current_evaluator: str
    status: str
    progress_percentage: float

class MetricResult(BaseModel):
    name: str
    score: float
    description: str
    weight: Optional[float] = 1.0

class EvaluationResult(BaseModel):
    metrics: List[MetricResult]
    final_score: float
    processed_dataset: List[Dict[str, Any]]
    evaluation_summary: Dict[str, Any]
    timestamp: str

class AvailableEvaluator(BaseModel):
    name: str
    class_name: str
    description: str
    parameters: Dict[str, str]
    default_weight: float = 1.0

class DataAugmentationRequest(BaseModel):
    dataset: List[Dict[str, Any]]
    llm_provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    
    # New configuration options
    selected_error_types: Optional[List[str]] = ["Entity_Error", "Negation", "Missing_Information", "Out_of_Reference", "Numerical_Error"]
    error_probabilities: Optional[List[float]] = [0.0, 0.7, 0.3]  # [0, 1, 2] errors
    include_key_points: Optional[bool] = True

class DataAugmentationResult(BaseModel):
    augmented_dataset: List[Dict[str, Any]]
    mistake_summary: Dict[str, Any]
    processing_time: float
    timestamp: str

class EvaluationHistory(BaseModel):
    id: str
    name: str
    timestamp: str
    dataset_info: Dict[str, Any]  # Only store dataset metadata: size, columns, etc.
    llm_provider: str
    model_name: Optional[str] = None
    evaluation_config: Dict[str, Any]
    results_summary: Dict[str, Any]
    notes: Optional[str] = None

class EvaluationComparison(BaseModel):
    evaluation_ids: List[str]
    comparison_metrics: List[str]
    timestamp: str

class SaveEvaluationRequest(BaseModel):
    evaluation_result: EvaluationResult
    name: str
    dataset_info: Dict[str, Any]
    llm_provider: str
    model_name: Optional[str] = None
    evaluation_config: Dict[str, Any]
    notes: Optional[str] = None

class EvaluationResultSummary(BaseModel):
    """Lightweight evaluation result summary for history storage"""
    metrics: List[MetricResult]
    final_score: float
    evaluation_summary: Dict[str, Any]
    timestamp: str
    sample_count: int  # Dataset sample count 