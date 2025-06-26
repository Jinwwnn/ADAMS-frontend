import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import importlib
import inspect
import json
import logging
import uuid
import time
from pathlib import Path

from execution_pipeline.execution_pipeline import ExecutionPipeline, CompoundScoreExecutionPipeline
from evaluator.base_evaluator import RAGEvaluator
from utils.llm import OpenAIClientLLM, LLMClient
from api.models import (
    EvaluationRequest, 
    EvaluationResult, 
    EvaluationProgress, 
    MetricResult,
    AvailableEvaluator,
    LLMProvider,
    DataAugmentationRequest,
    DataAugmentationResult,
    EvaluationHistory,
    SaveEvaluationRequest
)

logger = logging.getLogger(__name__)

class EvaluationService:
    
    def __init__(self):
        self.current_progress: Optional[EvaluationProgress] = None
        self.start_time = None
        self.history_file = Path("backend/evaluation_history.json")
        self._ensure_history_file()
        self.available_evaluators = self._get_available_evaluators()
        
    def _ensure_history_file(self):
        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump([], f)
    
    def _get_available_evaluators(self) -> List[AvailableEvaluator]:
        try:
            module = importlib.import_module("evaluator.evaluators")
            evaluators = []
            
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if (issubclass(cls, RAGEvaluator) and 
                    cls.__module__ == module.__name__ and 
                    cls.__name__.endswith("Evaluator") and 
                    cls is not RAGEvaluator):
                    
                    # get evaluator description
                    description_info = cls.description() if hasattr(cls, 'description') else {}
                    
                    evaluators.append(AvailableEvaluator(
                        name=cls.__name__,
                        class_name=cls.__name__,
                        description=description_info.get('description', ''),
                        parameters=description_info.get('parameters', {}),
                        default_weight=1.0
                    ))
            
            return evaluators
        except Exception as e:
            logger.error(f"get evaluators failed: {e}")
            return []
    
    def get_available_evaluators(self) -> List[AvailableEvaluator]:
        return self.available_evaluators
    
    def _get_llm_client(self, provider: LLMProvider, model_name: str = None, base_url: str = None) -> LLMClient:
        if provider == LLMProvider.OPENAI:
            return OpenAIClientLLM(
                model=model_name or "gpt-4o-mini",
                base_url=base_url or "https://api.openai.com/v1/"
            )
        elif provider == LLMProvider.QWEN:
            return OpenAIClientLLM(
                model=model_name or "qwen-plus",
                base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        elif provider == LLMProvider.LOCAL:
            return OpenAIClientLLM(
                model=model_name or "local-model",
                base_url=base_url or "http://localhost:8000/v1"
            )
        else:
            # default using OpenAI
            return OpenAIClientLLM(
                model=model_name or "gpt-4o-mini",
                base_url=base_url or "https://api.openai.com/v1/"
            )
    
    def _get_evaluator_classes(self, selected_evaluators: List[str] = None) -> List[type]:
        module = importlib.import_module("evaluator.evaluators")
        evaluator_classes = []
        
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if (issubclass(cls, RAGEvaluator) and 
                cls.__module__ == module.__name__ and 
                cls.__name__.endswith("Evaluator") and 
                cls is not RAGEvaluator):
                
                if selected_evaluators is None or cls.__name__ in selected_evaluators:
                    evaluator_classes.append(cls)
        
        return evaluator_classes
    
    async def start_evaluation(self, request: EvaluationRequest) -> str:
        try:
            df = pd.DataFrame(request.dataset)
            
            evaluator_classes = self._get_evaluator_classes(request.selected_evaluators)
            
            if not evaluator_classes:
                raise ValueError("no available evaluators")
            
            llm_client = self._get_llm_client(
                request.llm_provider, 
                request.model_name, 
                request.base_url
            )
            
            self.current_progress = EvaluationProgress(
                total_samples=len(df),
                processed_samples=0,
                current_evaluator="starting",
                status="starting", 
                progress_percentage=0.0
            )
            
            if request.metric_weights:
                evaluators_with_weights = [
                    (cls, request.metric_weights.get(cls.__name__, 1.0)) 
                    for cls in evaluator_classes
                ]
                pipeline = CompoundScoreExecutionPipeline(evaluators_with_weights)
                result = await pipeline.run_pipeline_with_weight(
                    dataset_df=df,
                    llm_class=type(llm_client),
                    model=request.model_name,
                    base_url=request.base_url
                )
            else:
                pipeline = ExecutionPipeline(evaluator_classes)
                result = await pipeline.run_pipeline(
                    dataset_df=df,
                    llm_class=type(llm_client),
                    model=request.model_name,
                    base_url=request.base_url
                )
            
            evaluation_result = self._process_results(result, evaluator_classes, request.metric_weights)
            
            self.current_progress.status = "completed"
            self.current_progress.progress_percentage = 100.0
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"evaluation failed: {e}")
            if self.current_progress:
                self.current_progress.status = "error"
            raise e
    
    def _process_results(self, result_df: pd.DataFrame, evaluator_classes: List[type], metric_weights: Dict[str, float] = None) -> EvaluationResult:
        metrics = []
        
        original_columns = {'Question', 'Reference_Answer', 'Model_Answer', 'Context'}
        eval_columns = [col for col in result_df.columns if col not in original_columns]
        
        # calculate each metric's average score
        for evaluator_cls in evaluator_classes:
            evaluator_name = evaluator_cls.__name__
            
            matching_columns = [col for col in eval_columns if evaluator_name.lower().replace('evaluator', '') in col.lower()]
            
            if matching_columns:
                col_name = matching_columns[0]
                if col_name in result_df.columns:
                    scores = pd.to_numeric(result_df[col_name], errors='coerce')
                    avg_score = scores.mean() if not scores.isna().all() else 0.0
                    
                    description_info = evaluator_cls.description() if hasattr(evaluator_cls, 'description') else {}
                    
                    metrics.append(MetricResult(
                        name=evaluator_name.replace('Evaluator', ''),
                        score=float(avg_score),
                        description=description_info.get('description', ''),
                        weight=metric_weights.get(evaluator_name, 1.0) if metric_weights else 1.0
                    ))
        
        # calculate final score
        if metric_weights and 'Final_Score' in result_df.columns:
            final_score = float(result_df['Final_Score'].mean())
        else:
            # weighted average
            total_weighted = sum(metric.score * metric.weight for metric in metrics)
            total_weights = sum(metric.weight for metric in metrics)
            final_score = total_weighted / total_weights if total_weights > 0 else 0.0
        
        return EvaluationResult(
            metrics=metrics,
            final_score=final_score,
            processed_dataset=result_df.to_dict('records'),
            evaluation_summary={
                'total_samples': len(result_df),
                'avg_scores': {metric.name: metric.score for metric in metrics},
                'weights_used': metric_weights or {}
            },
            timestamp=datetime.now().isoformat()
        )
    
    def get_progress(self) -> Optional[EvaluationProgress]:
        return self.current_progress
    
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> bool:
        if not dataset:
            return False
        
        required_fields = {'Question', 'Reference_Answer', 'Model_Answer'}
        first_row = dataset[0]
        
        return all(field in first_row for field in required_fields)

    async def start_data_augmentation(self, request: DataAugmentationRequest) -> str:
        task_id = str(uuid.uuid4())
        self.current_progress = EvaluationProgress(
            task_id=task_id,
            status="running",
            progress=0.0,
            message="Starting data augmentation...",
            current_step="initialization"
        )
        
        asyncio.create_task(self._process_data_augmentation(task_id, request))
        return task_id
    
    async def _process_data_augmentation(self, task_id: str, request: DataAugmentationRequest):
        try:
            from execution_pipeline.execution_pipeline import ExecutionPipeline
            from data_annotator.annotators import (
                NumMistakesAnnotator,
                MistakeDistributionAnnotator, 
                MistakeAnswerGenerator
            )
            from utils.llm import OpenAIClientLLM
            
            self.current_progress.progress = 0.1
            self.current_progress.message = "Preparing augmentation pipeline..."
            
            df = pd.DataFrame(request.dataset)
            
            if 'question' in df.columns:
                df = df.rename(columns={
                    'question': 'Question',
                    'response': 'Reference_Answer', 
                    'documents': 'Context'
                })
            
            self.current_progress.progress = 0.2
            self.current_progress.message = "Configuring LLM..."
            
            llm_kwargs = {}
            if request.model_name:
                llm_kwargs['model'] = request.model_name
            if request.base_url:
                llm_kwargs['base_url'] = request.base_url
            
            annotators = [NumMistakesAnnotator, MistakeDistributionAnnotator, MistakeAnswerGenerator]
            pipeline = ExecutionPipeline(annotators)
            
            self.current_progress.progress = 0.3
            self.current_progress.message = "Executing data augmentation..."
            
            result_df = await pipeline.run_pipeline(
                dataset_df=df,
                save_path="./tmp_data",
                upload_to_hub=False,
                llm_class=OpenAIClientLLM,
                **llm_kwargs
            )
            
            self.current_progress.progress = 0.8
            self.current_progress.message = "Processing augmented results..."
            
            augmented_data = result_df.to_dict('records')
            
            mistake_summary = {
                'total_samples': len(augmented_data),
                'mistake_types_added': request.mistake_types or ['Entity_Error', 'Negation', 'Missing_Information'],
                'avg_mistakes_per_sample': request.num_mistakes
            }
            
            if not hasattr(self, 'augmentation_results'):
                self.augmentation_results = {}
            
            self.augmentation_results[task_id] = DataAugmentationResult(
                augmented_dataset=augmented_data,
                mistake_summary=mistake_summary,
                processing_time=time.time() - getattr(self, 'start_time', time.time()),
                timestamp=datetime.now().isoformat()
            )
            
            self.current_progress.progress = 1.0
            self.current_progress.status = "completed"
            self.current_progress.message = "Data augmentation completed!"
            
        except Exception as e:
            self.current_progress.status = "error"
            self.current_progress.message = f"Data augmentation failed: {str(e)}"
            logger.error(f"Data augmentation error: {e}")
    
    def get_augmentation_result(self, task_id: str) -> Optional[DataAugmentationResult]:
        return getattr(self, 'augmentation_results', {}).get(task_id)
    
    def validate_hf_dataset(self, dataset: List[Dict[str, Any]]) -> bool:
        if not dataset:
            return False
        
        required_fields = {'question', 'response', 'documents'}
        first_row = dataset[0]
        
        return all(field in first_row for field in required_fields)

    def save_evaluation_result(self, request: SaveEvaluationRequest) -> str:
        evaluation_id = str(uuid.uuid4())
        
        # 创建轻量级的结果摘要，不保存完整数据集
        results_summary = {
            "metrics": [metric.dict() for metric in request.evaluation_result.metrics],
            "final_score": request.evaluation_result.final_score,
            "evaluation_summary": request.evaluation_result.evaluation_summary,
            "timestamp": request.evaluation_result.timestamp,
            "sample_count": len(request.evaluation_result.processed_dataset)
        }
        
        history_entry = EvaluationHistory(
            id=evaluation_id,
            name=request.name,
            timestamp=datetime.now().isoformat(),
            dataset_info=request.dataset_info,
            llm_provider=request.llm_provider,
            model_name=request.model_name,
            evaluation_config=request.evaluation_config,
            results_summary=results_summary,  # 只保存摘要
            notes=request.notes
        )
        
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
        
        history.append(history_entry.dict())
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        return evaluation_id
    
    def get_evaluation_history(self) -> List[EvaluationHistory]:
        try:
            with open(self.history_file, 'r') as f:
                history_data = json.load(f)
            
            return [EvaluationHistory(**entry) for entry in history_data]
        except:
            return []
    
    def get_evaluation_by_id(self, evaluation_id: str) -> Optional[EvaluationHistory]:
        history = self.get_evaluation_history()
        for entry in history:
            if entry.id == evaluation_id:
                return entry
        return None
    
    def delete_evaluation(self, evaluation_id: str) -> bool:
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            history = [entry for entry in history if entry['id'] != evaluation_id]
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            return True
        except:
            return False
    
    def update_evaluation_notes(self, evaluation_id: str, notes: str) -> bool:
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            for entry in history:
                if entry['id'] == evaluation_id:
                    entry['notes'] = notes
                    break
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            return True
        except:
            return False


evaluation_service = EvaluationService() 