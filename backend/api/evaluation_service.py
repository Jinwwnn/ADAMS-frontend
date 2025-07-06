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
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

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
        self.augmentation_results = {}  # Store augmentation results
        
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
        elif provider == LLMProvider.DEEPSEEK:
            return OpenAIClientLLM(
                model=model_name or "deepseek-chat",
                base_url=base_url or "https://api.deepseek.com/v1"
            )
        elif provider == LLMProvider.MISTRAL:
            return OpenAIClientLLM(
                model=model_name or "mistral-7b-instruct",
                base_url=base_url or "https://api.mistral.ai/v1"
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
    
    async def start_evaluation(self, request: EvaluationRequest) -> EvaluationResult:
        try:
            # Set required environment variable for evaluators
            os.environ['ANSWER_TYPE'] = 'gold'  # Maps to 'generated_answer' field
            logger.info(f"Set ANSWER_TYPE environment variable to: {os.environ.get('ANSWER_TYPE')}")
            
            df = pd.DataFrame(request.dataset)
            logger.info(f"Original DataFrame columns: {list(df.columns)}")
            logger.info(f"Original DataFrame sample: {df.head(1).to_dict('records')}")
            
            # Smart field mapping and data preprocessing
            df_processed, available_fields = self._preprocess_dataset(df)
            logger.info(f"After preprocessing columns: {list(df_processed.columns)}")
            logger.info(f"Available fields: {available_fields}")
            
            # Filter evaluators based on available data fields
            evaluator_classes = self._get_evaluator_classes(request.selected_evaluators)
            filtered_evaluators, skipped_evaluators = self._filter_evaluators_by_requirements(
                evaluator_classes, available_fields
            )
            
            if skipped_evaluators:
                logger.info(f"Skipped evaluators due to missing fields: {[cls.__name__ for cls in skipped_evaluators]}")
            
            if not filtered_evaluators:
                raise ValueError("no compatible evaluators available for this dataset")
            
            logger.info(f"Using {len(filtered_evaluators)} compatible evaluators: {[cls.__name__ for cls in filtered_evaluators]}")
            
            llm_client = self._get_llm_client(
                request.llm_provider, 
                request.model_name, 
                request.base_url
            )
            
            self.current_progress = EvaluationProgress(
                total_samples=len(df_processed),
                processed_samples=0,
                current_evaluator="starting",
                status="starting", 
                progress_percentage=0.0,
                error=None
            )
            
            logger.info(f"Starting evaluation pipeline with {len(filtered_evaluators)} evaluators")
            
            try:
                if request.metric_weights:
                    # Filter weights to only include available evaluators
                    filtered_weights = {
                        cls.__name__: request.metric_weights.get(cls.__name__, 1.0) 
                        for cls in filtered_evaluators
                    }
                    evaluators_with_weights = [
                        (cls, filtered_weights.get(cls.__name__, 1.0)) 
                        for cls in filtered_evaluators
                    ]
                    pipeline = CompoundScoreExecutionPipeline(evaluators_with_weights)
                    result = await pipeline.run_pipeline_with_weight(
                        dataset_df=df_processed,
                        llm_class=type(llm_client),
                        model=request.model_name,
                        base_url=request.base_url
                    )
                else:
                    pipeline = ExecutionPipeline(filtered_evaluators)
                    result = await pipeline.run_pipeline(
                        dataset_df=df_processed,
                        llm_class=type(llm_client),
                        model=request.model_name,
                        base_url=request.base_url
                    )
                
                logger.info(f"Pipeline completed. Result columns: {list(result.columns)}")
                
            except Exception as pipeline_error:
                logger.error(f"Pipeline execution failed: {pipeline_error}")
                logger.error(f"Pipeline error type: {type(pipeline_error)}")
                import traceback
                logger.error(f"Pipeline traceback: {traceback.format_exc()}")
                
                # Store the specific error message for the progress API
                if self.current_progress:
                    self.current_progress.status = "error"
                    self.current_progress.error = str(pipeline_error)
                
                # Re-raise the exception to be caught by the outer try-catch
                raise pipeline_error
            
            evaluation_result = self._process_results(result, filtered_evaluators, request.metric_weights)
            
            # Add information about skipped evaluators
            if skipped_evaluators:
                evaluation_result.evaluation_summary['skipped_evaluators'] = [
                    {
                        'name': cls.__name__,
                        'reason': 'Missing required fields'
                    } for cls in skipped_evaluators
                ]
            
            self.current_progress.status = "completed"
            self.current_progress.progress_percentage = 100.0
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"evaluation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            if self.current_progress:
                self.current_progress.status = "error"
                self.current_progress.error = str(e)
            raise e
    
    def _preprocess_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
        """
        Preprocess dataset by mapping field names and handling data types.
        Returns processed dataframe and set of available fields.
        """
        df_processed = df.copy()
        
        # Standard field mapping from API format to evaluator format
        field_mapping = {
            'Question': 'question',
            'Reference_Answer': 'response', 
            'Model_Answer': 'generated_answer',
            'Context': 'documents'
        }
        
        # Apply standard field mapping
        df_processed = df_processed.rename(columns=field_mapping)
        logger.info(f"Applied field mapping: {field_mapping}")
        
        # Ensure generated_answer exists (fallback to response if missing)
        if 'generated_answer' not in df_processed.columns and 'response' in df_processed.columns:
            df_processed['generated_answer'] = df_processed['response']
            logger.info("Added generated_answer as copy of response")
        
        # Handle key_points field if it exists
        if 'key_points' in df_processed.columns:
            df_processed = self._process_key_points_field(df_processed)
        
        # Determine available fields for evaluator filtering
        available_fields = set(df_processed.columns)
        logger.info(f"Available fields after preprocessing: {available_fields}")
        
        return df_processed, available_fields
    
    def _process_key_points_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process key_points field to ensure it's in the correct format (List[str]).
        """
        logger.info("Processing key_points field")
        
        def convert_key_points(value):
            # Handle None or NaN values
            if value is None:
                return []
            
            # For scalar values, check if it's NaN
            try:
                if pd.isna(value):
                    return []
            except (ValueError, TypeError):
                # pd.isna() can fail on some types, just continue
                pass
            
            # If already a list, return as is
            if isinstance(value, list):
                return [str(item) for item in value]  # Ensure all items are strings
            
            # If string, try to parse as JSON list
            if isinstance(value, str):
                try:
                    # Try parsing as JSON
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                    else:
                        # If not a list, treat as single item
                        return [str(value)]
                except json.JSONDecodeError:
                    # If not valid JSON, split by common delimiters
                    if '|' in value:
                        return [item.strip() for item in value.split('|') if item.strip()]
                    elif ';' in value:
                        return [item.strip() for item in value.split(';') if item.strip()]
                    elif '\n' in value:
                        return [item.strip() for item in value.split('\n') if item.strip()]
                    else:
                        # Treat as single key point
                        return [str(value)]
            
            # For other types, convert to string and wrap in list
            return [str(value)]
        
        df['key_points'] = df['key_points'].apply(convert_key_points)
        
        # Log sample conversion
        logger.info(f"Key points sample after conversion: {df['key_points'].iloc[0] if len(df) > 0 else 'No data'}")
        
        return df
    
    def _filter_evaluators_by_requirements(self, evaluator_classes: List[type], available_fields: set) -> tuple[List[type], List[type]]:
        """
        Filter evaluators based on dataset field requirements.
        Returns (compatible_evaluators, skipped_evaluators).
        """
        compatible = []
        skipped = []
        
        # Define field requirements for evaluators
        evaluator_requirements = {
            'KeyPointEvaluators': {'key_points'},
            'KeyPointCompletenessEvaluator': {'key_points'},
            'KeyPointIrrelevantEvaluator': {'key_points'},
            'KeyPointHallucinationEvaluator': {'key_points'},
            # Most other evaluators only need basic fields
        }
        
        # Basic required fields that all evaluators need
        basic_required = {'question', 'response', 'documents'}
        
        # Check if basic requirements are met
        missing_basic = basic_required - available_fields
        if missing_basic:
            logger.error(f"Missing basic required fields: {missing_basic}")
            raise ValueError(f"Missing basic required fields: {missing_basic}")
        
        for evaluator_cls in evaluator_classes:
            evaluator_name = evaluator_cls.__name__
            
            # Check specific requirements
            required_fields = evaluator_requirements.get(evaluator_name, set())
            missing_fields = required_fields - available_fields
            
            if missing_fields:
                logger.info(f"Skipping {evaluator_name}: missing fields {missing_fields}")
                skipped.append(evaluator_cls)
            else:
                logger.info(f"Including {evaluator_name}: all requirements met")
                compatible.append(evaluator_cls)
        
        return compatible, skipped
    
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
            total_samples=len(request.dataset),
            processed_samples=0,
            current_evaluator="Data Augmentation",
            status="running",
            progress_percentage=0.0,
            error=None
        )
        
        # Process immediately instead of creating a task
        await self._process_data_augmentation(task_id, request)
        return task_id
    
    async def _process_data_augmentation(self, task_id: str, request: DataAugmentationRequest):
        try:
            self.current_progress.progress_percentage = 10.0
            self.current_progress.current_evaluator = "Preparing data augmentation"
            
            df = pd.DataFrame(request.dataset)
            
            # Convert to standard format for pipeline
            if 'question' in df.columns:
                df = df.rename(columns={
                    'question': 'Question',
                    'response': 'Reference_Answer', 
                    'documents': 'Context'
                })
            
            # Add Model_Answer column (copy from Reference_Answer for now)
            if 'Model_Answer' not in df.columns:
                df['Model_Answer'] = df['Reference_Answer']
            
            self.current_progress.progress_percentage = 50.0
            self.current_progress.current_evaluator = "Generating synthetic errors"
            
            # Create simple synthetic errors for testing
            augmented_data = []
            for idx, row in df.iterrows():
                # Original record
                original_record = row.to_dict()
                augmented_data.append(original_record)
                
                # Generate synthetic error version
                error_record = original_record.copy()
                error_record['Model_Answer'] = error_record['Reference_Answer'] + " [SYNTHETIC ERROR ADDED]"
                error_record['Error_Type'] = 'Synthetic_Error'
                error_record['Original_Answer'] = error_record['Reference_Answer']
                augmented_data.append(error_record)
            
            self.current_progress.progress_percentage = 90.0
            self.current_progress.current_evaluator = "Finalizing results"
            
            mistake_summary = {
                'total_samples': len(augmented_data),
                'mistake_types_added': ['Synthetic_Error'],
                'avg_mistakes_per_sample': 1
            }
            
            self.augmentation_results[task_id] = DataAugmentationResult(
                augmented_dataset=augmented_data,
                mistake_summary=mistake_summary,
                processing_time=time.time() - getattr(self, 'start_time', time.time()),
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Stored augmentation result for task {task_id}, total stored: {len(self.augmentation_results)}")
            
            self.current_progress.progress_percentage = 100.0
            self.current_progress.status = "completed"
            self.current_progress.current_evaluator = "Data augmentation completed"
            
        except Exception as e:
            self.current_progress.status = "error"
            self.current_progress.current_evaluator = f"Data augmentation failed: {str(e)}"
            logger.error(f"Data augmentation error: {e}")
    
    def get_augmentation_result(self, task_id: str) -> Optional[DataAugmentationResult]:
        logger.info(f"Looking for task {task_id}, available tasks: {list(self.augmentation_results.keys())}")
        return self.augmentation_results.get(task_id)
    
    async def run_annotation_pipeline(self, request: DataAugmentationRequest) -> DataAugmentationResult:
        """Run annotation pipeline using individual annotators (API-compatible version)"""
        try:
            start_time = time.time()
            logger.info("Starting annotation pipeline processing")
            
            # Set required environment variables for annotators
            os.environ['ANSWER_TYPE'] = 'gold'  # Required for annotators
            if not os.getenv('OPENAI_API_KEY'):
                logger.error("OPENAI_API_KEY environment variable is not set")
                raise ValueError("OpenAI API key is required for annotation pipeline")
            
            logger.info(f"Environment variables set - ANSWER_TYPE: {os.environ.get('ANSWER_TYPE')}")
            logger.info(f"OpenAI API Key present: {bool(os.getenv('OPENAI_API_KEY'))}")
            
            from data_annotator.annotators import (
                KeyPointAnnotator,
                NumMistakesAnnotator,
                MistakeDistributionAnnotator, 
                MistakeAnswerGenerator
            )
            from utils.llm import OpenAIClientLLM
            
            df = pd.DataFrame(request.dataset)
            logger.info(f"Created DataFrame: {len(df)} rows, columns: {list(df.columns)}")
            
            # The original pipeline expects: question, response, documents
            # No need to rename columns - use as is
            
            # Add required id column if missing
            if 'id' not in df.columns:
                df['id'] = range(len(df))
                logger.info("Added id column")
            
            # Setup LLM kwargs - force OpenAI compatible settings
            llm_kwargs = {
                'model': 'gpt-4o-mini',  # Use standard model name
                'base_url': 'https://api.openai.com/v1'  # Use standard OpenAI endpoint
            }
            
            logger.info(f"LLM configuration: {llm_kwargs}")
            
            # Use the original ExecutionPipeline exactly like the script
            pipeline = ExecutionPipeline([
                KeyPointAnnotator,
                NumMistakesAnnotator,
                MistakeDistributionAnnotator, 
                MistakeAnswerGenerator
            ])
            
            logger.info("Running original ExecutionPipeline")
            result_df = await pipeline.run_pipeline(
                dataset_df=df,
                llm_class=OpenAIClientLLM,
                **llm_kwargs
            )
            
            # Convert all data to JSON-serializable format
            result_df_clean = result_df.copy()
            for col in result_df_clean.columns:
                result_df_clean[col] = result_df_clean[col].astype(str)
            
            annotated_data = result_df_clean.to_dict('records')
            logger.info(f"Converted to dictionary: {len(annotated_data)} records")
            logger.info(f"Result columns: {list(result_df.columns)}")
            
            annotation_summary = {
                'total_samples': len(annotated_data),
                'annotation_types': ['key_points', 'num_mistake', 'mistake_distribution', 'Paraphrased', 'Incorrect', 'Error_Locations'],
                'original_samples': len(request.dataset),
                'columns_generated': list(result_df.columns),
                'llm_model': llm_kwargs.get('model', 'gpt-4o-mini')
            }
            
            logger.info("Annotation pipeline processing completed successfully")
            processing_time = time.time() - start_time
            
            return DataAugmentationResult(
                augmented_dataset=annotated_data,
                mistake_summary=annotation_summary,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Annotation pipeline error: {e}")
            import traceback
            logger.error(f"Full stack trace: {traceback.format_exc()}")
            raise e
    
    def validate_hf_dataset(self, dataset: List[Dict[str, Any]]) -> bool:
        if not dataset:
            return False
        
        required_fields = {'question', 'response', 'documents'}
        first_row = dataset[0]
        
        return all(field in first_row for field in required_fields)

    def save_evaluation_result(self, request: SaveEvaluationRequest) -> str:
        evaluation_id = str(uuid.uuid4())
        
        # Create lightweight result summary, not saving full dataset
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
            results_summary=results_summary,  # Only save summary
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