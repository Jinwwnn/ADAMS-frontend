from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import json
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

from api.models import (
    EvaluationRequest, 
    EvaluationResult, 
    EvaluationProgress,
    AvailableEvaluator,
    DataAugmentationRequest,
    DataAugmentationResult,
    SaveEvaluationRequest,
    EvaluationHistory
)
from api.evaluation_service import evaluation_service

app = FastAPI(
    title="RAG-LLM-Metric API",
    description="RAG-LLM-Metric API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

evaluation_tasks = {}
agent_evaluation_tasks = {}

@app.get("/")
async def root():
    return {
        "message": "RAG-LLM-Metric API service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/test-evaluation")
async def test_evaluation_direct(request: EvaluationRequest):
    """Direct test without background tasks"""
    try:
        print("DEBUG: Starting direct evaluation test")
        result = await evaluation_service.start_evaluation(request)
        print(f"DEBUG: Direct evaluation completed, result type: {type(result)}")
        return {"status": "success", "result_type": str(type(result))}
    except Exception as e:
        print(f"DEBUG: Direct evaluation failed: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}

@app.get("/evaluators", response_model=List[AvailableEvaluator])
async def get_available_evaluators():
    try:
        return evaluation_service.get_available_evaluators()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get evaluators failed: {str(e)}")

@app.post("/agent-evaluate")
async def start_agent_evaluation(request: dict, background_tasks: BackgroundTasks):
    """Start Agent-based evaluation with dynamic metric selection"""
    try:
        print(f"DEBUG: Received agent evaluation request")
        
    
        required_fields = ['dataset', 'user_criteria']
        for field in required_fields:
            if field not in request:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
    
        task_id = str(uuid.uuid4())
        print(f"DEBUG: Generated agent task ID: {task_id}")
        
    
        background_tasks.add_task(run_agent_evaluation_task, task_id, request)
        
        return {
            "task_id": task_id,
            "message": "Agent-based evaluation task started",
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Error in start_agent_evaluation: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"start agent evaluation failed: {str(e)}")

@app.get("/agent-evaluate/{task_id}/progress")
async def get_agent_evaluation_progress(task_id: str):
    """Get Agent evaluation progress"""
    try:
        if task_id not in agent_evaluation_tasks:
            raise HTTPException(status_code=404, detail="task not found")
        
        task_info = agent_evaluation_tasks[task_id]
        return {
            "task_id": task_id,
            "status": task_info["status"],
            "progress": task_info.get("progress"),
            "metrics_discussion": task_info.get("metrics_discussion"),
            "selected_metrics": task_info.get("selected_metrics"),
            "error": task_info.get("error")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent progress: {str(e)}")

@app.get("/agent-evaluate/{task_id}/result")
async def get_agent_evaluation_result(task_id: str):
    """Get Agent evaluation result"""
    try:
        if task_id not in agent_evaluation_tasks:
            raise HTTPException(status_code=404, detail="task not found")
        
        task_info = agent_evaluation_tasks[task_id]
        
        if task_info["status"] == "error":
            # Return error information but still include any available metrics
            return {
                "status": "error",
                "error": task_info.get("error", "Unknown error"),
                "error_timestamp": task_info.get("failed_at"),
                "selected_metrics": task_info.get("selected_metrics", {}),
                "discussion_summary": task_info.get("discussion_summary", ""),
                "chat_history": task_info.get("chat_history", []),
                "evaluation_result": task_info.get("evaluation_result", None)
            }
        elif task_info["status"] != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"task not completed, current status: {task_info['status']}"
            )
        
        # Return successful result with complete evaluation data
        result = task_info["result"]
        return {
            "status": "success",
            "selected_metrics": result.get("selected_metrics", {}),
            "discussion_summary": result.get("discussion_summary", ""),
            "chat_history": result.get("chat_history", []),
            "evaluation_result": result.get("evaluation_result", None),
            "timestamp": task_info.get("completed_at")
        }
        
    except HTTPException:
        raise
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
        
        return {
            "status": "error", 
            "error": f"Failed to retrieve agent results: {str(e)}",
            "selected_metrics": fallback_metrics,
            "discussion_summary": "Failed to retrieve agent results due to connection error. Using default metrics.",
            "chat_history": [],
            "evaluation_result": None
        }

@app.post("/evaluate/update-weights")
async def update_evaluation_weights(request: dict):
    """Update metric weights and recalculate scores"""
    try:
        dataset = request.get('dataset')
        metric_weights = request.get('metric_weights')
        evaluator_results = request.get('evaluator_results')  # Previous evaluation raw results
        
        if not all([dataset, metric_weights, evaluator_results]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        
        updated_result = evaluation_service.recalculate_weighted_scores(
            evaluator_results, metric_weights
        )
        
        return {
            "updated_scores": updated_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update weights: {str(e)}")

@app.post("/evaluate")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    try:
        print(f"DEBUG: Received evaluation request with {len(request.dataset)} samples")
        
        # validate dataset
        print("DEBUG: Validating dataset")
        if not evaluation_service.validate_dataset(request.dataset):
            print("DEBUG: Dataset validation failed")
            raise HTTPException(
                status_code=400, 
                detail="dataset format is not correct, must contain Question, Reference_Answer, Model_Answer fields"
            )
        print("DEBUG: Dataset validation passed")
        
        # generate task id
        task_id = str(uuid.uuid4())
        print(f"DEBUG: Generated task ID: {task_id}")
        
        # start evaluation task in background
        print("DEBUG: Adding background task")
        background_tasks.add_task(run_evaluation_task, task_id, request)
        print(f"DEBUG: Background task added for {task_id}")
        
        return {
            "task_id": task_id,
            "message": "evaluation task started",
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Error in start_evaluation: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"start evaluation failed: {str(e)}")

@app.get("/evaluate/{task_id}/progress")
async def get_evaluation_progress(task_id: str):
    try:
        print(f"DEBUG: Getting progress for task {task_id}")
        print(f"DEBUG: Available tasks: {list(evaluation_tasks.keys())}")
        
        if task_id not in evaluation_tasks:
            print(f"DEBUG: Task {task_id} not found")
            raise HTTPException(status_code=404, detail="task not found")
        
        task_info = evaluation_tasks[task_id]
        print(f"DEBUG: Task info: {task_info}")
        
        # Get live progress from evaluation service if task is running
        if task_info["status"] == "running":
            print("DEBUG: Getting live progress from evaluation service")
            try:
                progress = evaluation_service.get_progress()
                print(f"DEBUG: Progress object: {progress}")
                if progress:
                    print(f"DEBUG: Progress attributes: status={progress.status}, error={getattr(progress, 'error', 'NO_ERROR_ATTR')}")
                    return {
                        "task_id": task_id,
                        "status": progress.status,
                        "progress": {
                            "progress_percentage": progress.progress_percentage,
                            "current_evaluator": progress.current_evaluator,
                            "processed_samples": progress.processed_samples,
                            "total_samples": progress.total_samples
                        },
                        "error": getattr(progress, 'error', None)
                    }
                else:
                    print("DEBUG: No progress object available")
            except Exception as progress_error:
                print(f"DEBUG: Error getting progress: {progress_error}")
                import traceback
                print(f"DEBUG: Progress error traceback: {traceback.format_exc()}")
                raise progress_error
        
        print("DEBUG: Returning static task info")
        return {
            "task_id": task_id,
            "status": task_info["status"],
            "progress": task_info.get("progress"),
            "result": task_info.get("result"),
            "error": task_info.get("error")
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: Unexpected error in get_evaluation_progress: {e}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to get progress: {str(e)}")

@app.get("/evaluate/{task_id}/result")
async def get_evaluation_result(task_id: str):
    if task_id not in evaluation_tasks:
        raise HTTPException(status_code=404, detail="task not found")
    
    task_info = evaluation_tasks[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"task not completed, current status: {task_info['status']}"
        )
    
    return task_info["result"]

@app.post("/validate-dataset")
async def validate_dataset(dataset: List[dict]):
    try:
        is_valid = evaluation_service.validate_dataset(dataset)
        
        if is_valid:
            return {
                "valid": True,
                "message": "dataset format is correct",
                "samples": len(dataset)
            }
        else:
            return {
                "valid": False,
                "message": "dataset format is not correct, must contain Question, Reference_Answer, Model_Answer fields",
                "samples": len(dataset) if dataset else 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/annotation")
async def start_data_annotation(request: DataAugmentationRequest):
    try:
        if not evaluation_service.validate_hf_dataset(request.dataset):
            raise HTTPException(
                status_code=400, 
                detail="Invalid dataset format, must contain question, response, documents fields"
            )
        
        # Process using original ExecutionPipeline with annotators
        result = await evaluation_service.run_annotation_pipeline(request)
        
        return {
            "annotated_dataset": result.augmented_dataset,
            "annotation_summary": result.mistake_summary,
            "processing_time": result.processing_time,
            "timestamp": result.timestamp,
            "status": "completed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process annotation: {str(e)}")

@app.post("/validate-hf-dataset")
async def validate_hf_dataset(dataset: List[dict]):
    try:
        is_valid = evaluation_service.validate_hf_dataset(dataset)
        
        if is_valid:
            return {
                "valid": True,
                "message": "Hugging Face dataset format is valid",
                "samples": len(dataset)
            }
        else:
            return {
                "valid": False,
                "message": "Invalid dataset format, must contain question, response, documents fields",
                "samples": len(dataset) if dataset else 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/evaluation/save")
async def save_evaluation_result(request: SaveEvaluationRequest):
    try:
        evaluation_id = evaluation_service.save_evaluation_result(request)
        
        return {
            "success": True,
            "evaluation_id": evaluation_id,
            "message": "Evaluation result saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save evaluation: {str(e)}")

@app.get("/evaluation/history")
async def get_evaluation_history():
    try:
        history = evaluation_service.get_evaluation_history()
        return {
            "history": [entry.dict() for entry in history],
            "total": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get("/evaluation/history/{evaluation_id}")
async def get_evaluation_by_id(evaluation_id: str):
    try:
        evaluation = evaluation_service.get_evaluation_by_id(evaluation_id)
        
        if evaluation is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        return evaluation.dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation: {str(e)}")

@app.delete("/evaluation/history/{evaluation_id}")
async def delete_evaluation(evaluation_id: str):
    try:
        success = evaluation_service.delete_evaluation(evaluation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        return {"success": True, "message": "Evaluation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete evaluation: {str(e)}")

@app.put("/evaluation/history/{evaluation_id}/notes")
async def update_evaluation_notes(evaluation_id: str, notes: dict):
    try:
        success = evaluation_service.update_evaluation_notes(evaluation_id, notes.get("notes", ""))
        
        if not success:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        
        return {"success": True, "message": "Notes updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update notes: {str(e)}")

@app.post("/evaluation/compare")
async def compare_evaluations(evaluation_ids: List[str]):
    try:
        evaluations = []
        for eval_id in evaluation_ids:
            evaluation = evaluation_service.get_evaluation_by_id(eval_id)
            if evaluation:
                evaluations.append(evaluation.dict())
        
        if len(evaluations) < 2:
            raise HTTPException(status_code=400, detail="At least 2 evaluations required for comparison")
        
        return {
            "evaluations": evaluations,
            "comparison_timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare evaluations: {str(e)}")

@app.post("/evaluate-pipeline")
async def run_evaluation_pipeline(request: dict):
    """Run evaluation pipeline with custom metrics configuration"""
    try:
        dataset = request.get("dataset", [])
        llm_provider = request.get("llm_provider", "openai")
        model_name = request.get("model_name", "gpt-4o-mini")
        metrics_config = request.get("metrics_config", [])
        
        # Convert metrics config to evaluation request format
        selected_evaluators = []
        metric_weights = {}
        
        for metric_config in metrics_config:
            evaluator_name = metric_config.get("evaluator_name")
            weight = metric_config.get("weight", 0.0)
            
            if evaluator_name and weight > 0:
                selected_evaluators.append(evaluator_name)
                metric_weights[evaluator_name] = weight
        
        # Create evaluation request
        evaluation_request = EvaluationRequest(
            dataset=dataset,
            selected_evaluators=selected_evaluators,
            metric_weights=metric_weights,
            llm_provider=llm_provider,
            model_name=model_name
        )
        
        # Run evaluation
        result = await evaluation_service.start_evaluation(evaluation_request)
        
        # Return structured result
        return {
            "status": "success",
            "final_score": result.final_score,
            "metrics": [metric.dict() for metric in result.metrics],
            "processed_dataset": result.processed_dataset,
            "evaluation_summary": result.evaluation_summary,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Pipeline evaluation failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

async def run_agent_evaluation_task(task_id: str, request: dict):
    """Run Agent-based evaluation task"""
    try:
        print(f"DEBUG: Starting agent evaluation task {task_id}")
        agent_evaluation_tasks[task_id] = {
            "status": "running",
            "progress": {"stage": "initializing", "percentage": 0},
            "metrics_discussion": None,
            "selected_metrics": None,
            "result": None,
            "error": None,
            "started_at": datetime.now().isoformat()
        }
        
        # Update progress
        agent_evaluation_tasks[task_id]["progress"] = {"stage": "agent_negotiation", "percentage": 25}
        
        # Run Agent evaluation (both metrics selection and evaluation)
        result = await evaluation_service.start_agent_evaluation(request)
        
        # Extract information for task tracking
        selected_metrics = result.get("selected_metrics", {})
        discussion_summary = result.get("discussion_summary", "")
        
        # Update task status
        if result.get("status") == "success":
            agent_evaluation_tasks[task_id].update({
                "status": "completed",
                "result": result,
                "metrics_discussion": discussion_summary,
                "selected_metrics": selected_metrics,
                "completed_at": datetime.now().isoformat()
            })
        else:
            # Handle error case but still store available information
            agent_evaluation_tasks[task_id].update({
                "status": "error",
                "error": result.get("error"),
                "selected_metrics": selected_metrics,
                "discussion_summary": discussion_summary,
                "chat_history": result.get("chat_history", []),
                "evaluation_result": result.get("evaluation_result"),
                "failed_at": datetime.now().isoformat()
            })
        
    except Exception as e:
        print(f"DEBUG: Agent evaluation task {task_id} failed: {e}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        agent_evaluation_tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

async def run_evaluation_task(task_id: str, request: EvaluationRequest):
    try:
        print(f"DEBUG: Starting evaluation task {task_id}")
        evaluation_tasks[task_id] = {
            "status": "running",
            "progress": None,
            "result": None,
            "error": None,
            "started_at": datetime.now().isoformat()
        }
        print(f"DEBUG: Task {task_id} status set to running")
        
        # execute evaluation
        print(f"DEBUG: About to call evaluation_service.start_evaluation for task {task_id}")
        result = await evaluation_service.start_evaluation(request)
        print(f"DEBUG: Evaluation completed for task {task_id}, result type: {type(result)}")
        
        # Convert EvaluationResult to dict format for API response
        result_dict = {
            "status": "completed",
            "dataset": result.processed_dataset,
            "metrics": [metric.dict() for metric in result.metrics],
            "final_score": result.final_score,
            "evaluation_summary": result.evaluation_summary,
            "timestamp": result.timestamp
        }
        
        # update task status
        evaluation_tasks[task_id].update({
            "status": "completed",
            "result": result_dict,
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        # record error
        evaluation_tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 