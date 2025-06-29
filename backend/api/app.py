from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import json
import uuid
import asyncio
from datetime import datetime

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

@app.get("/")
async def root():
    return {
        "message": "RAG-LLM-Metric API service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/evaluators", response_model=List[AvailableEvaluator])
async def get_available_evaluators():
    try:
        return evaluation_service.get_available_evaluators()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get evaluators failed: {str(e)}")

@app.post("/evaluate")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    try:
        # validate dataset
        if not evaluation_service.validate_dataset(request.dataset):
            raise HTTPException(
                status_code=400, 
                detail="dataset format is not correct, must contain Question, Reference_Answer, Model_Answer fields"
            )
        
        # generate task id
        task_id = str(uuid.uuid4())
        
        # start evaluation task in background
        background_tasks.add_task(run_evaluation_task, task_id, request)
        
        return {
            "task_id": task_id,
            "message": "evaluation task started",
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"start evaluation failed: {str(e)}")

@app.get("/evaluate/{task_id}/progress")
async def get_evaluation_progress(task_id: str):
    if task_id not in evaluation_tasks:
        raise HTTPException(status_code=404, detail="task not found")
    
    task_info = evaluation_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info.get("progress"),
        "result": task_info.get("result"),
        "error": task_info.get("error")
    }

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
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

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
            "annotated_dataset": result.annotated_dataset,
            "annotation_summary": result.annotation_summary,
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

async def run_evaluation_task(task_id: str, request: EvaluationRequest):
    try:
        evaluation_tasks[task_id] = {
            "status": "running",
            "progress": None,
            "result": None,
            "error": None,
            "started_at": datetime.now().isoformat()
        }
        
        # execute evaluation
        result = await evaluation_service.start_evaluation(request)
        
        # update task status
        evaluation_tasks[task_id].update({
            "status": "completed",
            "result": result,
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