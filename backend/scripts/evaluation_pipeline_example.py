import asyncio
import sys

from utils.llm import OpenAIClientLLM

sys.path.append("..")

from execution_pipeline.execution_pipeline import ExecutionPipeline

from dotenv import load_dotenv

load_dotenv()


import logging
import os
from datetime import datetime

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
# Generate log filename with current timestamp
log_filename = os.path.join(
    log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
)
# Configure logging to write to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler(),  # Log to console
    ],
)
logger = logging.getLogger(__name__)

DATASET_NAME = "RAGEVALUATION-HJKMY/TSBC_cleaned_demo"

from evaluator.evaluators import *


async def main():
    logger.info("Start processing pipeline")
    pipeline = ExecutionPipeline([FactualAccuracyEvaluator])
    return await pipeline.run_pipeline(
        dataset_name=DATASET_NAME,
        save_path="./tmp_data",
        upload_to_hub=True,
        repo_id="RAGEVALUATION-HJKMY/test-jinw-TSBC_cleaned_demo_experiment",
        llm_class=OpenAIClientLLM,
        model="o1-2024-12-17",
        base_url="https://api.openai.com/v1/",
    )


if __name__ == "__main__":
    print(asyncio.run(main())["train"].to_pandas())