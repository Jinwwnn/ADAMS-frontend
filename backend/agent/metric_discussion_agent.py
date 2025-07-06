import importlib
import json
import random
from datetime import datetime
import re
import os
import asyncio
import inspect
import pandas as pd
from typing import Dict, List, Optional, Tuple
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Disable Docker for autogen
os.environ["AUTOGEN_USE_DOCKER"] = "0"

from autogen.agentchat import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from evaluator.base_evaluator import RAGEvaluator
from execution_pipeline.execution_pipeline import CompoundScoreExecutionPipeline
from utils.llm import LLMClient, OpenAIClientLLM


def get_evaluator_classes():
    """Retrieve all implemented evaluators derived from RAGEvaluator."""
    module = importlib.import_module("evaluator.evaluators")
    evaluator_classes = []

    for _, cls in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(cls, RAGEvaluator)
            and cls.__module__ == module.__name__
            and cls.__name__.endswith("Evaluator")
            and cls is not RAGEvaluator
        ):
            evaluator_classes.append(cls)

    return evaluator_classes


def make_valid_identifier(input_str):
    cleaned_str = re.sub(r"[^a-zA-Z0-9_]", "", input_str)
    if cleaned_str and cleaned_str[0].isdigit():
        cleaned_str = "_" + cleaned_str
    return cleaned_str if cleaned_str else "identifier"


class DynamicEvaluationOrchestrator:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_df: Optional[pd.DataFrame] = None,
        evaluate_llm_class: type[LLMClient] = OpenAIClientLLM,
        evaluate_llm_model: str = "gpt-4o-mini",
        evaluate_llm_base_url: str = "https://api.openai.com/v1",
        agent_llm_model: str = "gpt-4o-mini",
        upload_to_hub: bool = False,
        repo_name: Optional[str] = None,
        max_discussion_round: Optional[int] = 10,
    ):
        if dataset_name is None:
            if upload_to_hub and repo_name is None:
                raise ValueError(
                    "must offer repo name when uploading result from pandas df to HF"
                )
            self.dataset = dataset_df
        elif dataset_df is None:
            self.dataset = dataset_name
        else:
            raise ValueError("must offer dataset by name to HF or a pandas dataframe")
            
        self.evaluate_llm_class = evaluate_llm_class
        self.evaluate_llm_model = evaluate_llm_model
        self.evaluate_llm_base_url = evaluate_llm_base_url
        self.agent_llm_model = agent_llm_model
        self.upload_to_hub = upload_to_hub
        self.max_discussion_round = max_discussion_round
        
        if not repo_name:
            self.repo_name = f"{dataset_name}-Evaluated-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{self.evaluate_llm_model}"
        else:
            self.repo_name = repo_name

        # Create LLM config for agents (new autogen format)
        self.llm_config = {
            "config_list": [
                {
                    "model": self.agent_llm_model,
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": "https://api.openai.com/v1"
                }
            ],
            "temperature": 0.7
        }
        
        self.metric_info = self._get_metrics_metadata()

    def get_sample_data(self):
        """Get sample data from dataset for agent analysis."""
        if isinstance(self.dataset, str):
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "Hugging Face datasets library required: pip install datasets"
                )
            try:
                hf_dataset = load_dataset(self.dataset, split="train")
                dataset_size = len(hf_dataset)
                n_samples = min(2, dataset_size)

                if dataset_size == 0:
                    return []

                indices = random.sample(range(dataset_size), n_samples)
                return json.dumps(
                    [
                        {
                            "question": hf_dataset[i]["question"],
                            "context": hf_dataset[i]["documents"],
                            "golden_answer": hf_dataset[i]["response"],
                        }
                        for i in indices
                    ]
                )

            except Exception as e:
                raise ValueError(f"Failed to load dataset: {str(e)}")

        elif isinstance(self.dataset, pd.DataFrame):
            n_samples = min(2, len(self.dataset))
            return json.dumps(
                self.dataset.sample(n=n_samples)
                .rename(columns={"documents": "context", "response": "golden_answer"})[
                    ["question", "context", "golden_answer"]
                ]
                .to_dict(orient="records")
            )

        else:
            raise TypeError("Input must be HF dataset name (str) or pandas DataFrame")

    def _get_metrics_metadata(self) -> List[Dict]:
        """Get metadata about available evaluators."""
        evaluators = get_evaluator_classes()
        return [evaluator_class.description() for evaluator_class in evaluators]

    def _create_agents(self, user_criteria: str) -> Tuple[List[AssistantAgent], UserProxyAgent]:
        """Create specialized agents for metric selection discussion."""
        
        # Create sample data for agent context
        sample_data = self.get_sample_data()
        
        # Quality Guardian Agent
        quality_guardian = AssistantAgent(
            name="QualityGuardian",
            system_message=f"""You are a Quality Assessment Expert. Your role is to analyze evaluation metrics for RAG systems.

Available metrics: {json.dumps(self.metric_info, indent=2)}

User criteria: {user_criteria}

Sample data from dataset: {sample_data}

Focus on:
1. Factual accuracy and correctness
2. Relevance to the query
3. Comprehensive coverage
4. Robustness to errors

Select 3-5 most appropriate metrics and suggest weights (0.0-1.0) based on user criteria.
Provide clear reasoning for your choices.""",
            llm_config=self.llm_config,
        )

        # User Experience Agent
        user_advocate = AssistantAgent(
            name="UserAdvocate",
            system_message=f"""You are a User Experience Expert focusing on practical evaluation needs.

Available metrics: {json.dumps(self.metric_info, indent=2)}

User criteria: {user_criteria}

Sample data from dataset: {sample_data}

Focus on:
1. Readability and clarity
2. User satisfaction
3. Practical applicability
4. Coherence and engagement

Select 3-5 metrics that best serve end-user needs and suggest weights (0.0-1.0).
Consider what users actually care about when reading responses.""",
            llm_config=self.llm_config,
        )

        # Technical Expert Agent
        technical_expert = AssistantAgent(
            name="TechnicalExpert",
            system_message=f"""You are a Technical RAG Systems Expert with deep knowledge of evaluation methodologies.

Available metrics: {json.dumps(self.metric_info, indent=2)}

User criteria: {user_criteria}

Sample data from dataset: {sample_data}

Focus on:
1. Context utilization and relevance
2. Technical accuracy
3. Citation quality
4. Information completeness

Select 3-5 metrics that provide comprehensive technical evaluation and suggest weights (0.0-1.0).
Ensure balanced coverage across all technical aspects.""",
            llm_config=self.llm_config,
        )

        # User Proxy for orchestration
        user_proxy = UserProxyAgent(
            name="ModerationProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            system_message="""You orchestrate the discussion between agents to reach consensus on metric selection.
Ask each agent to provide their recommendations, then synthesize a final decision.
End your message with 'TERMINATE' when consensus is reached.""",
            llm_config=self.llm_config,
        )

        return [quality_guardian, user_advocate, technical_expert], user_proxy

    async def negotiate_metrics(self, user_criteria: str) -> Dict:
        """Run agent negotiation to select optimal metrics."""
        try:
            agents, user_proxy = self._create_agents(user_criteria)
            
            # Create group chat
            groupchat = GroupChat(
                agents=agents + [user_proxy],
                messages=[],
                max_round=self.max_discussion_round,
                speaker_selection_method="round_robin",
            )
            
            manager = GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)
            
            # Start discussion
            initial_message = f"""
Let's discuss the optimal evaluation metrics for this RAG system evaluation task.

        User Requirements: {user_criteria}

Each agent should:
1. Analyze the user criteria and sample data
2. Recommend 3-5 most suitable metrics from the available options
3. Suggest weights (0.0-1.0) for each recommended metric
4. Provide clear reasoning

Available metrics: {', '.join([m['name'] for m in self.metric_info])}

QualityGuardian, please start by sharing your recommendations focusing on accuracy and robustness.
"""
            
            # Run the conversation
            chat_result = user_proxy.initiate_chat(
                manager,
                message=initial_message,
                clear_history=True,
            )
            
            # Extract final decision from chat history
            final_metrics = self._extract_final_metrics(chat_result.chat_history)

            return {
                "status": "success",
                "selected_metrics": final_metrics,
                "discussion_summary": self._summarize_discussion(chat_result.chat_history),
                "chat_history": [{"sender": msg.get("name", "Unknown"), "content": msg.get("content", "")} 
                               for msg in chat_result.chat_history if isinstance(msg, dict)]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Agent negotiation failed: {str(e)}",
                "selected_metrics": self._get_default_metrics(),
                "discussion_summary": "Failed to conduct agent discussion, using default metrics.",
                "chat_history": []
            }

    def _extract_final_metrics(self, chat_history: List[Dict]) -> Dict[str, float]:
        """Extract final metric selection from chat conversation."""
        try:
            # Get the last few messages to find consensus
            final_content = ""
            for msg in reversed(chat_history[-5:]):  # Check last 5 messages
                if isinstance(msg, dict):
                    final_content += msg.get("content", "") + "\n"
            
            # Try to extract metric names and weights using regex
            metrics = {}
            available_metric_names = [m['name'] for m in self.metric_info]
            
            for metric_name in available_metric_names:
                # Look for patterns like "MetricName: 0.8" or "MetricName (0.8)"
                patterns = [
                    rf"{re.escape(metric_name)}[\s:]+([0-9]*\.?[0-9]+)",
                    rf"{re.escape(metric_name)}\s*\(([0-9]*\.?[0-9]+)\)",
                    rf"({metric_name})",  # Just presence without weight
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, final_content, re.IGNORECASE)
                    if matches:
                        try:
                            weight = float(matches[0]) if matches[0].replace('.', '').isdigit() else 0.8
                            metrics[metric_name] = min(1.0, max(0.0, weight))  # Clamp between 0 and 1
                            break
                        except ValueError:
                            metrics[metric_name] = 0.8  # Default weight
            
            # If no metrics found, use default selection
            if not metrics:
                return self._get_default_metrics()
                
            return metrics
            
        except Exception:
            return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metric selection."""
        return {
            "Factual Accuracy": 0.9,
            "Context Relevance": 0.8,
            "Coherence": 0.7,
            "Answer Equivalence": 0.8,
            "Factual Correctness": 0.85
        }

    def _summarize_discussion(self, chat_history: List[Dict]) -> str:
        """Summarize the agent discussion."""
        try:
            agent_contributions = {}
            for msg in chat_history:
                if isinstance(msg, dict):
                    sender = msg.get("name", "Unknown")
                    content = msg.get("content", "")
                    if sender not in agent_contributions:
                        agent_contributions[sender] = []
                    agent_contributions[sender].append(content[:200] + "..." if len(content) > 200 else content)
            
            summary = "Agent Discussion Summary:\n\n"
            for agent, messages in agent_contributions.items():
                if agent != "ModerationProxy":
                    summary += f"**{agent}**: {messages[0] if messages else 'No contribution'}\n\n"
            
            return summary
        except Exception:
            return "Discussion summary unavailable due to processing error."

    async def evaluate(self, user_criteria: str):
        """Run the complete evaluation pipeline with agent-selected metrics."""
        try:
            # Get metrics from agent negotiation
            negotiation_result = await self.negotiate_metrics(user_criteria)
            selected_metrics = negotiation_result.get("selected_metrics", {})
            
            # Map metric names to evaluator classes
            evaluator_classes = get_evaluator_classes()
            evaluators_with_weights = []
            
            for evaluator_class in evaluator_classes:
                metric_info = evaluator_class.description()
                metric_name = metric_info.get("name", "")
                
                if metric_name in selected_metrics:
                    weight = selected_metrics[metric_name]
                    evaluators_with_weights.append((evaluator_class, weight))
            
            # If no matching evaluators found, use defaults
            if not evaluators_with_weights:
                default_metrics = self._get_default_metrics()
                for evaluator_class in evaluator_classes[:5]:  # Take first 5 as defaults
                    metric_info = evaluator_class.description()
                    metric_name = metric_info.get("name", "")
                    weight = default_metrics.get(metric_name, 0.7)
                    evaluators_with_weights.append((evaluator_class, weight))
            
            # Create and run pipeline
            pipeline = CompoundScoreExecutionPipeline(evaluators_with_weights)
            
            result = await pipeline.run_pipeline_with_weight(
                dataset_name=self.dataset if isinstance(self.dataset, str) else None,
                dataset_df=self.dataset if isinstance(self.dataset, pd.DataFrame) else None,
                llm_class=self.evaluate_llm_class,
                model=self.evaluate_llm_model,
                base_url=self.evaluate_llm_base_url,
                upload_to_hub=self.upload_to_hub,
                repo_id=self.repo_name if self.upload_to_hub else None,
            )
            
            return {
                "status": "success",
                "result": result,
                "negotiation_result": negotiation_result,
                "selected_evaluators": [
                    {"name": cls.__name__, "weight": weight} 
                    for cls, weight in evaluators_with_weights
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "negotiation_result": {"status": "error", "error": str(e)}
            }


async def main():
    """Example usage of the DynamicEvaluationOrchestrator."""
    import dotenv
    dotenv.load_dotenv()
    
    # Example with pandas DataFrame
    sample_data = pd.DataFrame([
        {
            "question": "What is the capital of France?",
            "response": "Paris is the capital of France.",
            "documents": "France is a country in Europe. Paris is its capital city."
        }
    ])
    
    orchestrator = DynamicEvaluationOrchestrator(
        dataset_df=sample_data,
        agent_llm_model="gpt-4o-mini",
        upload_to_hub=False
    )
    
    user_criteria = "I want to evaluate a customer service chatbot. Focus on accuracy and helpfulness."
    
    result = await orchestrator.negotiate_metrics(user_criteria)
    print("Negotiation Result:", json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
