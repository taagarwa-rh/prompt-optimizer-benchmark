import argparse
import time
from typing import Any, Literal, Callable, Optional
import logging

import mlflow
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic_yaml import parse_yaml_file_as
from pathlib import Path
from pydantic import BaseModel, SecretStr
from datasets import load_dataset

from prompt_optimizer import PredictionError, Prompt
from prompt_optimizer.optimizers import APEOptimizer, BaseOptimizer, OPROOptimizer, PromptAgentOptimizer, ProtegiOptimizer

load_dotenv()

logger = logging.getLogger(__name__)

class RunMetadata(BaseModel):
    
    name: str = "prompt-optimization"
    experiment_name: str = "Default"

class ClientConfig(BaseModel):
    
    model: str
    api_key: SecretStr
    kwargs: dict = {}

class ClientsConfig(BaseModel):
    
    evaluator: ClientConfig
    optimizer: ClientConfig
    
    
class DatasetConfig(BaseModel):
    
    path: str
    name: str
    split: Optional[str] = None
    max_size: int = -1
    test_split: Optional[str] = None
    train_pct: float = 0.8
    
    
class OptimizerConfig(BaseModel):
    
    name: str
    optimizer: Literal["APEOptimizer", "OPROOptimizer", "PromptAgentOptimizer", "ProtegiOptimizer"]
    kwargs: dict = {}
    optimizer_cls: Callable[[Any], BaseOptimizer] = None
    
    def model_post_init(self, context):
        cls_map = {
            "APEOptimizer": APEOptimizer,
            "OPROOptimizer": OPROOptimizer,
            "PromptAgentOptimizer": PromptAgentOptimizer,
            "ProtegiOptimizer": ProtegiOptimizer,
        }
        self.optimizer_cls = cls_map[self.optimizer]
        
        if "seed_prompts" in self.kwargs and isinstance(self.kwargs.get("seed_prompts", None), list):
            self.kwargs["seed_prompts"] = [Prompt(content=content) for content in self.kwargs.get("seed_prompts", [])]
    

class BenchmarkConfig(BaseModel):
    meta: RunMetadata
    clients: ClientsConfig
    dataset: DatasetConfig
    optimizers: list[OptimizerConfig]


def parse_args() -> BenchmarkConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Parse config into pydantic model
    config = parse_yaml_file_as(BenchmarkConfig, args.config)
    
    return config


def get_client(config: ClientConfig):
    """Get the LLM client for the AI system."""
    client = ChatOpenAI(model=config.model, api_key=config.api_key, **config.kwargs)
    return client


def get_dataset(path: str, name: str = None, max_size: int = -1, split: str = None, test_split: str = None, train_pct: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Get validation and test data."""
    train_ds = load_dataset(path, name, split=split, trust_remote_code=True)
    if split is None:
        splits = list(train_ds.keys())
        train_ds = train_ds[splits[0]]
    
    # Split the dataset into training and testing sets
    if test_split:
        test_ds = load_dataset(path, name, split=test_split)
    elif train_pct is not None:
        if max_size > 0:
            train_size = int(max_size * train_pct)
            test_size = max_size - train_size
        else:
            train_size = train_pct
            test_size = None
        split_ds = train_ds.train_test_split(train_size=train_size, test_size=test_size, shuffle=False)
        train_ds, test_ds = split_ds["train"], split_ds["test"]
    else:
        raise ValueError("One of test_split or train_pct must be specified.")
    
    # Reduce the sizes to the specified size
    if max_size > 0:
        train_ds = train_ds.select(range(0, min(max_size, len(train_ds))))
        test_ds = test_ds.select(range(0, min(max_size, len(test_ds))))
        
    return train_ds.to_pandas().to_dict(orient="records"), test_ds.to_pandas().to_dict(orient="records")
    


class Evaluator:
    def __init__(self, config: ClientConfig):
        """Initialize."""
        self.client = get_client(config)

    def _evaluate(self, prompt: Prompt, validation_set: list[dict]) -> list[str]:
        """Prompt evaluator function."""
        # Run the prompt through the AI system
        predictions = []
        num_correct = 0
        for row in validation_set:
            # Get the prediction
            input = row["input"]
            messages = [{"role": "system", "content": prompt.content}, {"role": "user", "content": input}]
            response = self.client.invoke(messages)
            prediction = response.content.strip()
            predictions.append(prediction)

            # Reward exact matches and collect errors
            actual = row["target"]
            if actual == prediction:
                num_correct += 1
            else:
                num_correct += 0
                error = PredictionError(input=input, actual=actual, prediction=prediction)
                prompt.errors.append(error)

        # Compute the score
        score = num_correct / len(validation_set)

        # Save the predictions and score
        prompt.metadata["predictions"] = predictions

        return score

    def __call__(self, prompt: Prompt, validation_set: list[dict]):
        return self._evaluate(prompt=prompt, validation_set=validation_set)


def main():
    """Run main process."""
    # Load Configuration
    config = parse_args()

    # Fetch dataset
    dataset_cfg = config.dataset
    train_ds, test_ds = get_dataset(
        path=dataset_cfg.path,
        name=dataset_cfg.name,
        split=dataset_cfg.split,
        max_size=dataset_cfg.max_size, 
        train_pct=dataset_cfg.train_pct,
    )
    logger.info(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")

    # Configure LLM Clients
    evaluator_cfg = config.clients.evaluator
    optimizer_cfg = config.clients.optimizer
    logger.info(f"Evaluator Config: {evaluator_cfg.model_dump()}")
    logger.info(f"Optimizer Config: {optimizer_cfg.model_dump()}")
    
    # Get evaluator
    evaluator = Evaluator(evaluator_cfg)
    
    # Get optimizer client
    optimizer_client = get_client(optimizer_cfg)
    
    # Run optimizations
    optimizer_cfgs = config.optimizers
    mlflow.set_experiment(config.meta.experiment_name)
    logger.info(f"MLflow Experiment Set: {config.meta.experiment_name}")
    with mlflow.start_run(run_name=config.meta.name):
        for cfg in optimizer_cfgs:
            # Create optimizer
            optimizer_kwargs: dict[str, Any] = cfg.kwargs
            optimizer: BaseOptimizer = cfg.optimizer_cls(client=optimizer_client, evaluator=evaluator, validation_set=train_ds, **optimizer_kwargs)
            
            logger.info(f"Starting Run: {cfg.name}")
            with mlflow.start_run(nested=True, run_name=cfg.name):
                # Log input params
                params = {
                    "optimizer": optimizer.__class__.__name__,
                    "dataset": dataset_cfg.path,
                    "dataset_name": dataset_cfg.name,
                }
                params = {
                    **params, 
                    **{f"optimizer_kwargs_{key}": value for key, value in optimizer_kwargs.items()} 
                    **{f"evaluator_{key}": value for key, value in evaluator_cfg.model_dump().items()},
                    **{f"optimizer_{key}": value for key, value in optimizer_cfg.model_dump().items()},
                }
                mlflow.log_params(params=params)

                # Run optimization
                start = time.time()
                best_prompt = optimizer.run()
                end = time.time()

                # Log the results
                all_prompts = optimizer.get_all_prompts(include_candidates=True)
                test_score = evaluator(prompt=best_prompt, validation_set=test_ds)
                metrics = {
                    "best_prompt_train_score": best_prompt.score,
                    "best_prompt_test_score": test_score,
                    "prompts_generated": len(sum(all_prompts, start=[])),
                    "time_sec": end - start,
                }
                if len(optimizer.seed_prompts) > 0:
                    best_seed_prompt = max(optimizer.seed_prompts, key=lambda x: x.score)
                    baseline_train_score = best_seed_prompt.score
                    baseline_test_score = evaluator(prompt=best_seed_prompt, validation_set=test_ds)
                    metrics = {**metrics, "baseline_train_score": baseline_train_score, "baseline_test_score": baseline_test_score}
                
                mlflow.log_metrics(metrics=metrics)

                log_dict = {
                    "best_prompt": best_prompt.model_dump(),
                    "all_prompts": [[p.model_dump() for p in step] for step in all_prompts],
                }
                mlflow.log_dict(log_dict, artifact_file="all_prompts.json")

                max_score_each_step = [max(step, key=lambda x: x.score).score if len(step) != 0 else 0.0 for step in all_prompts]
                avg_score_each_step = [sum(p.score for p in step) / len(step) if len(step) != 0 else 0.0 for step in all_prompts]
                min_score_each_step = [min(step, key=lambda x: x.score).score if len(step) != 0 else 0.0 for step in all_prompts]
                table = {
                    "step": [i for i in range(len(max_score_each_step))],
                    "max_score": max_score_each_step,
                    "avg_score": avg_score_each_step,
                    "min_score": min_score_each_step,
                }
                mlflow.log_table(table, artifact_file="table.json")
                logger.info(f"Run Complete: {cfg.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting benchmarking...")
    main()
    logger.info("Benchmarking completed successfully.")
