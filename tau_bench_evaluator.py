import tau_bench.envs
import tau_bench.run
from prompt_optimizer import PredictionError, Prompt
from tau_bench.envs.retail.tasks_test import TASKS_TEST
from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
from tau_bench.envs.retail.wiki import WIKI
from tau_bench.run import RunConfig, run

_original_get_env = tau_bench.envs.get_env


class TauBenchRetailEvaluator:
    """Tau Bench retail evaluator."""

    def __init__(self, model: str, temperature: float = 0.0, end_index: int = 50):
        """Initialize."""
        self.train_config = RunConfig(
            model=model,
            model_provider="openai",
            user_model=model,
            user_model_provider="openai",
            num_trials=1,
            env="retail",
            temperature=temperature,
            task_split="train",
            end_index=end_index,
        )
        self.test_config = RunConfig(
            model=model,
            model_provider="openai",
            user_model=model,
            user_model_provider="openai",
            num_trials=1,
            env="retail",
            temperature=temperature,
            task_split="test",
            end_index=end_index,
        )

        self.baseline_prompt = WIKI
        self.train_set = [
            {"input": f"User Profile: {t.instruction}", "target": f"Expected Actions: {[a.model_dump_json() for a in t.actions]}"}
            for t in TASKS_TRAIN
        ]
        self.test_set = [
            {"input": f"User Profile: {t.instruction}", "target": f"Expected Actions: {[a.model_dump_json() for a in t.actions]}"}
            for t in TASKS_TEST
        ]

    def evaluate(self, prompt: Prompt, validation_set: list[dict]) -> float:
        """Evaluate the given prompt on the validation set."""

        def _patched_get_env(*args, **kwargs):
            """Replace system prompt with wiki prompt."""
            env = _original_get_env(*args, **kwargs)
            env.wiki = prompt.content
            return env

        tau_bench.run.get_env = _patched_get_env
        tau_bench.envs.get_env = _patched_get_env

        if validation_set == self.train_set:
            run_config = self.train_config
        elif validation_set == self.test_set:
            run_config = self.test_config
        else:
            raise ValueError("Invalid split. Must be 'train' or 'test'.")

        results = run(run_config)
        for r in results:
            if r.reward < 1.0:
                input = f"User Profile: {r.info['task']['instruction']}"
                actual = f"Expected Actions: {r.info['task']['actions']}"
                prediction = f"Actual Actions: {r.traj[2:]}"
                error = PredictionError(input=input, prediction=prediction, actual=actual)
                prompt.errors.append(error)

        rewards = [r.reward for r in results]
        score = sum(rewards) / len(rewards)

        return score

    def __call__(self, prompt: Prompt, validation_set: list[dict]):
        """Evaluate the given prompt on the validation set."""
        score = self.evaluate(prompt, validation_set=validation_set)
        return score
