from datasets import load_dataset, Dataset
from .config import TrainingConfig


def format_prompt(sample: dict) -> str:
    docstring = (sample.get("func_documentation_string") or "").strip()
    code = (sample.get("whole_func_string") or "").strip()
    if not docstring or not code:
        return ""
    return f"[INST] {docstring} [/INST]\n{code}"


def load_training_data(config: TrainingConfig) -> Dataset:
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_language,
        split="train",
    )

    def process(sample):
        return {"text": format_prompt(sample)}

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    dataset = dataset.select(range(500))
    return dataset
