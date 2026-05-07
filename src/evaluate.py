import os
import json
import torch
from typing import List
from .config import load_config, GenerationConfig
from .model import load_trained_model


def generate_completions(
    model,
    tokenizer,
    prompt: str,
    gen_config: GenerationConfig,
) -> List[str]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    completions = []
    with torch.no_grad():
        for _ in range(gen_config.num_samples):
            output = model.generate(
                **inputs,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                do_sample=gen_config.do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return completions


def evaluate(
    adapter_path: str = "./outputs/final",
    config_path: str = "configs/qlora_config.yaml",
    output_file: str = "results/samples.jsonl",
):
    config = load_config(config_path)

    print("Loading trained model...")
    model, tokenizer = load_trained_model(adapter_path, config.training, config.quantization)
    model.eval()

    from human_eval.data import read_problems
    problems = read_problems()
    print(f"Loaded {len(problems)} HumanEval problems")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    samples = []
    for i, (task_id, problem) in enumerate(problems.items()):
        prompt = f"[INST] Complete the following Python function:\n{problem['prompt']} [/INST]\n"
        try:
            completions = generate_completions(model, tokenizer, prompt, config.generation)
            for completion in completions:
                samples.append({"task_id": task_id, "completion": completion})
        except Exception as e:
            print(f"[ERROR] {task_id}: {e}")
            for _ in range(config.generation.num_samples):
                samples.append({"task_id": task_id, "completion": ""})

        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{len(problems)}")

    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Wrote {len(samples)} samples to {output_file}")

    print("Running pass@k evaluation...")
    from human_eval.evaluation import evaluate_functional_correctness
    results = evaluate_functional_correctness(output_file)

    print("\n=== HumanEval Results ===")
    for metric, value in sorted(results.items()):
        print(f"  {metric}: {value:.4f}")

    metrics_path = os.path.join(os.path.dirname(output_file), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return results


if __name__ == "__main__":
    import sys
    adapter = sys.argv[1] if len(sys.argv) > 1 else "./outputs/final"
    evaluate(adapter)
