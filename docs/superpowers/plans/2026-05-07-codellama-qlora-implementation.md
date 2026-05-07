# CodeLlama-7B QLoRA Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully modular Python project that fine-tunes CodeLlama-7B with QLoRA on CodeSearchNet Python and evaluates on HumanEval (pass@1/5/10), runnable on Google Colab free tier (T4).

**Architecture:** Thin Colab notebook orchestrates a `src/` Python package via `!python` shell calls. All hyperparameters live in `configs/qlora_config.yaml` and are parsed into typed dataclasses by `src/config.py`. Training uses `SFTTrainer` from `trl`; evaluation uses the official `human-eval` library.

**Tech Stack:** Python 3.10, PyTorch 2.2, Transformers 4.40, PEFT 0.10, TRL 0.8, BitsAndBytes 0.43, Datasets 2.18, Accelerate 0.29, human-eval 1.0, scipy 1.12

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All pip dependencies |
| `configs/qlora_config.yaml` | All hyperparameters (LoRA, training, quantization, generation) |
| `src/__init__.py` | Empty package marker |
| `src/config.py` | Parses YAML into `LoRAConfig`, `TrainingConfig`, `QuantizationConfig`, `GenerationConfig`, `ProjectConfig` dataclasses |
| `src/data.py` | Loads CodeSearchNet Python split, formats prompts, returns HuggingFace `Dataset` |
| `src/model.py` | Loads CodeLlama-7B with 4-bit NF4 quantization and applies LoRA adapter |
| `src/train.py` | Runs `SFTTrainer`, saves adapter weights, supports checkpoint resumption |
| `src/evaluate.py` | Generates HumanEval completions, writes samples.jsonl, calls `evaluate_functional_correctness` |
| `scripts/run_evaluation.py` | CLI entry point wrapping `src/evaluate.py` with argparse |
| `notebooks/colab_training.ipynb` | Colab orchestration: Drive mount → install → train → eval → display results |
| `README.md` | Setup, usage, architecture diagram, results table, citation |

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `configs/qlora_config.yaml`
- Create: `src/__init__.py`
- Create: `results/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p configs src notebooks scripts results
touch src/__init__.py results/.gitkeep
```

- [ ] **Step 2: Write `requirements.txt`**

```
torch>=2.2.0
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
bitsandbytes>=0.43.0
datasets>=2.18.0
accelerate>=0.29.0
scipy>=1.12.0
pyyaml>=6.0
human-eval @ git+https://github.com/openai/human-eval.git
```

- [ ] **Step 3: Write `configs/qlora_config.yaml`**

```yaml
lora:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
  bias: none
  task_type: CAUSAL_LM

training:
  model_name: codellama/CodeLlama-7b-hf
  dataset_name: code_search_net
  dataset_language: python
  output_dir: ./outputs
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 0.0002
  lr_scheduler_type: cosine
  warmup_ratio: 0.03
  max_seq_length: 512
  save_steps: 100
  logging_steps: 25
  bf16: true
  fp16: false
  optim: adamw_torch

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16

generation:
  temperature: 0.2
  top_p: 0.95
  max_new_tokens: 256
  num_samples: 10
  do_sample: true
```

- [ ] **Step 4: Verify YAML parses cleanly**

```bash
python -c "import yaml; d = yaml.safe_load(open('configs/qlora_config.yaml')); print(list(d.keys()))"
```

Expected output: `['lora', 'training', 'quantization', 'generation']`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt configs/qlora_config.yaml src/__init__.py results/.gitkeep
git commit -m "feat: add project scaffold, requirements, and config YAML"
```

---

## Task 2: Config Module (`src/config.py`)

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write `src/config.py`**

```python
from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    model_name: str = "codellama/CodeLlama-7b-hf"
    dataset_name: str = "code_search_net"
    dataset_language: str = "python"
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_seq_length: int = 512
    save_steps: int = 100
    logging_steps: int = 25
    bf16: bool = True
    fp16: bool = False
    optim: str = "adamw_torch"


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 256
    num_samples: int = 10
    do_sample: bool = True


@dataclass
class ProjectConfig:
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def load_config(config_path: str = "configs/qlora_config.yaml") -> ProjectConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(
        lora=LoRAConfig(**raw.get("lora", {})),
        training=TrainingConfig(**raw.get("training", {})),
        quantization=QuantizationConfig(**raw.get("quantization", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
    )
```

- [ ] **Step 2: Verify import and config loading**

```bash
python -c "
from src.config import load_config
cfg = load_config()
print('LoRA r:', cfg.lora.r)
print('Model:', cfg.training.model_name)
print('4-bit:', cfg.quantization.load_in_4bit)
print('Temp:', cfg.generation.temperature)
"
```

Expected output:
```
LoRA r: 8
Model: codellama/CodeLlama-7b-hf
4-bit: True
Temp: 0.2
```

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: add config dataclasses with YAML loader"
```

---

## Task 3: Data Module (`src/data.py`)

**Files:**
- Create: `src/data.py`

- [ ] **Step 1: Write `src/data.py`**

```python
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
        trust_remote_code=True,
    )

    def process(sample):
        return {"text": format_prompt(sample)}

    dataset = dataset.map(process, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)
    return dataset
```

- [ ] **Step 2: Verify prompt formatting logic (no network call)**

```bash
python -c "
from src.data import format_prompt
sample = {
    'func_documentation_string': 'Return the sum of two numbers.',
    'whole_func_string': 'def add(a, b):\n    return a + b',
}
result = format_prompt(sample)
print(result)
assert result.startswith('[INST]')
assert '[/INST]' in result
assert 'def add' in result
print('OK')
"
```

Expected output:
```
[INST] Return the sum of two numbers. [/INST]
def add(a, b):
    return a + b
OK
```

- [ ] **Step 3: Verify empty sample handling**

```bash
python -c "
from src.data import format_prompt
assert format_prompt({}) == ''
assert format_prompt({'func_documentation_string': '', 'whole_func_string': 'def f(): pass'}) == ''
print('Empty sample handling OK')
"
```

Expected output: `Empty sample handling OK`

- [ ] **Step 4: Commit**

```bash
git add src/data.py
git commit -m "feat: add data loading and prompt formatting for CodeSearchNet"
```

---

## Task 4: Model Module (`src/model.py`)

**Files:**
- Create: `src/model.py`

- [ ] **Step 1: Write `src/model.py`**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from .config import TrainingConfig, LoRAConfig, QuantizationConfig


def _bnb_config(quant_config: QuantizationConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=quant_config.load_in_4bit,
        bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_base_model_and_tokenizer(
    config: TrainingConfig,
    quant_config: QuantizationConfig,
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=_bnb_config(quant_config),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def apply_lora(model, lora_config: LoRAConfig):
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )
    return get_peft_model(model, peft_config)


def load_trained_model(
    adapter_path: str,
    config: TrainingConfig,
    quant_config: QuantizationConfig,
):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=_bnb_config(quant_config),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer
```

- [ ] **Step 2: Verify module imports cleanly (no GPU needed)**

```bash
python -c "
from src.model import load_base_model_and_tokenizer, apply_lora, load_trained_model, _bnb_config
from src.config import QuantizationConfig
cfg = QuantizationConfig()
bnb = _bnb_config(cfg)
print('load_in_4bit:', bnb.load_in_4bit)
print('quant_type:', bnb.bnb_4bit_quant_type)
print('Model module OK')
"
```

Expected output:
```
load_in_4bit: True
quant_type: nf4
Model module OK
```

- [ ] **Step 3: Commit**

```bash
git add src/model.py
git commit -m "feat: add model loading with 4-bit QLoRA and PEFT adapter setup"
```

---

## Task 5: Training Module (`src/train.py`)

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Write `src/train.py`**

```python
import os
from transformers import TrainingArguments
from trl import SFTTrainer
from .config import load_config
from .data import load_training_data
from .model import load_base_model_and_tokenizer, apply_lora


def train(config_path: str = "configs/qlora_config.yaml"):
    config = load_config(config_path)
    tc = config.training

    print("Loading dataset...")
    dataset = load_training_data(tc)
    print(f"Dataset size: {len(dataset)} samples")

    print("Loading model and tokenizer...")
    model, tokenizer = load_base_model_and_tokenizer(tc, config.quantization)
    model = apply_lora(model, config.lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=tc.output_dir,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler_type,
        warmup_ratio=tc.warmup_ratio,
        save_steps=tc.save_steps,
        logging_steps=tc.logging_steps,
        bf16=tc.bf16,
        fp16=tc.fp16,
        optim=tc.optim,
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=tc.max_seq_length,
        args=training_args,
    )

    checkpoint_dir = tc.output_dir
    resume = os.path.isdir(checkpoint_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(checkpoint_dir)
    )
    if resume:
        print(f"Resuming from checkpoint in {checkpoint_dir}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume)

    final_path = os.path.join(tc.output_dir, "final")
    print(f"Saving adapter to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print("Training complete.")


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Verify module imports cleanly**

```bash
python -c "
from src.train import train
print('train() function signature:', train.__code__.co_varnames[:1])
print('Train module OK')
"
```

Expected output:
```
train() function signature: ('config_path',)
Train module OK
```

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add SFTTrainer training loop with checkpoint resumption"
```

---

## Task 6: Evaluation Module (`src/evaluate.py`)

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 1: Write `src/evaluate.py`**

```python
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
```

- [ ] **Step 2: Verify module imports and function signatures**

```bash
python -c "
from src.evaluate import evaluate, generate_completions
import inspect
print('evaluate params:', list(inspect.signature(evaluate).parameters.keys()))
print('generate_completions params:', list(inspect.signature(generate_completions).parameters.keys()))
print('Evaluate module OK')
"
```

Expected output:
```
evaluate params: ['adapter_path', 'config_path', 'output_file']
generate_completions params: ['model', 'tokenizer', 'prompt', 'gen_config']
Evaluate module OK
```

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat: add HumanEval evaluation with pass@k and per-problem error handling"
```

---

## Task 7: Evaluation Entry Point (`scripts/run_evaluation.py`)

**Files:**
- Create: `scripts/run_evaluation.py`

- [ ] **Step 1: Write `scripts/run_evaluation.py`**

```python
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Run HumanEval evaluation on a trained CodeLlama QLoRA adapter"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="./outputs/final",
        help="Path to saved LoRA adapter directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qlora_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/samples.jsonl",
        help="Path for output samples JSONL file",
    )
    args = parser.parse_args()
    evaluate(args.adapter_path, args.config, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help text**

```bash
python scripts/run_evaluation.py --help
```

Expected output contains:
```
usage: run_evaluation.py [-h] [--adapter_path ADAPTER_PATH] [--config CONFIG] [--output OUTPUT]
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_evaluation.py
git commit -m "feat: add CLI evaluation entry point with argparse"
```

---

## Task 8: Colab Notebook (`notebooks/colab_training.ipynb`)

**Files:**
- Create: `notebooks/colab_training.ipynb`

- [ ] **Step 1: Write `notebooks/colab_training.ipynb`**

Create this file with the following content (valid Jupyter notebook JSON):

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# CodeLlama-7B QLoRA Fine-Tuning\n", "Paper: *Optimizing Python Code Completion with Parameter-Efficient Fine-Tuning*\n\n", "Run cells top-to-bottom. Training checkpoints are saved to Google Drive automatically."],
   "id": "cell-md-title"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 1: Mount Google Drive"],
   "id": "cell-md-1"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "print('Drive mounted at /content/drive')"
   ],
   "id": "cell-mount-drive"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 2: Clone Repository"],
   "id": "cell-md-2"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "REPO_URL = 'https://github.com/YOUR_USERNAME/Python-Code-Completion-CodeLlama-7B-.git'\n",
    "REPO_DIR = '/content/Python-Code-Completion-CodeLlama-7B-'\n",
    "\n",
    "if not os.path.exists(REPO_DIR):\n",
    "    !git clone $REPO_URL $REPO_DIR\n",
    "\n",
    "%cd $REPO_DIR\n",
    "print('Working directory:', os.getcwd())"
   ],
   "id": "cell-clone"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 3: Install Dependencies\n\n⚠️ Runtime will restart after installation. Re-run from Step 4 after restart."],
   "id": "cell-md-3"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q -r requirements.txt\n",
    "print('Dependencies installed. If prompted, restart the runtime and re-run from Step 4.')"
   ],
   "id": "cell-install"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 4: Set Up Drive Persistence\n\nSymlinks `./outputs/` to Google Drive so checkpoints survive Colab disconnections."],
   "id": "cell-md-4"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "REPO_DIR = '/content/Python-Code-Completion-CodeLlama-7B-'\n",
    "%cd $REPO_DIR\n",
    "\n",
    "DRIVE_OUTPUTS = '/content/drive/MyDrive/codellama-qlora/outputs'\n",
    "LOCAL_OUTPUTS = os.path.join(REPO_DIR, 'outputs')\n",
    "\n",
    "os.makedirs(DRIVE_OUTPUTS, exist_ok=True)\n",
    "\n",
    "if os.path.islink(LOCAL_OUTPUTS):\n",
    "    print('Symlink already exists, skipping.')\n",
    "elif os.path.isdir(LOCAL_OUTPUTS):\n",
    "    print('outputs/ directory exists locally (not symlinked). Checkpoints will NOT persist to Drive.')\n",
    "else:\n",
    "    os.symlink(DRIVE_OUTPUTS, LOCAL_OUTPUTS)\n",
    "    print(f'Symlinked: {LOCAL_OUTPUTS} -> {DRIVE_OUTPUTS}')"
   ],
   "id": "cell-drive-symlink"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 5: Verify GPU"],
   "id": "cell-md-5"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "print('CUDA available:', torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print('GPU:', torch.cuda.get_device_name(0))\n",
    "    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')\n",
    "else:\n",
    "    print('WARNING: No GPU detected. Go to Runtime > Change runtime type > GPU.')"
   ],
   "id": "cell-gpu-check"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 6: Verify Config"],
   "id": "cell-md-6"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.config import load_config\n",
    "cfg = load_config()\n",
    "print(f'Model: {cfg.training.model_name}')\n",
    "print(f'LoRA r={cfg.lora.r}, alpha={cfg.lora.lora_alpha}, targets={cfg.lora.target_modules}')\n",
    "print(f'Epochs: {cfg.training.num_train_epochs}, LR: {cfg.training.learning_rate}')\n",
    "print(f'4-bit NF4: {cfg.quantization.load_in_4bit}')"
   ],
   "id": "cell-verify-config"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 7: Train\n\n⏱️ Estimated time: 6-10 hours on T4. Checkpoints saved every 100 steps to Drive."],
   "id": "cell-md-7"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python src/train.py"
   ],
   "id": "cell-train"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 8: Evaluate on HumanEval\n\nGenerates 10 samples per problem and computes pass@1, pass@5, pass@10."],
   "id": "cell-md-8"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python scripts/run_evaluation.py --adapter_path ./outputs/final --output results/samples.jsonl"
   ],
   "id": "cell-eval"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## Step 9: Display Results"],
   "id": "cell-md-9"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "\n",
    "with open('results/metrics.json') as f:\n",
    "    metrics = json.load(f)\n",
    "\n",
    "print('\\n=== HumanEval Results ===')\n",
    "print(f\"pass@1:  {metrics.get('pass@1', 'N/A'):.4f}\")\n",
    "print(f\"pass@5:  {metrics.get('pass@5', 'N/A'):.4f}\")\n",
    "print(f\"pass@10: {metrics.get('pass@10', 'N/A'):.4f}\")"
   ],
   "id": "cell-results"
  }
 ]
}
```

- [ ] **Step 2: Verify notebook JSON is valid**

```bash
python -c "
import json
with open('notebooks/colab_training.ipynb') as f:
    nb = json.load(f)
print('nbformat:', nb['nbformat'])
print('cell count:', len(nb['cells']))
print('cell types:', [c['cell_type'] for c in nb['cells']])
print('Notebook JSON valid')
"
```

Expected output:
```
nbformat: 4
cell count: 19
cell types: ['markdown', 'markdown', 'code', 'markdown', 'code', 'markdown', 'code', 'markdown', 'code', 'markdown', 'code', 'markdown', 'code', 'markdown', 'code', 'markdown', 'code', 'markdown', 'code']
Notebook JSON valid
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_training.ipynb
git commit -m "feat: add Colab orchestration notebook with Drive persistence"
```

---

## Task 9: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Python Code Completion with CodeLlama-7B + QLoRA

Replication of *"Optimizing Python Code Completion with Parameter-Efficient Fine-Tuning"*.  
Fine-tunes CodeLlama-7B with LoRA (rank=8) on CodeSearchNet Python, evaluated on HumanEval (pass@1/5/10).  
Runs on Google Colab free tier (T4, 16GB VRAM) using QLoRA 4-bit quantization.

---

## Architecture

```
configs/qlora_config.yaml   ← all hyperparameters
        │
        ▼
src/config.py               ← typed dataclasses
        │
   ┌────┴────┐
   ▼         ▼
src/data.py  src/model.py   ← CodeSearchNet loader / CodeLlama + QLoRA setup
        │
        ▼
src/train.py                ← SFTTrainer (trl) → outputs/final/
        │
        ▼
src/evaluate.py             ← HumanEval → results/metrics.json
```

---

## Quick Start (Google Colab)

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. Set runtime to **GPU (T4)**  
   `Runtime → Change runtime type → GPU`
3. Update the `REPO_URL` in Step 2 to your fork
4. Run cells top-to-bottom

Checkpoints are automatically saved to `My Drive/codellama-qlora/outputs/`.  
If the session disconnects, re-run from **Step 4** to resume training.

---

## Project Structure

```
├── configs/
│   └── qlora_config.yaml      # all hyperparameters
├── src/
│   ├── config.py              # dataclass config loader
│   ├── data.py                # CodeSearchNet → formatted dataset
│   ├── model.py               # 4-bit model + LoRA adapter
│   ├── train.py               # SFTTrainer training loop
│   └── evaluate.py            # HumanEval pass@k evaluation
├── notebooks/
│   └── colab_training.ipynb   # Colab orchestration notebook
├── scripts/
│   └── run_evaluation.py      # standalone eval CLI
└── results/                   # samples.jsonl + metrics.json (git-ignored)
```

---

## Running Locally

```bash
pip install -r requirements.txt

# Train
python src/train.py

# Evaluate (after training)
python scripts/run_evaluation.py --adapter_path ./outputs/final
```

---

## Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| LoRA rank (r) | 8 | Paper |
| LoRA alpha | 16 | Paper |
| LoRA dropout | 0.1 | Paper |
| Target modules | q_proj, v_proj | Paper |
| Epochs | 3 | Paper |
| Optimizer | AdamW | Paper |
| Learning rate | 2e-4 | Standard QLoRA default |
| Batch size (effective) | 16 | 4 × 4 grad accum |
| Max seq length | 512 | T4 memory constraint |
| Quantization | 4-bit NF4 + double quant | QLoRA adaptation |

---

## Results

| Metric | Paper (BF16) | This Replication (QLoRA) |
|--------|-------------|--------------------------|
| pass@1 | — | — |
| pass@5 | — | — |
| pass@10 | — | — |

*Fill in after running evaluation. QLoRA results may differ slightly from full-precision paper results.*

---

## Dependencies

See `requirements.txt`. Key versions:
- `transformers>=4.40.0`
- `peft>=0.10.0`
- `trl>=0.8.0`
- `bitsandbytes>=0.43.0`
- `human-eval` (installed from OpenAI GitHub)

---

## Citation

```bibtex
@article{optimizing-python-code-completion,
  title={Optimizing Python Code Completion with Parameter-Efficient Fine-Tuning},
  note={Paper replication with QLoRA adaptation for T4 GPU}
}
```
```

- [ ] **Step 2: Verify README renders (check for broken markdown)**

```bash
python -c "
with open('README.md') as f:
    content = f.read()
assert '## Quick Start' in content
assert '## Results' in content
assert 'pass@1' in content
assert 'qlora_config.yaml' in content
print('README structure OK')
print(f'README length: {len(content)} chars')
"
```

Expected output:
```
README structure OK
README length: <some number> chars
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: write comprehensive README with architecture, setup, and results table"
```

---

## Task 10: Final Verification

- [ ] **Step 1: Verify all source files are importable**

```bash
python -c "
from src.config import load_config, ProjectConfig
from src.data import load_training_data, format_prompt
from src.model import load_base_model_and_tokenizer, apply_lora, load_trained_model
from src.train import train
from src.evaluate import evaluate, generate_completions
print('All src modules import cleanly')
"
```

Expected output: `All src modules import cleanly`

- [ ] **Step 2: Verify full project file tree**

```bash
find . -not -path './.git/*' -not -path './outputs/*' -not -path './results/*' | sort
```

Expected output includes:
```
./README.md
./configs/qlora_config.yaml
./notebooks/colab_training.ipynb
./requirements.txt
./results/.gitkeep
./scripts/run_evaluation.py
./src/__init__.py
./src/config.py
./src/data.py
./src/evaluate.py
./src/model.py
./src/train.py
```

- [ ] **Step 3: Add `.gitignore`**

```bash
cat > .gitignore << 'EOF'
outputs/
results/*.jsonl
results/metrics.json
__pycache__/
*.pyc
.env
*.egg-info/
.ipynb_checkpoints/
EOF
git add .gitignore
git commit -m "chore: add .gitignore for outputs, results, and cache"
```

- [ ] **Step 4: Final commit with plan doc**

```bash
git add docs/superpowers/plans/2026-05-07-codellama-qlora-implementation.md
git commit -m "docs: add implementation plan"
```
