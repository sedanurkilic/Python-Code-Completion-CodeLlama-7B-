---
language: python
license: llama2
base_model: codellama/CodeLlama-7b-hf
tags:
  - code
  - code-generation
  - python
  - lora
  - qlora
  - peft
datasets:
  - code_search_net
metrics:
  - pass@1
  - pass@5
  - pass@10
---

# CodeLlama-7B QLoRA — Python Code Completion

Fine-tuned version of [CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) with QLoRA (4-bit NF4 quantization + LoRA rank=8) on CodeSearchNet Python, replicating *"Optimizing Python Code Completion with Parameter-Efficient Fine-Tuning"*.

## Model Details

| Property | Value |
|----------|-------|
| Base model | codellama/CodeLlama-7b-hf |
| Fine-tuning method | QLoRA (LoRA + 4-bit NF4 quantization) |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, v_proj |
| Training dataset | CodeSearchNet Python (500-sample subset) |
| Training hardware | Google Colab T4 (16GB VRAM) |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Effective batch size | 16 (4 × 4 gradient accumulation) |

## Results

Evaluated on [HumanEval](https://github.com/openai/human-eval) benchmark using the official `evaluate_functional_correctness` scorer with 10 samples per problem.

| Metric | Paper (full dataset, BF16) | This adapter (500 samples, QLoRA) |
|--------|---------------------------|-----------------------------------|
| pass@1 | 37.8% | **26.83%** |
| pass@5 | 58.4% | **35.91%** |
| pass@10 | 66.1% | **38.41%** |

> The gap relative to the paper is expected: this adapter was trained on a 500-sample subset due to Colab free-tier constraints, and uses 4-bit quantization instead of full BF16 precision.

## Training

Trained using the [`sedanurkilic/Python-Code-Completion-CodeLlama-7B-`](https://github.com/sedanurkilic/Python-Code-Completion-CodeLlama-7B-) codebase. Each CodeSearchNet sample was formatted as:

```
[INST] {docstring} [/INST]
{function_body}
```

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained("sedaklc/codellama-7b-qlora-humaneval")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, "sedaklc/codellama-7b-qlora-humaneval")
model.eval()

prompt = "[INST] Return the n-th Fibonacci number. [/INST]\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

## Limitations

- Trained on a 500-sample subset; performance improves significantly with more data.
- 4-bit quantization introduces a small quality degradation vs full-precision weights.
- Optimised for Python function completion from docstrings; not a general-purpose chat model.

## Citation

```bibtex
@article{optimizing-python-code-completion,
  title={Optimizing Python Code Completion with Parameter-Efficient Fine-Tuning},
  note={QLoRA replication — sedaklc/codellama-7b-qlora-humaneval}
}
```
