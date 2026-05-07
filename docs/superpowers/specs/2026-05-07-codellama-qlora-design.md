# Design Spec: Python Code Completion with CodeLlama-7B + QLoRA

## Goal

Fine-tuning CodeLlama-7B with LoRA (rank=8) on CodeSearchNet Python and evaluating on HumanEval with exact pass@1, pass@5, pass@10 metrics. Training runs on Google Colab free tier (T4, 16GB VRAM), requiring QLoRA (4-bit quantization) to fit the model in memory.

---

## Project Structure

```
Python-Code-Completion-CodeLlama-7B-/
├── README.md
├── requirements.txt
├── configs/
│   └── qlora_config.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
│   └── colab_training.ipynb
├── scripts/
│   └── run_evaluation.py
└── docs/
    └── superpowers/specs/
```

---

## Architecture & Data Flow

1. `configs/qlora_config.yaml` holds all hyperparameters. `src/config.py` parses it into typed Python dataclasses (`LoRAConfig`, `TrainingConfig`, `GenerationConfig`).
2. `src/data.py` streams `code_search_net` (Python split) from HuggingFace Datasets. Each sample is formatted as:
   ```
   [INST] {docstring} [/INST]
   {function_body}
   ```
   Returns a HuggingFace `Dataset` ready for `SFTTrainer`.
3. `src/model.py` loads `codellama/CodeLlama-7b-hf` in 4-bit NF4 quantization via `BitsAndBytesConfig`, then wraps it with PEFT `LoraConfig` targeting `q_proj` and `v_proj`.
4. `src/train.py` instantiates `SFTTrainer` (from `trl`) with the model, tokenizer, dataset, and `TrainingArguments`. Saves adapter weights to `./outputs/` with best-checkpoint tracking.
5. `src/evaluate.py` loads the saved LoRA adapter on top of the quantized base model, generates 10 completions per HumanEval problem, and calls `human_eval.evaluate_functional_correctness` for pass@1/5/10.
6. `notebooks/colab_training.ipynb` mounts Google Drive, installs dependencies, and orchestrates steps 1–5 via `!python` shell calls. Checkpoints are persisted to Google Drive to survive session drops.

---

## Hyperparameters

### LoRA (from paper)
| Parameter | Value |
|-----------|-------|
| rank (r) | 8 |
| alpha | 16 |
| dropout | 0.1 |
| target modules | `q_proj`, `v_proj` |
| bias | none |

### Training
| Parameter | Value |
|-----------|-------|
| epochs | 3 |
| optimizer | AdamW |
| learning rate | 2e-4 |
| per-device batch size | 4 |
| gradient accumulation steps | 4 (effective batch = 16) |
| max sequence length | 512 tokens |
| lr scheduler | cosine |
| warmup ratio | 0.03 |
| save steps | 100 |
| fp16 | False (bfloat16 for compute) |

### Quantization (QLoRA adaptation for T4)
| Parameter | Value |
|-----------|-------|
| load_in_4bit | True |
| bnb_4bit_quant_type | nf4 |
| bnb_4bit_use_double_quant | True |
| bnb_4bit_compute_dtype | bfloat16 |

### Generation (HumanEval)
| Parameter | Value |
|-----------|-------|
| temperature | 0.2 |
| top_p | 0.95 |
| max_new_tokens | 256 |
| samples per problem | 10 |
| do_sample | True |

---

## Key Implementation Decisions

- **Prompt format:** CodeLlama instruction format `[INST] ... [/INST]` — matches the model's pretraining format and the paper's use of "function signatures and docstrings as prompts."
- **Max seq length 512:** Balances T4 VRAM constraints against coverage of CodeSearchNet samples. Sequences longer than 512 tokens are truncated.
- **10 samples per HumanEval problem:** Required to compute pass@5 and pass@10 accurately via the unbiased estimator in the `human-eval` library.
- **Google Drive symlink:** The Colab notebook symlinks `./outputs/` to `/content/drive/MyDrive/codellama-qlora/` so checkpoints survive Colab disconnections.
- **Resume from checkpoint:** `TrainingArguments(resume_from_checkpoint=True)` is set so training can be restarted without loss of progress.

---

## Error Handling

- `evaluate.py` wraps each HumanEval problem's code execution in a try/except block, logging failures individually without aborting the full evaluation run.
- `train.py` relies on HuggingFace `Trainer`'s built-in checkpoint resumption for session drops.
- No custom retry logic — Colab disconnects are handled at the notebook level by re-running cells from the checkpoint step.

---

## Dependencies

```
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
bitsandbytes>=0.43.0
datasets>=2.18.0
accelerate>=0.29.0
human-eval>=1.0.0
torch>=2.2.0
scipy>=1.12.0
```

---

## Out of Scope

- Pushing the trained adapter to HuggingFace Hub (can be added manually)
- Quantitative ablation studies (paper replication only)
- Unit tests (research replication; Colab notebook is the integration test)
- Multi-GPU training
