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

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | q_proj, v_proj |
| Epochs | 3 |
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Batch size (effective) | 16 |
| Max seq length | 512 |
| Quantization | 4-bit NF4 + double quant |

---

## Results

| Metric | Paper (BF16) | This Replication (QLoRA) |
|--------|-------------|--------------------------|
| pass@1 | 37.8% | 26.83% |
| pass@5 | 58.4% | 35.91% |
| pass@10 | 66.1% | 38.41% |

> **Note:** This replication trained on a 500-sample subset of CodeSearchNet Python due to Colab free tier T4 memory and time constraints. The paper trained on the full dataset with full-precision BF16 weights. The gap in pass@k scores is expected given the reduced training data and 4-bit quantization.

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
