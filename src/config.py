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
