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
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=_bnb_config(quant_config),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model, tokenizer
