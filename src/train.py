import os
from trl import SFTTrainer, SFTConfig
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

    training_args = SFTConfig(
        output_dir=tc.output_dir,
        num_train_epochs=tc.num_train_epochs,
        per_device_train_batch_size=tc.per_device_train_batch_size,
        gradient_accumulation_steps=tc.gradient_accumulation_steps,
        learning_rate=tc.learning_rate,
        lr_scheduler_type=tc.lr_scheduler_type,
        warmup_steps=100,
        save_steps=tc.save_steps,
        logging_steps=tc.logging_steps,
        bf16=tc.bf16,
        fp16=tc.fp16,
        optim=tc.optim,
        report_to="none",
        load_best_model_at_end=False,
        max_length=tc.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
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
