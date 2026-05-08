import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "codellama/CodeLlama-7b-hf"
ADAPTER_ID = "sedaklc/codellama-7b-qlora-humaneval"

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()
print("Model ready.")


def generate_completion(docstring: str, temperature: float, max_new_tokens: int) -> str:
    if not docstring.strip():
        return ""
    prompt = f"[INST] {docstring.strip()} [/INST]\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


EXAMPLES = [
    ["Return n-th Fibonacci number.", 0.2, 256],
    ["Filter an input list of strings only for ones that start with a given prefix.", 0.2, 256],
    ["Return True if list elements are monotonically increasing or decreasing.\n>>> monotonic([1, 2, 4, 20])\nTrue\n>>> monotonic([1, 20, 4, 10])\nFalse", 0.2, 256],
    ["Return median of elements in the list l.\n>>> median([3, 1, 2, 4, 5])\n3\n>>> median([-10, 4, 6, 1000, 10, 3])\n8.0", 0.2, 256],
    ["Return list of prime factors of given integer in the order from smallest to largest.\n>>> factorize(8)\n[2, 2, 2]\n>>> factorize(25)\n[5, 5]", 0.2, 256],
]

with gr.Blocks(title="CodeLlama-7B QLoRA — Python Code Completion") as demo:
    gr.Markdown(
        """
# CodeLlama-7B QLoRA — Python Code Completion

Fine-tuned on CodeSearchNet Python with LoRA (rank=8) and evaluated on HumanEval.
**Results:** pass@1 = 26.83% · pass@5 = 35.91% · pass@10 = 38.41%
Model: [`sedaklc/codellama-7b-qlora-humaneval`](https://huggingface.co/sedaklc/codellama-7b-qlora-humaneval)
        """
    )

    with gr.Row():
        with gr.Column():
            docstring = gr.Textbox(
                label="Python function docstring",
                placeholder="Describe the function you want implemented...",
                lines=6,
            )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.2, step=0.01, label="Temperature"
                )
                max_tokens = gr.Slider(
                    minimum=64, maximum=512, value=256, step=32, label="Max new tokens"
                )
            submit_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Generated code", lines=16, show_copy_button=True)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[docstring, temperature, max_tokens],
        outputs=output,
        fn=generate_completion,
        cache_examples=False,
    )

    submit_btn.click(fn=generate_completion, inputs=[docstring, temperature, max_tokens], outputs=output)

if __name__ == "__main__":
    demo.launch()
