import gradio as gr

EXAMPLES = {
    "has_close_elements — check threshold proximity": {
        "docstring": (
            "Check if in given list of numbers, are any two numbers closer to each\n"
            "other than given threshold.\n"
            ">>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "False\n"
            ">>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "True"
        ),
        "completion": (
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False"
        ),
    },
    "separate_paren_groups — split nested parentheses": {
        "docstring": (
            "Input to this function is a string containing multiple groups of nested\n"
            "parentheses. Your goal is to separate those groups into separate strings\n"
            "and return the list of those. Separate groups are balanced (each open\n"
            "brace is properly closed) and not nested within each other. Ignore any\n"
            "spaces in the input string.\n"
            ">>> separate_paren_groups('( ) (( )) (( )( ))')\n"
            "['()', '(())', '(()())']"
        ),
        "completion": (
            "def separate_paren_groups(paren_string: str) -> List[str]:\n"
            "    result = []\n"
            "    depth = 0\n"
            "    current = ''\n"
            "    for char in paren_string:\n"
            "        if char == '(':\n"
            "            depth += 1\n"
            "            current += char\n"
            "        elif char == ')':\n"
            "            depth -= 1\n"
            "            current += char\n"
            "            if depth == 0:\n"
            "                result.append(current)\n"
            "                current = ''\n"
            "    return result"
        ),
    },
    "rescale_to_unit — linear normalisation to [0, 1]": {
        "docstring": (
            "Given a list of numbers (of at least two elements), apply a linear\n"
            "transform to that list, such that the smallest number will become 0 and\n"
            "the largest will become 1.\n"
            ">>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "[0.0, 0.25, 0.5, 0.75, 1.0]"
        ),
        "completion": (
            "def rescale_to_unit(numbers: List[float]) -> List[float]:\n"
            "    min_val = min(numbers)\n"
            "    max_val = max(numbers)\n"
            "    return [(x - min_val) / (max_val - min_val) for x in numbers]"
        ),
    },
    "remove_duplicates — keep only unique elements": {
        "docstring": (
            "From a list of integers, remove all elements that occur more than once.\n"
            "Keep the order of elements left the same as in the input.\n"
            ">>> remove_duplicates([1, 2, 3, 2, 4])\n"
            "[1, 3, 4]"
        ),
        "completion": (
            "def remove_duplicates(numbers: List[int]) -> List[int]:\n"
            "    from collections import Counter\n"
            "    counts = Counter(numbers)\n"
            "    return [x for x in numbers if counts[x] == 1]"
        ),
    },
    "sort_third — sort every third index in-place": {
        "docstring": (
            "This function takes a list l and returns a list l' such that l' is\n"
            "identical to l in the indices that are not divisible by three, while\n"
            "its values at the indices that are divisible by three are equal to the\n"
            "values of the corresponding indices of l, but sorted.\n"
            ">>> sort_third([1, 2, 3])\n"
            "[1, 2, 3]\n"
            ">>> sort_third([5, 6, 3, 4, 8, 9, 2])\n"
            "[2, 6, 3, 4, 8, 9, 5]"
        ),
        "completion": (
            "def sort_third(l: list) -> list:\n"
            "    thirds = sorted(l[i] for i in range(0, len(l), 3))\n"
            "    result = list(l)\n"
            "    j = 0\n"
            "    for i in range(0, len(l), 3):\n"
            "        result[i] = thirds[j]\n"
            "        j += 1\n"
            "    return result"
        ),
    },
}

EXAMPLE_NAMES = list(EXAMPLES.keys())


def load_example(name: str):
    ex = EXAMPLES[name]
    return ex["docstring"], ex["completion"]


with gr.Blocks(title="CodeLlama-7B QLoRA — Python Code Completion Demo") as demo:
    gr.Markdown(
        """
# CodeLlama-7B QLoRA — Python Code Completion

Fine-tuned on CodeSearchNet Python with LoRA (rank=8) · Evaluated on HumanEval

| pass@1 | pass@5 | pass@10 |
|--------|--------|---------|
| 26.83% | 35.91% | 38.41% |

> **Pre-computed outputs from fine-tuned CodeLlama-7B + QLoRA model (inference requires GPU)**
> Model: [`sedaklc/codellama-7b-qlora-humaneval`](https://huggingface.co/sedaklc/codellama-7b-qlora-humaneval)
        """
    )

    dropdown = gr.Dropdown(
        choices=EXAMPLE_NAMES,
        value=EXAMPLE_NAMES[0],
        label="Select a HumanEval problem",
    )

    with gr.Row():
        docstring_box = gr.Textbox(
            label="Docstring (input prompt)",
            lines=10,
            interactive=False,
        )
        completion_box = gr.Code(
            label="Model completion (output)",
            language="python",
            lines=10,
            interactive=False,
        )

    dropdown.change(fn=load_example, inputs=dropdown, outputs=[docstring_box, completion_box])
    demo.load(fn=load_example, inputs=dropdown, outputs=[docstring_box, completion_box])

if __name__ == "__main__":
    demo.launch()
