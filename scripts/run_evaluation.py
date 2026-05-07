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
