import argparse
import json
import datasets
import vllm
from constants import SYSTEM_PROMPT, PROMPT_TEMPLATE


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    return parser.parse_args()


def create_dataset(llm, sampling_params, prompts):
    outputs = llm.generate(prompts, sampling_params)
    return outputs


def main():
    args = parse_args()

    with open("prompts.json", "r") as f:
        prompts = json.load(f)

    llm = vllm.LLM(model=args.model, tensor_parallel_size=4)
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature, top_p=args.top_p
    )

    outputs = create_dataset(llm, sampling_params, prompts)

    with open("outputs.json", "w") as f:
        json.dump(outputs, f)


if __name__ == "__main__":
    main()
