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
    parser.add_argument("-N", type=int, default=100000)

    return parser.parse_args()


def create_prompts(num_texts: int):
    prompts = []
    categories = {
        "Non-plagiarised": 0,
        "Light revision": 1,
        "Near copy": 2,
        "Heavy revision": 3,
        "Original": 4,
    }

    def create_user_prompts(sample):
        text = sample["TEXT"]
        return [
            PROMPT_TEMPLATE.format(text=text, category=category)
            for category in categories
        ] * 5

    gutenberg_english = datasets.load_dataset(
        "sedthh/gutenberg_english", split="train", streaming=True
    )
    gutenberg_english.shuffle()

    x = 0
    for sample in gutenberg_english:
        user_prompts = create_user_prompts(sample)
        prompts += user_prompts
        x += 1

        if x == num_texts:
            break

    return prompts


def create_dataset(llm, sampling_params, prompts):
    outputs = llm.generate(prompts, sampling_params)


def main():
    args = parse_args()

    try:
        with open("prompts.json", "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        prompts = create_prompts(args.N)
        with open("prompts.json", "w") as f:
            json.dump(prompts, f)

    llm = vllm.LLM(model=args.model)
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature, top_p=args.top_p
    )

    outputs = create_dataset(llm, sampling_params, prompts)

    with open("outputs.json", "w") as f:
        json.dump(outputs, f)


if __name__ == "__main__":
    main()
