from argparse import ArgumentParser
from llama_cpp import Llama
from pydantic import BaseModel
from openai.lib._parsing._completions import type_to_response_format_param
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from constants import SYSTEM_PROMPT, PROMPT_TEMPLATE


class Response(BaseModel):
    output: str


def get_args():
    parser = ArgumentParser(
        description="Create a synthetic causal language modeling dataset for code."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_dataset_parser = subparsers.add_parser("create-dataset")
    create_dataset_parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output dataset jsonl file"
    )
    create_dataset_parser.add_argument(
        "-N", type=int, required=True, help="Number of samples to create"
    )

    return parser.parse_args()


def generate_sample(llm, snippet, prompt, category):
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=8192,
            temperature=0.7,
            response_format=type_to_response_format_param(Response),
        )
        content = response.choices[0].message.parsed
        if content:
            return {"snippet": snippet, "output": content.output, "category": category}
    except Exception as e:
        print(f"Error generating sample: {e}")
        return None


def generate_code_samples(num_samples: int) -> list:
    client = Llama.from_pretrained(
        repo_id="bartowski/gemma-2-9b-it-GGUF",
        filename="gemma-2-9b-it-Q5_K_S.gguf",
        verbose=False,
    )

    with open("prompts.json", "r") as f:
        pairs = json.load(f)

    categories = [
        "Non-plagiarised",
        "Light revision",
        "Near copy",
        "Heavy revision",
        "Original",
    ]

    input = []
    for pair in pairs:
        snippet = pair["snippet"]
        for prompt, category in zip(pair["prompts"], categories):
            input.append((snippet, prompt, category))

    samples = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for snippet, prompt, category in input[
            :num_samples
        ]:  # Limit to required samples
            futures.append(
                executor.submit(generate_sample, client, snippet, prompt, category)
            )

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Generating samples"
        ):
            result = future.result()
            if result:
                samples.append(result)

    return samples


def main():
    args = get_args()

    if args.command == "create-dataset":
        print(f"Generating {args.N} samples using the OpenAI API...")
        samples = generate_code_samples(args.N)

        with open(args.output, "w") as f:
            json.dump(samples, f)

        print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
