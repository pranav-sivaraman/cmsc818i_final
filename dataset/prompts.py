import argparse
import json
import datasets
import itertools
from constants import SYSTEM_PROMPT, PROMPT_TEMPLATE
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser()

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
        ]

    gutenberg_english = datasets.load_dataset(
        "sedthh/gutenberg_english", split="train", streaming=True
    )
    gutenberg_english = gutenberg_english.shuffle()

    def process_sample(sample):
        return create_user_prompts(sample)

    with ThreadPoolExecutor() as executor:
        # Collect up to num_texts samples
        samples = itertools.islice(gutenberg_english, num_texts)
        results = executor.map(process_sample, samples)

    # Flatten results into a single list
    prompts = list(itertools.chain.from_iterable(results))

    return prompts

def create_dataset(llm, sampling_params, prompts):
    outputs = llm.generate(prompts, sampling_params)


def main():
    args = parse_args()

    print("Creating Prompts!")

    try:
        with open("prompts1.json", "r") as f:
            prompts = json.load(f)
    except FileNotFoundError:
        prompts = create_prompts(args.N)
        with open("prompts1.json", "w") as f:
            json.dump(prompts, f)

    print("Finished Creating Prompts!")


if __name__ == "__main__":
    main()
