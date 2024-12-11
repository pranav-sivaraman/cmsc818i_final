import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import itertools
import datasets
from constants import SYSTEM_PROMPT, PROMPT_TEMPLATE
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N", type=int, default=10000, help="Number of texts to process"
    )
    parser.add_argument(
        "-batch_size", type=int, default=1000, help="Batch size for processing prompts"
    )
    return parser.parse_args()


def save_prompts_to_file(prompts, filename):
    """Appends prompts to a file, one per line."""
    with open(filename, "a") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")


def create_prompts(num_texts: int, batch_size: int, output_file: str):
    """Generates prompts in batches and saves them incrementally to a file."""
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

    # Load and shuffle the dataset
    gutenberg_english = datasets.load_dataset(
        "sedthh/gutenberg_english", split="train", streaming=True
    )
    gutenberg_english = gutenberg_english.shuffle()

    batch = []
    x = 0
    with tqdm(total=num_texts, desc="Processing prompts") as pbar:
        for sample in gutenberg_english:
            batch.append(sample)
            x += 1

            if x % batch_size == 0:
                # Process the batch in parallel
                with ThreadPoolExecutor() as executor:
                    batch_results = executor.map(create_user_prompts, batch)
                save_prompts_to_file(
                    itertools.chain.from_iterable(batch_results), output_file
                )
                batch = []  # Clear batch
                pbar.update(batch_size)

            if x == num_texts:
                break

        # Process any remaining samples in the final batch
        if batch:
            with ThreadPoolExecutor() as executor:
                batch_results = executor.map(create_user_prompts, batch)
            save_prompts_to_file(
                itertools.chain.from_iterable(batch_results), output_file
            )
            pbar.update(len(batch))


def main():
    args = parse_args()

    output_file = "prompts.json"
    # Ensure the file is empty before starting
    open(output_file, "w").close()

    print("Creating Prompts!")
    create_prompts(args.N, args.batch_size, output_file)
    print("Finished Creating Prompts!")


if __name__ == "__main__":
    main()
