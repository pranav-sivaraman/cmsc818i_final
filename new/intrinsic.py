import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load an open-source LLM (adjust the model path as needed)
model_name = "roberta-base"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True) 

# Set up the function to prompt the model
def get_similarity_decision(text1, text2):
    prompt = f"""
    You are a jury member tasked with determining whether two pieces of text are substantially similar. Consider their themes, phrasing, and overall structure. Provide a step-by-step explanation of your reasoning before making a final judgment.
    Text 1: {text1}
    Text 2: {text2}
    Do you believe these texts are substantially similar? Why or why not? Before giving your reasoning, just output one word - Yes or No.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract "Yes" or "No" from the response
    decision = response.split("\n")[0].strip().lower()
    if "yes" in decision[0:5]:
        return 1  # Yes
    elif "no" in decision[0:5]:
        return 0  # No
    else:
        return None  # Handle unexpected cases

# List of tuples to process
import json
with open('dataset.json', "r") as f:
    pairs = json.load(f)

# Store results
results = []

i = 0
for text1, text2, _ in pairs:
    print('HELLO ' + str(i))
    result = get_similarity_decision(text1, text2)
    results.append(result)
    i += 1

import csv

# Open a CSV file in write mode
with open(model_name + '_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Similar'])

    # Write the results
    for pair in results:
        writer.writerow(['Yes' if pair == 1 else 'No'])
