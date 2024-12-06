import stanza
import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import pickle

# Load the Stanza NLP pipeline
stanza.download("en")
nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
depparse = nlp.processors["depparse"]
dep_relations = depparse.get_known_relations()
dep_relations.sort()

UNIVERSAL_DEPENDENCIES = {}
for idx, rel in enumerate(dep_relations):
    UNIVERSAL_DEPENDENCIES[rel] = idx

# Load a more modern transformer model and tokenizer (e.g., BERT-large or RoBERTa)
device = torch.device("mps")  # Use the MPS device (Apple Silicon)
model_name = "roberta-base"
transformer_model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_transformer_embeddings(word):
    """
    Get transformer embeddings for a single word.

    Args:
        word (str): Input word.

    Returns:
        torch.Tensor: Transformer embedding of the word.
    """
    inputs = tokenizer(word, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Ensure all inputs are on the correct device
    with torch.no_grad():
        outputs = transformer_model(**inputs)
    # Use the embeddings from the last hidden state (taking the mean for simplicity if word is tokenized into subwords)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)

def paragraph_to_graph(paragraph):
    """
    Converts a paragraph into a PyTorch Geometric graph representation.

    Args:
        paragraph (str): The input paragraph.

    Returns:
        Data: A PyTorch Geometric Data object representing the dependency graph.
    """
    # Parse the paragraph using Stanza
    doc = nlp(paragraph)

    # Initialize lists for nodes, edges, edge features, and node features
    nodes = []
    edges = []
    edge_features = []
    node_features = []

    for sentence in doc.sentences:
        for word in sentence.words:
            # Add the word as a node
            nodes.append(word.text)

            # Add its transformer embedding as a feature
            embedding = get_transformer_embeddings(word.text)
            node_features.append(embedding)

            # Add directed edges for dependencies with features
            if word.head != 0:  # Word with a head of 0 is the root
                edges.append((word.head - 1 + len(nodes) - len(sentence.words), word.id - 1 + len(nodes) - len(sentence.words)))  # Convert 1-based to 0-based indexing and adjust for global index
                edge_features.append(UNIVERSAL_DEPENDENCIES.get(word.deprel, -1))  # Map dependency relation to index, use -1 if not found

    # Convert edges to tensor format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Convert edge features to tensor format
    edge_attr = torch.tensor(edge_features, dtype=torch.long)

    # Stack node features into a single tensor
    x = torch.stack(node_features)

    # Create the PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def main():
    df = pd.read_json("pairs.json")

    categories = {
        "Non-plagiarised": 0,
        "Light revision": 1,
        "Near copy": 2,
        "Heavy revision": 3,
        "Original": -1,
    }

    pair_num = 1

    graphs = []
    for _, row in df.iterrows():
        print(f"Creating graph {pair_num}")
        pair_num += 1

        original = row["original"]
        answer = row["answer"]

        original_graph = paragraph_to_graph(original)
        answer_graph = paragraph_to_graph(answer)
        label = categories[row["category"]]

        graphs.append((original_graph, answer_graph, label))

    # Save the list of graphs to a file
    with open("graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)

# Example usage
if __name__ == "__main__":
    main()
