import torch
import pickle
import random
import argparse
import zstandard as zstd
import torch_geometric.transforms as T
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool

# Model definition
class GraphPairModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphPairModel, self).__init__()
        
        # GNN for encoding graphs
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # MLP for prediction
        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels)
        )

    def _forward(self, data):
        data = T.GCNNorm().forward(data)
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        h = self.conv1(x, edge_index, edge_weight)
        h = torch.relu(h)
        h = self.conv2(h, edge_index, edge_weight)
        h = torch.relu(h)
        h = global_mean_pool(h, data.batch)
        return h
    
    def forward(self, data1, data2):
        h1 = self._forward(data1)
        h2 = self._forward(data2)
       
        # Combine graph embeddings
        
        h_pair = h1 - h2
        # h_pair = torch.cat([h1, h2], dim=1)

        # Predict label
        res = self.mlp(h_pair)

        return res

# Load compressed pickle data
def load_compressed_pickle(file_path):
    with open(file_path, 'rb') as compressed_file:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as decompressed_stream:
            data = pickle.load(decompressed_stream)
    return data

# Command-line arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--num_iters', type=int, default=10, help="Number of training iterations")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden layer dimension")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay")
    return parser

# Main script
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    # Load data
    graphs_data = load_compressed_pickle("graphs.pkl.zst")
    
    # Assuming graphs_data is a list of (data1, data2, label) tuples
    random.shuffle(graphs_data)
    train_data = graphs_data[:int(0.8 * len(graphs_data))]
    test_data = graphs_data[int(0.8 * len(graphs_data)):]

    # Set random seed
    torch.manual_seed(args.seed)
    
    # Select device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model and optimizer setup
    in_channels = train_data[0][0].x.size(1)  # Feature dimension of the node
    out_channels = len(set([label for _, _, label in graphs_data]))  # Number of classes
    model = GraphPairModel(in_channels, args.hidden_dim, out_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for iter in range(args.num_iters):
        model.train()  # Set model to training mode
        total_loss = 0
        
        # Training phase
        for data1, data2, label in train_data:
            data1, data2, label = data1.to(device), data2.to(device), torch.tensor([label]).to(device)

            optimizer.zero_grad()
            output = model(data1, data2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print the average loss for this iteration
        print(f"Iter {iter+1}/{args.num_iters}, Loss: {total_loss / len(train_data)}")

        # Print test accuracy every 10 iterations
        if (iter + 1) % 10 == 0:
            model.eval()  # Set model to evaluation mode
            correct = 0
            total = 0
            
            with torch.no_grad():  # No need to compute gradients for testing
                for data1, data2, label in test_data:
                    data1, data2, label = data1.to(device), data2.to(device), torch.tensor([label]).to(device)
                    output = model(data1, data2)
                    predicted = output.argmax(dim=1)  # Assuming classification task with `argmax` for prediction
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            accuracy = correct / total
            print(f"Test Accuracy after Iter {iter+1}: {accuracy * 100:.2f}%")

