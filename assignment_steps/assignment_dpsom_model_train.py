
import torch
import torch.nn as nn

class SOM(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super(SOM, self).__init__()
        self.input_dim = input_dim
        self.n_clusters = n_clusters
        # Initialize SOM weights (cluster centroids)
        self.weights = nn.Parameter(torch.randn(n_clusters, input_dim))

    def forward(self, x):
        # Calculate the Euclidean distance between input and each cluster centroid
        distances = torch.cdist(x, self.weights)
        return distances

class DPSOM_SOMOnly(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super(DPSOM_SOMOnly, self).__init__()
        self.som = SOM(input_dim, n_clusters)

    def forward(self, x):
        # Get the distances to all clusters
        cluster_distances = self.som(x)
        # Assign each input to the closest cluster
        cluster_assignments = torch.argmin(cluster_distances, dim=1)
        return cluster_assignments

    def loss(self, x, cluster_assignments):
        # Calculate the mean distance to the nearest cluster
        cluster_distances = self.som(x)
        loss = cluster_distances.gather(1, cluster_assignments.unsqueeze(1)).mean()
        return loss


import torch
from torch.utils.data import DataLoader

def train_som_only(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for data, labels in train_loader:
        # Move data to the proper device (GPU or CPU)
        data = data.to(device)

        # Forward pass through the model
        cluster_assignments = model(data)

        # Compute the SOM loss (mean distance to closest cluster)
        loss = model.loss(data, cluster_assignments)

        # Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for reporting
        total_loss += loss.item()

    return total_loss / len(train_loader)

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import torch

def evaluate_clustering(model, test_loader, device):
    model.eval()
    all_labels = []
    all_assignments = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            cluster_assignments = model(data)

            # Store labels and assignments for evaluation
            all_labels.append(labels.cpu().numpy())
            all_assignments.append(cluster_assignments.cpu().numpy())

    # Concatenate all labels and cluster assignments
    all_labels = np.concatenate(all_labels)
    all_assignments = np.concatenate(all_assignments)

    # Calculate NMI (Normalized Mutual Information)
    nmi = normalized_mutual_info_score(all_labels, all_assignments)

    # Calculate Purity (percentage of correct cluster assignments)
    purity = accuracy_score(all_labels, all_assignments)

    return nmi, purity


import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification

# Example: Create a random dataset (for demonstration purposes)
# X, y = # make_classification(n_samples=1000, n_features=20, n_classes=5, random_state=42)
X, y = make_classification(
    n_samples=1000,      # Number of samples
    n_features=20,       # Number of features
    n_informative=2,     # Number of informative features
    n_classes=4,         # Adjust number of classes
    n_clusters_per_class=1,  # Adjust number of clusters per class
    random_state=42
)

# Convert data to torch tensors and create DataLoader
dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and optimizer
input_dim = X.shape[1]  # Number of features
n_clusters = 5          # Number of clusters (classes in this example)
model = DPSOM_SOMOnly(input_dim, n_clusters).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Set to desired number of epochs
    train_loss = train_som_only(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}")

# Evaluate the clustering on the test data
nmi, purity = evaluate_clustering(model, train_loader, device)
print(f"Clustering Evaluation - NMI: {nmi:.4f}, Purity: {purity:.4f}")

