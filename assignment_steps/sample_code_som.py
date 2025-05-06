import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# --- Parameters ---
input_dim = 784  # For MNIST
latent_dim = 32
som_dim = (5, 5)  # 5x5 grid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Encoder and Decoder ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)


# --- SOM Module ---
class SOMLayer(nn.Module):
    def __init__(self, m, n, dim):
        super().__init__()
        self.m = m
        self.n = n
        self.num_prototypes = m * n
        self.prototype_dim = dim
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, dim))

        # Precompute SOM grid coordinates
        self.coords = self.create_som_grid(m, n).to(device)

    def create_som_grid(self, m, n):
        x, y = torch.meshgrid(torch.arange(m), torch.arange(n), indexing='ij')
        coords = torch.stack([x.flatten(), y.flatten()], dim=1).float()
        return coords

    def forward(self, z):
        # z: (batch, latent_dim), prototypes: (num_prototypes, latent_dim)
        # Compute squared Euclidean distances
        dists = torch.cdist(z, self.prototypes)
        min_idx = dists.argmin(dim=1)
        winner_prototypes = self.prototypes[min_idx]
        return winner_prototypes, min_idx

    def som_loss(self, z):
        # Encourage neighboring prototypes to be similar
        proto_dists = torch.cdist(self.prototypes, self.prototypes)
        coord_dists = torch.cdist(self.coords, self.coords)
        gaussian_weights = torch.exp(-coord_dists / (2 * 1.0**2))  # fixed sigma

        sim_loss = (proto_dists ** 2 * gaussian_weights).mean()
        return sim_loss


# --- Full Autoencoder with SOM ---
class DPSOM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.som = SOMLayer(*som_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        som_z, _ = self.som(z)
        x_recon = self.decoder(som_z)
        return x_recon, z, som_z

    def compute_loss(self, x):
        x_recon, z, som_z = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)
        som_loss = self.som.som_loss(z)
        total_loss = recon_loss + 0.1 * som_loss
        return total_loss, recon_loss, som_loss


# --- Training Loop ---
def train():
    model = DPSOM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=64, shuffle=True)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            loss, recon_loss, som_loss = model.compute_loss(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Total Loss = {total_loss / len(loader):.4f}")

train()

