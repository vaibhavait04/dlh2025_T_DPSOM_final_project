from DPSOM_model import DPSOM
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
dataset = MNIST(root='data', download=True, transform=transform)

# Model
model = DPSOM(input_dim=784, som_shape=(10, 10), latent_dim=32)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train
train_dpsom(model, dataset, optimizer, n_epochs=20, batch_size=64, device=device)
