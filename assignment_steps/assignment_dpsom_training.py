import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def train_dpsom(model, dataset, optimizer, scheduler=None, n_epochs=100, batch_size=128, device='cuda', beta=0.4):
    """
    General training loop for DPSOM.

    Parameters:
        model: DPSOM model instance (e.g., DPSOM or TempDPSOM)
        dataset: PyTorch Dataset
        optimizer: Torch optimizer
        scheduler: Optional learning rate scheduler
        n_epochs: Number of epochs
        batch_size: Batch size
        device: 'cuda' or 'cpu'
        beta: Weight for the SOM loss term
    """
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_recon_loss = 0
        epoch_som_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{n_epochs}"):
            x = batch[0].to(device)

            # Forward pass
            x_hat, q, bmu_locs, _ = model(x)

            # Compute losses
            recon_loss = torch.nn.functional.mse_loss(x_hat, x)
            som_loss = model.som_loss(q, bmu_locs)

            total_loss = recon_loss + beta * som_loss

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_som_loss += som_loss.item()

        if scheduler:
            scheduler.step()

        avg_recon = epoch_recon_loss / len(dataloader)
        avg_som = epoch_som_loss / len(dataloader)

        print(f"Epoch {epoch}: Recon Loss = {avg_recon:.4f}, SOM Loss = {avg_som:.4f}")

    print("Training finished.")
