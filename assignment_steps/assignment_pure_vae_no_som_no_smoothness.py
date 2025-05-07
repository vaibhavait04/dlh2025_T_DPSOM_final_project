import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ===== Simulated Dataset (toy sequence data) =====
def generate_dummy_sequence_data(batch_size, seq_len, input_dim):
    """
    Generate batch of sequences with some smooth transitions
    """
    x = torch.randn(batch_size, 1, input_dim)
    drift = torch.randn(batch_size, seq_len, input_dim) * 0.1
    return x + torch.cumsum(drift, dim=1)

# ===== Pure VAE Model (no SOM, no Temporal) =====
class VAE_Only(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE_Only, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mu and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_seq):
        batch_size, seq_len, input_dim = x_seq.shape
        x_flat = x_seq.view(-1, input_dim)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        x_recon = x_recon.view(batch_size, seq_len, input_dim)
        mu = mu.view(batch_size, seq_len, -1)
        logvar = logvar.view(batch_size, seq_len, -1)
        return x_recon, mu, logvar

    def loss_function(self, x_seq, x_recon, mu, logvar):
        recon_loss = F.mse_loss(x_recon, x_seq)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss

# ===== Training Parameters =====
input_dim = 20
latent_dim = 10
hidden_dim = 64
seq_len = 15
batch_size = 32
num_epochs = 20
lr = 1e-3

# ===== Model and Optimizer =====
model = VAE_Only(input_dim, latent_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# ===== Training Loop =====
for epoch in range(num_epochs):
    model.train()
    x_seq = generate_dummy_sequence_data(batch_size, seq_len, input_dim)
    x_recon, mu, logvar = model(x_seq)
    loss, recon_loss, kl_loss = model.loss_function(x_seq, x_recon, mu, logvar)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[Epoch {epoch+1:02d}] Total: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")

