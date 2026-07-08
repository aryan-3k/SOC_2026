import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevents OpenMP duplicate crash

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import urllib.request
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.decomposition import PCA

# ==========================================
# Step 1 — Load your category ('bicycle')
# ==========================================

category = 'bicycle'
os.makedirs('data', exist_ok=True)
path = f'data/{category}.ndjson'

if not os.path.exists(path):
    print(f"Downloading {category} dataset...")
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson'
    urllib.request.urlretrieve(url, path)
    print("Download complete.")

def drawing_to_stroke3(drawing):
    strokes = []
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        for i in range(len(xs)):
            dx = xs[i] - xs[i-1] if i > 0 else 0
            dy = ys[i] - ys[i-1] if i > 0 else 0
            pen_lifted = 1 if i == len(xs) - 1 else 0
            strokes.append([dx, dy, pen_lifted])
    return np.array(strokes, dtype=np.float32)

def normalise_stroke3(stroke3):
    s = stroke3.copy()
    coords = s[:, :2]
    std = coords.std(axis=0) + 1e-8
    s[:, :2] = (coords - coords.mean(axis=0)) / std
    return s

class QuickDrawDataset(Dataset):
    def __init__(self, file_path, max_len=200, max_samples=5000):
        self.samples = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= max_samples: break
                d  = json.loads(line)
                s3 = drawing_to_stroke3(d['drawing'])
                s3 = normalise_stroke3(s3)
                if len(s3) > max_len: 
                    s3 = s3[:max_len]
                self.samples.append(torch.tensor(s3, dtype=torch.float32))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    lengths = [seq.shape[0] for seq in batch]
    padded  = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return padded, lengths

# Create Dataset and DataLoader
dataset = QuickDrawDataset(path, max_len=200, max_samples=3000)
loader  = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
print(f"Dataset loaded with {len(dataset)} samples.")

# ==========================================
# Step 2 — Define and Train the VAE
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.lstm      = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu     = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[0], h[1]], dim=-1)
        return self.fc_mu(h), self.fc_logvar(h)

def reparameterise(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512, output_dim=3):
        super().__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_cell   = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, target_seq):
        batch_size = z.shape[0]
        h0 = torch.tanh(self.fc_hidden(z)).unsqueeze(0)
        c0 = torch.tanh(self.fc_cell(z)).unsqueeze(0)
        start_token = torch.zeros(batch_size, 1, 3, device=z.device)
        decoder_input = torch.cat([start_token, target_seq[:, :-1, :]], dim=1)
        output, _ = self.lstm(decoder_input, (h0, c0))
        return self.fc_out(output)

    def generate(self, z, max_len=200):
        batch_size = z.shape[0]
        h = torch.tanh(self.fc_hidden(z)).unsqueeze(0)
        c = torch.tanh(self.fc_cell(z)).unsqueeze(0)
        input_step = torch.zeros(batch_size, 1, 3, device=z.device)
        outputs = []
        for _ in range(max_len):
            out, (h, c) = self.lstm(input_step, (h, c))
            step = self.fc_out(out)
            outputs.append(step)
            input_step = step
        return torch.cat(outputs, dim=1)

class SketchVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(3, 256, 128)
        self.decoder = Decoder(128, 512, 3)

    def forward(self, x, lengths):
        mu, logvar = self.encoder(x, lengths)
        z          = reparameterise(mu, logvar)
        recon      = self.decoder(z, x)
        return recon, mu, logvar

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(recon, target, mu, logvar, kl_weight):
    recon_pos  = F.mse_loss(recon[:, :, :2], target[:, :, :2])
    pen_pred   = torch.sigmoid(recon[:, :, 2])
    recon_pen  = F.binary_cross_entropy(pen_pred, target[:, :, 2], reduction='mean')
    kl         = kl_loss(mu, logvar)
    return recon_pos + recon_pen + kl_weight * kl, recon_pos, kl

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SketchVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
history = {'total': [], 'recon': [], 'kl': []}

print("Starting VAE Training...")
for epoch in range(epochs):
    model.train()
    epoch_total, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
    kl_weight = min(0.5, epoch / epochs)

    for padded, lengths in loader:
        padded = padded.to(device)
        recon, mu, logvar = model(padded, lengths)
        loss, rp, kl = vae_loss(recon, padded, mu, logvar, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_total += loss.item()
        epoch_recon += rp.item()
        epoch_kl    += kl.item()

    n = len(loader)
    history['total'].append(epoch_total / n)
    history['recon'].append(epoch_recon / n)
    history['kl'].append(epoch_kl / n)
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Total: {history['total'][-1]:.4f} | Recon: {history['recon'][-1]:.4f} | KL: {history['kl'][-1]:.4f} | KLw: {kl_weight:.2f}")

# ==========================================
# Step 3 — Plot loss curves
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history['total'], color='blue'); axes[0].set_title('Total Loss')
axes[1].plot(history['recon'], color='green'); axes[1].set_title('Recon Loss (Position)')
axes[2].plot(history['kl'], color='red'); axes[2].set_title('KL Loss')
for ax in axes: ax.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Step 4 — Visualise reconstructions
# ==========================================

def stroke3_to_absolute(stroke3):
    abs_coords = np.cumsum(stroke3[:, :2], axis=0)
    pen_lifted  = stroke3[:, 2]
    return abs_coords, pen_lifted

def plot_sketch(stroke3_np, title='', ax=None):
    coords, pen_lifted = stroke3_to_absolute(stroke3_np)
    if ax is None: fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off'); ax.set_title(title, fontsize=10)
    start = 0
    for i in range(len(pen_lifted)):
        if pen_lifted[i] > 0.5:
            seg = coords[start : i + 1]
            ax.plot(seg[:, 0], seg[:, 1], 'k-', linewidth=1.5)
            start = i + 1

model.eval()
with torch.no_grad():
    sample_batch, sample_lengths = next(iter(loader))
    sample_batch = sample_batch.to(device)
    recon, _, _ = model(sample_batch, sample_lengths)
    recon_np = recon.cpu().numpy()
    recon_np[:, :, 2] = (recon_np[:, :, 2] > 0).astype(np.float32)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    plot_sketch(sample_batch[i].cpu().numpy(), title=f'Original {i+1}', ax=axes[0, i])
    plot_sketch(recon_np[i], title=f'Recon {i+1}', ax=axes[1, i])
plt.suptitle('Bicycle: Original vs Reconstructed', fontsize=14)
plt.tight_layout()
plt.savefig("reconstructions_grid.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Step 5 — Latent interpolation
# ==========================================

with torch.no_grad():
    s1 = dataset[0].unsqueeze(0).to(device)
    s2 = dataset[1].unsqueeze(0).to(device)
    mu1, _ = model.encoder(s1, [s1.shape[1]])
    mu2, _ = model.encoder(s2, [s2.shape[1]])

    steps = 6
    alphas = torch.linspace(0, 1, steps)
    z_interp = torch.stack([alpha * mu2 + (1 - alpha) * mu1 for alpha in alphas]).squeeze(1)
    generated = model.decoder.generate(z_interp, max_len=150).cpu().numpy()
    generated[:, :, 2] = (generated[:, :, 2] > 0).astype(np.float32)

fig, axes = plt.subplots(1, steps, figsize=(18, 3))
for i in range(steps):
    plot_sketch(generated[i], title=f'alpha={alphas[i]:.2f}', ax=axes[i])
plt.suptitle('Latent Interpolation (Bicycles)', fontsize=14)
plt.tight_layout()
plt.savefig("latent_interpolation.png", dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Bonus — PCA Visualization of Latent Space
# ==========================================

with torch.no_grad():
    mu_list = []
    for i in range(100):
        s = dataset[i].unsqueeze(0).to(device)
        mu, _ = model.encoder(s, [s.shape[1]])
        mu_list.append(mu.squeeze().cpu().numpy())
    
    mu_array = np.array(mu_list)
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(mu_array)

plt.figure(figsize=(6, 5))
plt.scatter(z_pca[:, 0], z_pca[:, 1], color='purple', alpha=0.6, edgecolors='k')
plt.title("PCA of 100 Bicycle Latent Vectors")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("bonus_pca_latent_space.png", dpi=300, bbox_inches='tight')
plt.close()

print("Assignment Complete! All plots saved to directory.")

# ==========================================
# Step 6 — Reflection questions (Answers)
# ==========================================

# Q1: What does the KL loss being very small early in training tell you? Why does KL annealing help?
# Answer: A very small KL loss early on indicates "posterior collapse," where the decoder ignores 
# the latent code z and acts as a standard autoregressive RNN. KL annealing (starting kl_weight at 0) 
# disables the KL penalty initially, forcing the model to learn to compress and use z to reconstruct 
# the drawing before gradually enforcing the standard normal prior.

# Q2: Which parts of the sketch does the model get right first? Which are hardest? Why?
# Answer: The model gets the global structure (the overarching wheels and frame of the bicycle) right first. 
# Fine-grained details (spokes, pedals, handlebars) are the hardest to reconstruct. This happens because 
# large macroscopic movements dominate the MSE position loss, whereas small intricacies require precise, 
# immediate sequence alignments that get smoothed out by the probabilistic latent space.

# Q3: Does your latent interpolation look smooth? What does smoothness in the latent space mean?
# Answer: Yes, the transitions appear smooth (e.g., wheels gradually morphing shape rather than disappearing). 
# Smoothness means the model has learned a continuous, dense manifold of "bicycle features", where 
# intermediate geometric coordinates correspond to logically valid semantic structures, rather than 
# disjointed, memorized data points.

# Q4: The decoder generates strokes autoregressively. What are the risks over long sequences?
# Answer: The primary risk is "exposure bias" or error accumulation. During generation, the model 
# feeds its own predicted outputs back into itself. A small prediction error early in the drawing 
# alters the trajectory, causing compounding deviations that make the final steps drift into noise.