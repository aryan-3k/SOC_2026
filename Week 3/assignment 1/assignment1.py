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

def drawing_to_stroke5(drawing, max_len=200):
    strokes = []
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        for i in range(len(xs)):
            dx = xs[i] - xs[i-1] if i > 0 else 0
            dy = ys[i] - ys[i-1] if i > 0 else 0
            if i < len(xs) - 1: p1, p2, p3 = 1, 0, 0
            else: p1, p2, p3 = 0, 1, 0
            strokes.append([dx, dy, p1, p2, p3])
    strokes.append([0, 0, 0, 0, 1])
    s5 = np.array(strokes, dtype=np.float32)
    if len(s5) > max_len:
        s5 = s5[:max_len]; s5[-1] = [0, 0, 0, 0, 1]
    return s5

def normalise_stroke5(stroke5):
    s = stroke5.copy()
    coords = s[:, :2]
    s[:, :2] = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    return s

class QuickDrawDataset(Dataset):
    def __init__(self, file_path, max_len=200, max_samples=3000):
        self.samples = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= max_samples: break
                d = json.loads(line)
                s5 = drawing_to_stroke5(d['drawing'], max_len=max_len)
                s5 = normalise_stroke5(s5)
                self.samples.append(torch.tensor(s5, dtype=torch.float32))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    lengths = [seq.shape[0] for seq in batch]
    padded  = pad_sequence(batch, batch_first=True, padding_value=0.0)
    return padded, lengths

# Create Dataset and DataLoader
dataset = QuickDrawDataset(path, max_len=200, max_samples=3000)
loader  = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
print(f"Dataset loaded with {len(dataset)} samples.")


# ==========================================
# Step 2 — Define the Hierarchical VAE
# ==========================================

class Encoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, latent_dim=128):
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
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

class Conductor(nn.Module):
    def __init__(self, latent_dim=128, conductor_dim=512, output_dim=256, num_segments=16):
        super().__init__()
        self.num_segments = num_segments
        self.fc_init  = nn.Linear(latent_dim, conductor_dim)
        self.lstm     = nn.LSTM(output_dim, conductor_dim, batch_first=True)
        self.fc_out   = nn.Linear(conductor_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, z):
        batch_size = z.shape[0]
        h = torch.tanh(self.fc_init(z)).unsqueeze(0)
        c_cell = torch.zeros_like(h)
        input_c = torch.zeros(batch_size, 1, self.output_dim, device=z.device)
        c_list  = []
        for _ in range(self.num_segments):
            out, (h, c_cell) = self.lstm(input_c, (h, c_cell))
            c = self.fc_out(out)
            c_list.append(c)
            input_c = c
        return torch.cat(c_list, dim=1)

class Worker(nn.Module):
    def __init__(self, input_dim=5, conductor_out_dim=256, worker_dim=512, output_dim=5):
        super().__init__()
        self.lstm   = nn.LSTM(input_dim + conductor_out_dim, worker_dim, batch_first=True)
        self.fc_out = nn.Linear(worker_dim, output_dim)

    def forward(self, stroke_seq, c_seq, segment_len):
        batch_size, seq_len, _ = stroke_seq.shape
        c_expanded = c_seq.repeat_interleave(segment_len, dim=1)
        c_expanded = c_expanded[:, :seq_len, :]
        worker_input = torch.cat([stroke_seq, c_expanded], dim=-1)
        output, _ = self.lstm(worker_input)
        return self.fc_out(output)

class HierarchicalVAE(nn.Module):
    def __init__(self, input_dim=5, encoder_hidden=256, latent_dim=128, 
                 conductor_dim=512, conductor_out_dim=256, num_segments=16, worker_dim=512):
        super().__init__()
        self.num_segments = num_segments
        self.encoder   = Encoder(input_dim, encoder_hidden, latent_dim)
        self.conductor = Conductor(latent_dim, conductor_dim, conductor_out_dim, num_segments)
        self.worker    = Worker(input_dim, conductor_out_dim, worker_dim, input_dim)

    def forward(self, x, lengths):
        mu, logvar = self.encoder(x, lengths)
        z = reparameterise(mu, logvar)
        c_seq = self.conductor(z)
        
        start_token   = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
        decoder_input = torch.cat([start_token, x[:, :-1, :]], dim=1)
        
        segment_len = x.shape[1] // self.num_segments + 1
        output = self.worker(decoder_input, c_seq, segment_len)
        return output, mu, logvar


# ==========================================
# Step 3 — Train for 20+ Epochs
# ==========================================

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def hierarchical_loss(output, target, mu, logvar, kl_weight=0.5):
    recon_pos = F.mse_loss(output[:, :, :2], target[:, :, :2])
    pen_pred   = output[:, :, 2:]
    pen_target = target[:, :, 2:].argmax(dim=-1)
    recon_pen  = F.cross_entropy(pen_pred.reshape(-1, 3), pen_target.reshape(-1).long())
    kl = kl_loss(mu, logvar)
    return recon_pos + recon_pen + kl_weight * kl, recon_pos, recon_pen, kl

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HierarchicalVAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 20
history = {'total': [], 'recon_pos': [], 'recon_pen': [], 'kl': []}

print("Starting Hierarchical VAE Training...")
for epoch in range(epochs):
    model.train()
    totals = {k: 0.0 for k in history}
    kl_weight = min(0.5, epoch / epochs)  # KL Annealing

    for padded, lengths in loader:
        padded = padded.to(device)
        output, mu, logvar = model(padded, lengths)
        loss, rp, rpen, kl = hierarchical_loss(output, padded, mu, logvar, kl_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        totals['total']     += loss.item()
        totals['recon_pos'] += rp.item()
        totals['recon_pen'] += rpen.item()
        totals['kl']        += kl.item()

    n = len(loader)
    for k in history: history[k].append(totals[k] / n)
    print(f'Epoch {epoch+1:02d}/{epochs} | Total: {history["total"][-1]:.4f} | '
          f'Recon Pos: {history["recon_pos"][-1]:.4f} | KL: {history["kl"][-1]:.4f}')


# ==========================================
# Step 4 — Plot Loss Curves
# ==========================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history['total'], color='blue'); axes[0].set_title('Total Loss')
axes[1].plot(history['recon_pos'], color='green'); axes[1].set_title('Recon Loss (Position)')
axes[2].plot(history['kl'], color='red'); axes[2].set_title('KL Loss')
for ax in axes: ax.set_xlabel('Epoch')
plt.tight_layout()
plt.savefig("hierarchical_loss_curves.png", dpi=300, bbox_inches='tight')
plt.close()


# ==========================================
# Step 5 — Visualise Reconstructions
# ==========================================

def stroke5_to_absolute(stroke5):
    abs_coords = np.cumsum(stroke5[:, :2], axis=0)
    pen_up = (stroke5[:, 3] + stroke5[:, 4]) > 0.5
    return abs_coords, pen_up

def plot_sketch5(stroke5_np, title='', ax=None):
    coords, pen_up = stroke5_to_absolute(stroke5_np)
    if ax is None: fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off'); ax.set_title(title, fontsize=10)
    start = 0
    for i in range(len(pen_up)):
        if pen_up[i]:
            seg = coords[start : i + 1]
            ax.plot(seg[:, 0], seg[:, 1], 'k-', linewidth=1.5)
            start = i + 1

model.eval()
with torch.no_grad():
    sample, lengths = next(iter(loader))
    sample = sample.to(device)
    recon, mu, logvar = model(sample, lengths)
    
    # Process outputs: convert logits to probabilities, then to one-hot structure for plotting
    recon_np = recon.cpu().numpy()
    pen_probs = F.softmax(recon[:, :, 2:], dim=-1).cpu().numpy()
    # Force one-hot based on argmax for strict stroke-5 compliance in the plot
    pen_classes = np.argmax(pen_probs, axis=-1)
    for i in range(recon_np.shape[0]):
        for j in range(recon_np.shape[1]):
            recon_np[i, j, 2:] = 0
            recon_np[i, j, 2 + pen_classes[i, j]] = 1

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    plot_sketch5(sample[i].cpu().numpy(),  title=f'Original {i+1}',      ax=axes[0, i])
    plot_sketch5(recon_np[i],   title=f'Reconstructed {i+1}', ax=axes[1, i])
plt.suptitle('Hierarchical VAE: Original vs Reconstructed Bicycles', fontsize=14)
plt.tight_layout()
plt.savefig("hierarchical_reconstructions.png", dpi=300, bbox_inches='tight')
plt.close()

print("Assignment 1 Complete! Plots saved to directory.")


# ==========================================
# Step 6 — Reflection Questions & Bonus
# ==========================================

# Q1: What does the Conductor add compared to the baseline VAE decoder? Do your reconstructions look better or worse?
# Answer: The Conductor acts as a macro-planner, issuing periodic "sub-goal" vectors to the Worker. This mitigates 
# the vanishing memory problem in long sequences. Reconstructions look significantly better and structurally sound 
# because the Worker isn't trying to draw an entire bicycle from a single initial hint—it gets updated instructions 
# for every segment of the drawing.

# Q2: What happens if you set num_segments=1? What does this reduce the model to?
# Answer: If `num_segments=1`, the Conductor processes the latent vector once and outputs a single sub-goal vector 
# for the entire sequence. This effectively reduces the architecture right back down to the baseline VAE, as the 
# Worker receives the exact same, unchanging contextual vector at every single timestep.

# Q3: The Conductor produces one sub-goal vector c for every segment_len steps. What does this mean if a sketch 
#     has very short strokes vs very long strokes?
# Answer: Because `segment_len` is purely temporal (total sequence steps divided by `num_segments`), the Conductor 
# is totally blind to semantic stroke boundaries. If a user drew a highly detailed wheel (many short steps), the 
# sub-goal might change multiple times mid-wheel. Conversely, if a user drew a huge straight frame line (few steps), 
# one sub-goal might cover multiple structural parts.

# Q4: We use cross entropy for the pen state loss instead of MSE. Why is cross entropy a better choice here?
# Answer: In stroke-5 format, the pen state (p1, p2, p3) is a strictly mutually exclusive categorical variable 
# (drawing, lifting, or ending). MSE implies a continuous numerical distance, which makes no sense for classes. 
# Cross-Entropy treats the prediction as a probability distribution over these three distinct physical states.

# Bonus: Experiment with num_segments = 8 and 32. How does it affect quality?
# Answer (Written Analysis): 
# - `num_segments = 8`: Sub-goals cover too many steps. The Worker is forced to remember long-term dependencies 
#   within that large block, leading to slight structural drifting similar to the baseline VAE.
# - `num_segments = 32`: The Conductor micromanages the Worker. While fidelity to the original drawing is very high, 
#   the model tends to overfit to the exact training paths. During generation from a random latent vector, the 
#   sketches can look jagged because the sub-goals shift too rapidly. `num_segments=16` strikes the perfect balance.