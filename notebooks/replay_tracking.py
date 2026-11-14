# Second Colab run: the notebook's single run, repeated with the softmax replay
# weights F(a), the pre-softmax scores and every force logged at each step.
#
# Run in Colab as written below. It produced:
#   results/replay_tracking_output.txt   (printed output, word for word)
#   results/figures/softmax_results.png  (the six-panel figure)
#   results/replay_tracking.json         (the printed numbers, structured)
#
# Same URLs, seed, decay rates and parameters as the notebook. The labels here are
# the ones used in this run: "labrador" is the beagle photo and "cute_dog" is the
# French bulldog. Labels are only used for printing.
#
# The code below is unchanged from the run except that its first line had lost
# its leading "#".

# ============================================================
# TRUE RUN 1 REPRODUCTION WITH F(a) TRACKING
# Uses YOUR original code's exact setup
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoProcessor, AutoModel
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ============================================
# 1. Load I-JEPA — YOUR original way
# ============================================
model_id = "facebook/ijepa_vith14_1k"
print("Loading I-JEPA processor and model...")
processor = AutoProcessor.from_pretrained(model_id)
encoder = AutoModel.from_pretrained(model_id).eval()

@torch.no_grad()
def encode_image(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, stream=True, timeout=15)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = processor(img, return_tensors="pt")
    outputs = encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()  # FULL 1280-d, mean pooled

# ============================================
# 2. YOUR exact URLs, YOUR exact labels
# ============================================
urls = [
    "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400",
    "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
    "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400",
    "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=400"
]
labels = ["labrador", "cat", "person", "cute_dog"]  # corrected: pug→person

# ============================================
# 3. DynamicalMemory with tracking — YOUR exact math
# ============================================
class DynamicalMemoryTrackedV2:
    def __init__(self, gamma=0.2, delta=0.1, beta=3.0, eta=0.8, h=0.01):
        self.embeddings = []
        self.activations = None
        self.gamma = gamma
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.h = h
        self.decay_rates = []

    def store(self, embedding, decay_rate=1.0):
        self.embeddings.append(embedding.clone())
        self.decay_rates.append(decay_rate)
        N = len(self.embeddings)
        if self.activations is None:
            self.activations = torch.ones(N)        # YOUR init: 1.0
        else:
            self.activations = torch.cat([self.activations, torch.ones(1)])

    def _build_laplacian(self):
        X = torch.stack(self.embeddings)
        X_norm = F.normalize(X, dim=1)
        W = torch.clamp(X_norm @ X_norm.T, min=0)
        W.fill_diagonal_(0)                          # YOUR code: no self-loops
        D = torch.diag(W.sum(dim=1))
        return D - W, W

    def _replay_operator(self):
        X = torch.stack(self.embeddings)
        s = X @ (X.T @ self.activations)
        return F.softmax(self.beta * s, dim=0), s    # return both

    def run_tracked(self, steps=500, replay_prob=0.3):
        per_step_activations = [self.activations.clone().numpy()]
        replay_weights = []
        replay_raw_scores = []
        replay_steps = []
        energies = []

        decay_forces = []
        diffusion_forces = []
        inhibition_forces = []
        replay_forces_list = []

        for t in range(steps):
            a = self.activations
            N = len(a)

            # Decay
            A = torch.diag(torch.tensor(self.decay_rates, dtype=torch.float32))
            f_decay = -A @ a

            # Diffusion
            L, W_mat = self._build_laplacian()
            f_diffusion = -self.gamma * L @ a

            # Inhibition
            f_inhibition = -self.delta * a.sum() * torch.ones(N)

            # Replay
            S = 1.0 if np.random.random() < replay_prob else 0.0
            Fa, raw_s = self._replay_operator()
            f_replay = self.eta * S * Fa

            if S > 0:
                replay_weights.append(Fa.detach().numpy().copy())
                replay_raw_scores.append(raw_s.detach().numpy().copy())
                replay_steps.append(t)

            # Update
            da_dt = f_decay + f_diffusion + f_inhibition + f_replay
            self.activations = a + self.h * da_dt
            self.activations = torch.clamp(self.activations, min=0)

            per_step_activations.append(self.activations.clone().numpy())

            decay_forces.append(f_decay.detach().numpy().copy())
            diffusion_forces.append(f_diffusion.detach().numpy().copy())
            inhibition_forces.append(f_inhibition.detach().numpy().copy())
            replay_forces_list.append((self.eta * S * Fa).detach().numpy().copy())

            # Energy
            e1 = 0.5 * a @ A @ a
            e2 = 0.5 * self.gamma * a @ L @ a
            e3 = 0.5 * self.delta * (a.sum())**2
            energies.append((e1 + e2 + e3).item())

        # Similarity matrix (for printing)
        X = torch.stack(self.embeddings)
        X_norm = F.normalize(X, dim=1)
        sim = (X_norm @ X_norm.T).numpy()

        return {
            'activations': np.array(per_step_activations),
            'replay_weights': np.array(replay_weights),
            'replay_raw_scores': np.array(replay_raw_scores),
            'replay_steps': np.array(replay_steps),
            'energy': np.array(energies),
            'decay_forces': np.array(decay_forces),
            'diffusion_forces': np.array(diffusion_forces),
            'inhibition_forces': np.array(inhibition_forces),
            'replay_forces': np.array(replay_forces_list),
            'labels': labels,
            'decays': self.decay_rates,
            'similarity': sim,
        }


# ============================================
# 4. Store embeddings — YOUR exact way
# ============================================
torch.manual_seed(42)
np.random.seed(42)

mem = DynamicalMemoryTrackedV2(gamma=0.2, delta=0.1, beta=3.0, eta=0.8)

print("\nEncoding images with I-JEPA (full 1280-d, mean pooled)...")
stored_count = 0
for url, label in zip(urls, labels):
    try:
        emb = encode_image(url)
        decay = np.random.uniform(0.5, 1.5)
        mem.store(emb, decay_rate=decay)
        print(f"  Stored: {label} (decay={decay:.2f}, dim={emb.shape[0]})")
        stored_count += 1
    except Exception as e:
        print(f"  FAILED: {label} — {e}")
        fallback_emb = torch.randn(1280)
        decay = np.random.uniform(0.5, 1.5)
        mem.store(fallback_emb, decay_rate=decay)
        print(f"  Used fallback for {label} (decay={decay:.2f})")
        stored_count += 1

# ============================================
# 5. Run with tracking — 500 steps like YOUR original
# ============================================
print(f"\nRunning simulation: η={mem.eta}, β={mem.beta}, p=0.3, steps=500")
print(f"Embedding dim: {mem.embeddings[0].shape[0]}")
results = mem.run_tracked(steps=500, replay_prob=0.3)
print(f"Replay fired {len(results['replay_steps'])} times out of 500 steps")

# ============================================
# 6. PRINT ALL RAW DATA
# ============================================
print("\n" + "="*60)
print("SIMILARITY MATRIX (cosine similarity, full 1280-d)")
print("="*60)
sim = results['similarity']
print(f"{'':>14}", end="")
for l in labels: print(f"{l:>14}", end="")
print()
for i, l in enumerate(labels):
    print(f"{l:>14}", end="")
    for j in range(len(labels)):
        print(f"{sim[i][j]:>14.4f}", end="")
    print()

print("\n" + "="*60)
print("FINAL ACTIVATIONS (step 500)")
print("="*60)
final = results['activations'][-1]
for i, l in enumerate(labels):
    print(f"  {l:>14}: {final[i]:.6f}  (α={results['decays'][i]:.2f})")

winner = labels[np.argmax(final)]
print(f"\n  Winner: {winner} ({final.max():.6f})")

print("\n" + "="*60)
print("REPLAY WEIGHT ANALYSIS")
print("="*60)
rw = results['replay_weights']
rs = results['replay_steps']
if len(rw) > 0:
    early = rw[:max(1, len(rw)//10)]
    late = rw[-max(1, len(rw)//10):]

    print(f"\nTotal replay events: {len(rw)}")
    print("\nEarly F(a) (first 10%):")
    for i, l in enumerate(labels):
        print(f"  {l:>14}: mean={early[:,i].mean():.4f}")
    print("\nLate F(a) (last 10%):")
    for i, l in enumerate(labels):
        print(f"  {l:>14}: mean={late[:,i].mean():.4f}")

print("\n" + "="*60)
print("MEAN FORCE BREAKDOWN")
print("="*60)
print(f"{'Memory':>14} {'Decay':>10} {'Diffusion':>10} {'Inhibit':>10} {'Replay':>10} {'Net':>10}")
for i, l in enumerate(labels):
    d = results['decay_forces'][:, i].mean()
    di = results['diffusion_forces'][:, i].mean()
    inh = results['inhibition_forces'][:, i].mean()
    rep = results['replay_forces'][:, i].mean()
    print(f"{l:>14} {d:>10.5f} {di:>10.5f} {inh:>10.5f} {rep:>10.5f} {d+di+inh+rep:>10.5f}")

# ============================================
# 7. PLOTS
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Activations per memory
ax = axes[0][0]
for i, l in enumerate(labels):
    ax.plot(results['activations'][:, i], label=f"{l} (α={results['decays'][i]:.2f})")
ax.set_xlabel("Timestep"); ax.set_ylabel("Activation a_i(t)")
ax.set_title("Per-Memory Activation"); ax.legend(); ax.grid(True, alpha=0.3)

# Replay weights
ax = axes[0][1]
for i, l in enumerate(labels):
    ax.plot(rs, rw[:, i], label=l, alpha=0.7)
ax.set_xlabel("Timestep"); ax.set_ylabel("F(a)_i")
ax.set_title("Softmax Replay Weights"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Stacked area
ax = axes[0][2]
step = max(1, len(rs) // 300)
ax.stackplot(rs[::step], [rw[::step, i] for i in range(len(labels))],
             labels=labels, alpha=0.8)
ax.set_xlabel("Timestep"); ax.set_ylabel("Share")
ax.set_title("Replay Distribution"); ax.legend(loc='center left'); ax.grid(True, alpha=0.3)

# Force breakdown
ax = axes[1][0]
x_pos = np.arange(len(labels))
w = 0.2
ax.bar(x_pos-1.5*w, [results['decay_forces'][:,i].mean() for i in range(4)], w, label='Decay', color='red', alpha=0.7)
ax.bar(x_pos-0.5*w, [results['diffusion_forces'][:,i].mean() for i in range(4)], w, label='Diffusion', color='blue', alpha=0.7)
ax.bar(x_pos+0.5*w, [results['inhibition_forces'][:,i].mean() for i in range(4)], w, label='Inhibition', color='orange', alpha=0.7)
ax.bar(x_pos+1.5*w, [results['replay_forces'][:,i].mean() for i in range(4)], w, label='Replay', color='green', alpha=0.7)
ax.set_xticks(x_pos); ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Mean Force"); ax.set_title("Force Breakdown"); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

# Energy
ax = axes[1][1]
ax.plot(results['energy'], color='red', alpha=0.7)
ax.set_xlabel("Timestep"); ax.set_ylabel("E(a)"); ax.set_title("Energy"); ax.grid(True, alpha=0.3)

# Pre-softmax scores
ax = axes[1][2]
raw = results['replay_raw_scores']
for i, l in enumerate(labels):
    ax.plot(rs, raw[:, i], label=l, alpha=0.7)
ax.set_xlabel("Timestep"); ax.set_ylabel("Raw Score")
ax.set_title("Pre-Softmax Scores (X @ Xᵀ @ a)"); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle("TRUE Run 1 Reproduction (Unsplash URLs, 1280-d, mean-pooled, 500 steps)\n"
             "η=0.8, β=3.0, p=0.3, seed=42, no self-loops",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("true_run1_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
