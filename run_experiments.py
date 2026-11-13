"""Run the notebook's experiments end to end and write figures + metrics.

Stages run in the same order, and on the same random stream, as the notebook
cells that produced results/ (cell 2 -> cell 3 -> cell 5):

  1. Encode four images with I-JEPA; store each as a memory with a random
     decay rate drawn from U(0.5, 1.5).
  2. Single run: 500 Euler steps at eta=0.8, replay probability 0.3, then
     query the memory with the first image.
  3. Three-regime sweep A (cell 3 settings).
  4. Three-regime sweep B (cell 5 settings).

One difference from the notebook: cell 5 plotted one set of runs, then re-ran
every regime with fresh randomness to print the final numbers. Here each regime
runs once, and the figure and the metrics come from the same run.

Usage:
    python run_experiments.py                      # downloads the I-JEPA model
    python run_experiments.py --embeddings outputs/embeddings.pt   # skip the model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from memory_dynamics import DynamicalMemory  # noqa: E402
from memory_dynamics.plotting import plot_demo, plot_regimes  # noqa: E402

IMAGES = [
    ("https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400", "golden_retriever"),
    ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400", "cat"),
    ("https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400", "pug"),
    ("https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=400", "cute_dog"),
]

BASE_PARAMS = dict(gamma=0.2, delta=0.1, beta=3.0, h=0.01)
DEMO_ETA = 0.8
DEMO_REPLAY_PROB = 0.3

SWEEP_A = [  # notebook cell 3
    {"name": "Collapse", "eta": 0.1, "replay_prob": 0.1},
    {"name": "Balanced", "eta": 2.0, "replay_prob": 0.5},
    {"name": "Runaway", "eta": 8.0, "replay_prob": 0.9},
]
SWEEP_B = [  # notebook cell 5
    {"name": "Collapse", "eta": 0.05, "replay_prob": 0.05},
    {"name": "Balanced", "eta": 2.0, "replay_prob": 0.4},
    {"name": "Runaway", "eta": 10.0, "replay_prob": 0.9},
]


def encode_images(images):
    from memory_dynamics.encoder import IJepaEncoder

    print("Loading I-JEPA processor and model...")
    encoder = IJepaEncoder()
    urls, labels, embeddings = [], [], []
    for url, label in images:
        try:
            embeddings.append(encoder.encode_url(url))
            urls.append(url)
            labels.append(label)
            print(f"  Encoded: {label}")
        except Exception as e:  # keep going, as the notebook did
            print(f"  Skipped {label}: {e}")
    return urls, labels, embeddings


def run_regimes(embeddings, decays, configs, rng, steps):
    runs = []
    for cfg in configs:
        mem = DynamicalMemory(eta=cfg["eta"], rng=rng, **BASE_PARAMS)
        for emb, dr in zip(embeddings, decays):
            mem.store(emb, decay_rate=dr)
        for _ in range(steps):
            mem.step(replay_prob=cfg["replay_prob"])
        runs.append({**cfg, "mem": mem, "history": mem.history})
    return runs


def regime_metrics(runs, labels):
    return [
        {
            "name": r["name"],
            "eta": r["eta"],
            "replay_prob": r["replay_prob"],
            "final_norm": r["mem"].activations.norm().item(),
            "final_energy": r["mem"].energy(),
            "final_activations": dict(zip(labels, r["mem"].activations.tolist())),
        }
        for r in runs
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="load embeddings saved by a previous run instead of running I-JEPA")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # One generator for every random draw, in notebook order. RandomState(seed)
    # yields the same stream as np.random.seed(seed) did in the notebook.
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # 1. Memories
    if args.embeddings:
        saved = torch.load(args.embeddings)
        urls, labels, embeddings = saved["urls"], saved["labels"], list(saved["embeddings"])
        print(f"Loaded {len(embeddings)} embeddings from {args.embeddings}")
    else:
        urls, labels, embeddings = encode_images(IMAGES)
        torch.save({"urls": urls, "labels": labels, "embeddings": torch.stack(embeddings)},
                   args.out / "embeddings.pt")

    if len(embeddings) < 2:
        raise SystemExit("Need at least two stored memories to run the dynamics.")

    decays = [float(rng.uniform(0.5, 1.5)) for _ in embeddings]
    for label, d in zip(labels, decays):
        print(f"  Stored: {label} (decay={d:.2f})")

    # 2. Single run + retrieval
    print(f"\nRunning memory dynamics ({args.steps} steps)...")
    mem = DynamicalMemory(eta=DEMO_ETA, rng=rng, **BASE_PARAMS)
    for emb, d in zip(embeddings, decays):
        mem.store(emb, decay_rate=d)
    energies = []
    for _ in range(args.steps):
        mem.step(replay_prob=DEMO_REPLAY_PROB)
        energies.append(mem.energy())

    final = mem.activations.tolist()
    plot_demo(mem.history, energies, labels, final, args.out / "dynamics_demo.png")
    print(f"Final activations: {np.array(final)}")
    print(f"Final energy: {energies[-1]:.4f}")

    retrieval = None
    if urls[0] == IMAGES[0][0]:
        indices, scores = mem.retrieve(embeddings[0])
        retrieval = {
            "query": labels[0],
            "top": [{"label": labels[i], "score": s.item()} for i, s in zip(indices, scores)],
        }
        print(f"\nQuerying with the {labels[0]} image...")
        for rank, hit in enumerate(retrieval["top"], 1):
            print(f"  {rank}. {hit['label']} (score={hit['score']:.4f})")

    # 3-4. Regimes
    runs_a = run_regimes(embeddings, decays, SWEEP_A, rng, args.steps)
    plot_regimes(runs_a, args.out / "three_regimes_sweep_a.png")
    runs_b = run_regimes(embeddings, decays, SWEEP_B, rng, args.steps)
    plot_regimes(runs_b, args.out / "three_regimes_sweep_b.png")

    metrics = {
        "seed": args.seed,
        "steps": args.steps,
        "base_params": BASE_PARAMS,
        "memories": [{"label": l, "url": u, "decay_rate": d}
                     for l, u, d in zip(labels, urls, decays)],
        "single_run": {
            "eta": DEMO_ETA,
            "replay_prob": DEMO_REPLAY_PROB,
            "final_activations": dict(zip(labels, final)),
            "final_norm": mem.activations.norm().item(),
            "final_energy": energies[-1],
            "retrieval": retrieval,
        },
        "regimes": {
            "sweep_a": regime_metrics(runs_a, labels),
            "sweep_b": regime_metrics(runs_b, labels),
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print()
    for sweep, runs in metrics["regimes"].items():
        for r in runs:
            print(f"{sweep} {r['name']}: final ||a|| = {r['final_norm']:.4f}, "
                  f"final energy = {r['final_energy']:.6f}")
    print(f"\nWrote figures and metrics.json to {args.out}/")


if __name__ == "__main__":
    main()
