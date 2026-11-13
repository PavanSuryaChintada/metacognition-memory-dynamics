"""Figures in the same layout as the notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_demo(history, energies, labels, activations, path):
    """Activation norm, energy (log scale) and final per-memory activations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history, color="royalblue")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("||a(t)||")
    axes[0].set_title("Memory Activation Trajectory")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(energies, color="crimson")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("E(a)")
    axes[1].set_title("Energy Descent")
    axes[1].set_yscale("log")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].bar(labels, activations, color="darkgreen")
    axes[2].set_ylabel("Activation a_i")
    axes[2].set_title("Final Memory Activations")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_regimes(runs, path):
    """One panel per regime: ||a(t)|| over time.

    ``runs`` is a list of dicts with keys name, eta, replay_prob, history.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4))
    for ax, run in zip(axes, runs):
        ax.plot(run["history"], color="dodgerblue", linewidth=1.5)
        ax.set_title(f"{run['name']} (̷={run['eta']}, p={run['replay_prob']})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("||a(t)||")
        ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
