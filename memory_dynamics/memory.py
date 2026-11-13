"""Dynamical memory over stored embedding patterns.

Implements the activation dynamics from "The Dynamics of Memory: Replay,
Diffusion, and Attractor Formation" (Ghosh, Jaiswal, Gupta - MetaCognition Labs):

    da/dt = -A a  -  gamma L a  -  delta (1^T a) 1  +  eta S(t) F(a)

integrated with forward Euler and clamped to a >= 0.

The maths here is identical to the class in notebooks/metacognition.ipynb that
produced the results in results/. The only change is that the source of
randomness for the replay gate S(t) can be injected (``rng``) so experiments
are reproducible without relying on NumPy's global state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DynamicalMemory:
    """Activation dynamics over a growing set of stored patterns.

    Args:
        gamma: diffusion strength (gamma) along the semantic similarity graph.
        delta: global inhibition strength (delta).
        beta: replay sharpness (beta), the inverse temperature of the softmax.
        eta: replay strength (eta).
        h: Euler step size.
        rng: anything with a ``random()`` method, used to sample the replay gate
            S(t). Defaults to NumPy's global legacy generator, which is what the
            notebook used.
    """

    def __init__(self, gamma=0.2, delta=0.1, beta=3.0, eta=0.8, h=0.01, rng=None):
        self.embeddings = []     # X: stored patterns (list of d-dim vectors)
        self.activations = None  # a(t): one activation per stored pattern
        self.gamma = gamma
        self.delta = delta
        self.beta = beta
        self.eta = eta
        self.h = h
        self.decay_rates = []    # alpha_i, the diagonal of A
        self.history = []        # ||a(t)|| after every step
        self.rng = np.random if rng is None else rng

    def store(self, embedding, decay_rate=1.0):
        """Add a pattern with activation 1 and its own decay rate."""
        self.embeddings.append(embedding.clone())
        self.decay_rates.append(decay_rate)
        if self.activations is None:
            self.activations = torch.ones(len(self.embeddings))
        else:
            self.activations = torch.cat([self.activations, torch.ones(1)])

    def _decay_matrix(self):
        return torch.diag(torch.tensor(self.decay_rates, dtype=torch.float32))

    def _build_laplacian(self):
        """Graph Laplacian L = D - W with W_ij = max(0, cos(x_i, x_j)), W_ii = 0."""
        X = torch.stack(self.embeddings)  # N x d
        X_norm = F.normalize(X, dim=1)
        W = torch.clamp(X_norm @ X_norm.T, min=0)
        W.fill_diagonal_(0)
        D = torch.diag(W.sum(dim=1))
        return D - W

    def _replay_operator(self):
        """F(a) = softmax(beta * X X^T a), with X the N x d matrix of raw embeddings."""
        X = torch.stack(self.embeddings)
        s = X @ (X.T @ self.activations)
        return F.softmax(self.beta * s, dim=0)

    def step(self, replay_on=False, replay_prob=0.3):
        """Advance one Euler step. S(t) is 1 with probability ``replay_prob``."""
        if len(self.embeddings) < 2:
            return

        a = self.activations
        N = len(a)

        decay = -self._decay_matrix() @ a                          # -A a
        diffusion = -self.gamma * self._build_laplacian() @ a      # -gamma L a
        inhibition = -self.delta * a.sum() * torch.ones(N)         # -delta (1^T a) 1

        S = 1.0 if (replay_on or self.rng.random() < replay_prob) else 0.0
        replay = self.eta * S * self._replay_operator()            # eta S(t) F(a)

        da_dt = decay + diffusion + inhibition + replay
        self.activations = torch.clamp(a + self.h * da_dt, min=0)

        self.history.append(self.activations.norm().item())

    def retrieve(self, query_embedding, top_k=3):
        """Rank stored patterns by cos(query, x_i) * a_i."""
        X = torch.stack(self.embeddings)
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)
        X_norm = F.normalize(X, dim=1)
        similarities = (query_norm @ X_norm.T).squeeze()
        scores = similarities * self.activations
        top_indices = scores.topk(min(top_k, len(scores))).indices
        return top_indices, scores[top_indices]

    def energy(self):
        """E(a) = 1/2 a^T A a + gamma/2 a^T L a + delta/2 (1^T a)^2.

        The first three terms of da/dt are exactly -grad E, so E is the potential
        of the replay-free system. Replay is not part of E.
        """
        a = self.activations
        A = self._decay_matrix()
        L = self._build_laplacian()
        e1 = 0.5 * a @ A @ a
        e2 = 0.5 * self.gamma * a @ L @ a
        e3 = 0.5 * self.delta * (a.sum()) ** 2
        return (e1 + e2 + e3).item()
