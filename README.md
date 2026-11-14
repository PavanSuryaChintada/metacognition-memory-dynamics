# Dynamical Memory with JEPA Embeddings

Implementation of the dynamical memory framework from
"The Dynamics of Memory: Replay, Diffusion, and Attractor Formation"
(Ghosh, Jaiswal, Gupta — MetaCognition Labs) using real I-JEPA
embeddings from Hugging Face instead of synthetic Gaussians.

The same results are also written up as a single-page website in [`web/`](web/); see [Website](#website).

## What This Implements

The core memory evolution equation:

$$
\frac{da}{dt} = -Aa \;-\; \gamma L a \;-\; \delta\,(\mathbf{1}^\top a)\,\mathbf{1} \;+\; \eta\,S(t)\,F(a)
$$

where $a \in \mathbb{R}^N$ holds one activation per stored memory and every memory starts at $a_i(0) = 1$.

| Term | What it does | Code |
|---|---|---|
| $-Aa$ | **Decay.** $A = \mathrm{diag}(\alpha_i)$, one rate per memory, drawn from $U(0.5,\ 1.5)$. | `decay` |
| $-\gamma L a$ | **Diffusion** over the semantic graph. $L = D - W$ with $W_{ij} = \max(0, \cos(x_i, x_j))$ and $W_{ii} = 0$. | `_build_laplacian` |
| $-\delta(\mathbf{1}^\top a)\mathbf{1}$ | **Global inhibition.** Total activation suppresses every memory equally. | `inhibition` |
| $\eta\,S(t)\,F(a)$ | **Replay.** $S(t) \sim \mathrm{Bernoulli}(p)$ at every step. $F(a) = \mathrm{softmax}(\beta X X^\top a)$, with $X$ the $N \times d$ matrix of raw embeddings. | `_replay_operator` |

- **Integration:** forward Euler, $a \leftarrow \max(0,\ a + h\,\dot a)$, with $h = 0.01$. So 500 steps cover $t \in [0, 5]$.
- **Energy:** $E(a) = \tfrac12 a^\top A a + \tfrac{\gamma}{2} a^\top L a + \tfrac{\delta}{2}(\mathbf{1}^\top a)^2$.
  - The decay, diffusion and inhibition terms of $\dot a$ are exactly $-\nabla E$. Without replay, the system is a gradient flow on $E$.
  - Replay is an external drive and is not part of $E$.
- **Retrieval:** the score of memory $i$ is $\cos(q, x_i) \cdot a_i$, i.e. similarity weighted by current activation.
- **Embeddings:** each image goes through [`facebook/ijepa_vith14_1k`](https://huggingface.co/facebook/ijepa_vith14_1k) (I-JEPA, ViT-H/14). The last hidden state is mean-pooled over patches to a 1280-d vector.

## Setup

Four Unsplash images, each encoded once and stored as a memory. All runs share these settings: $\gamma = 0.2$, $\delta = 0.1$, $\beta = 3.0$, $h = 0.01$, 500 steps, seed 42.

| # | Notebook label | What the image shows | Decay rate $\alpha$ |
|---|---|---|---|
| 0 | `golden_retriever` | a beagle | 0.87 |
| 1 | `cat` | a cat | 1.45 |
| 2 | `pug` | **a person** (portrait photo, not a dog) | 1.23 |
| 3 | `cute_dog` | a French bulldog | 1.10 |

> **One label doesn't match its image.** In the notebook's `urls` and `labels` lists, entry 2 is labelled `pug`, but its URL (`photo-1517841905240-472988babdf9`) is a photo of a person. That memory is the one that survives, so the results are about a person photo, not a pug. The other three labels fit their images; `golden_retriever` is a beagle, which is only a breed detail.
>
> The figures in [`results/`](results/) are the notebook's original output, so they keep its labels. **Read `pug` in any figure as "person".** The code in this repo uses descriptive labels.

## Results

All numbers below come from the notebook run in [`notebooks/metacognition.ipynb`](notebooks/metacognition.ipynb):

- The printed output is in [`results/notebook_output.txt`](results/notebook_output.txt), copied verbatim.
- Every value is collected in [`results/metrics.json`](results/metrics.json).

### 1. Single run (η = 0.8, p = 0.3)

![Activation norm, energy and final activations for the single run](results/figures/dynamics_demo.png)

| Memory | Decay $\alpha$ | Final activation |
|---|---|---|
| beagle | 0.87 | 0 |
| cat | 1.45 | 0 |
| person (`pug`) | 1.23 | **0.1379** |
| French bulldog | 1.10 | 0 |

Final energy is **0.0150**.

- **One memory survives.** Three of the four activations reach exactly 0. The survivor is not the memory with the slowest decay: that was the beagle ($\alpha = 0.87$). Replay picked the winner, not decay.
- **Activation norm.** ‖a‖ falls from 2.0 and settles into a noisy plateau around 0.15 from step ~280 onward.
- **Energy.** $E$ starts at about 3.13 and falls roughly 200× to 0.015. From step ~280 it fluctuates around 0.015–0.02, where replay puts in about as much as decay removes.

### 2. Retrieval

The query is the beagle image, the same image as memory 0.

| Rank | Memory | Score |
|---|---|---|
| 1 | person (`pug`) | 0.0414 |
| 2 | French bulldog (`cute_dog`) | 0.0000 |
| 3 | beagle (`golden_retriever`) | 0.0000 |

**The exact-match memory ranks last.**

- The query is identical to the beagle memory (cosine similarity 1), but the beagle's activation decayed to 0.
- The score is cosine × activation, so once activations collapse, only the surviving memory can score above zero.
- Ranks 2 and 3 are tied at 0, so their order is arbitrary.
- The top score implies a cosine similarity between the beagle and person embeddings of about $0.0414 / 0.1379 \approx 0.30$.

### 3. Three dynamical regimes

The same four memories were run under three settings of replay strength $\eta$ and replay probability $p$. The notebook ran two sweeps of this.

#### Sweep B (final version, notebook cell 5)

![Collapse, balanced and runaway regimes, sweep B](results/figures/three_regimes_sweep_b.png)

| Regime | η | p | η·p | Final ‖a‖ | Final E |
|---|---|---|---|---|---|
| Collapse | 0.05 | 0.05 | 0.0025 | 0.0048 | 0.000014 |
| Balanced | 2.0 | 0.4 | 0.8 | 0.5003 | 0.197150 |
| Runaway | 10.0 | 0.9 | 9.0 | 5.7369 | 25.923445 |

- **Collapse.** Replay is too weak to matter. Activation decays smoothly to about 0, as it would with decay alone.
- **Balanced.** ‖a‖ drops over the first ~100 steps, then hovers between about 0.45 and 0.65. Replay and decay offset each other.
- **Runaway.** ‖a‖ climbs from 2.0 to about 5.5–6.1 and stays there. Final energy is about 8× the starting energy. Replay dominates.
  - "Runaway" here means *replay-dominated*, not divergent. $F(a)$ sums to 1, so replay adds at most η per unit time in total, while decay and inhibition grow with $a$. That caps ‖a‖, which is why the curve levels off instead of blowing up.

> The final numbers in the table do **not** come from the plotted runs. Cell 5 plotted the three regimes, then ran each one again with fresh random draws to print the final state. That is why the plotted Balanced curve ends near 0.47 while the table says 0.5003. [`run_experiments.py`](run_experiments.py) takes the figure and the numbers from the same run.

#### Sweep A (first version, notebook cell 3)

![Collapse, balanced and runaway regimes, sweep A](results/figures/three_regimes_sweep_a.png)

| Regime | η | p | η·p | Final ‖a‖ (read off the figure) |
|---|---|---|---|---|
| Collapse | 0.1 | 0.1 | 0.01 | ≈ 0.01 |
| Balanced | 2.0 | 0.5 | 1.0 | ≈ 0.6 (fluctuating 0.5–0.7) |
| Runaway | 8.0 | 0.9 | 7.2 | ≈ 4.7 |

This sweep printed no numbers. Both sweeps show the same three regimes. In both, the level where ‖a‖ settles rises with the replay drive η·p.

## Comparison with the paper

The paper's experiments (section 5.1) and this run share every parameter except the embeddings.

| | Paper, section 5 | This project |
|---|---|---|
| Embeddings | Random draws from a standard Gaussian | I-JEPA ViT-H/14 features of real photos, mean-pooled over patches |
| Dimension $d$ | 64 | 1,280 |
| Memories $N$ | 200 | 4 |
| Run length and seeds | 3,000 steps, averaged over 10 seeds | 500 steps, one seed (42) |
| Parameters | $\gamma = 0.2$, $\delta = 0.1$, $\beta = 3$, $h = 0.01$, $\alpha \sim U[0.5, 1.5]$, $S_t \sim \mathrm{Bernoulli}(p)$ | Same |
| Similarity graph | Cosine between independent random vectors: centred on 0, spread about $1/\sqrt{64} \approx 0.13$, so roughly half the edges are clipped to 0 | Cosine between image features; the run pins down one pair, beagle–person ≈ 0.30 |
| Regimes | Collapse, balanced and runaway (Figures 1 and 3) | The same three (Results §3) |
| Energy | Falls steadily with replay off (Figure 4) | With replay on, falls from about 3.13 to 0.015, then wobbles |

The paper names this gap itself, in section 6.6:

> "The experiments presented in this work focus on synthetic memory patterns. Applying the framework to realistic embeddings derived from language models or multimodal agents would provide a more direct evaluation of its practical relevance."

This run is a small first step at that: one image model, four photos, one seed.

**Boundedness check (paper section 3.2).**

- The proof shows a trajectory never leaves $\|a\| \le \max(\|a(0)\|,\ \eta/\lambda_{\min}(R))$, with $R = A + \gamma L + \delta\mathbf{1}\mathbf{1}^\top$.
- $L$ and $\mathbf{1}\mathbf{1}^\top$ are positive semidefinite, so $\lambda_{\min}(R) \ge \min_i \alpha_i = 0.875$. That gives a bound from the recorded decay rates alone, without knowing the graph.

| Regime | η | Bound on ‖a‖ | Largest ‖a‖ in the figure |
|---|---|---|---|
| Collapse | 0.05 | 2.00 | ≈ 2.0, at the start |
| Balanced | 2 | 2.29 | ≈ 2.0, at the start |
| Runaway | 10 | 11.43 | ≈ 6.1 |

Every run stays inside its bound. "Runaway" is replay-dominated but bounded, which is what section 3.2 says must happen whatever replay does.

## Observations and caveats

- **Winner-take-all at moderate replay.** In the single run (η·p = 0.24), three of four memories went to exactly 0.
  - It is unknown whether the Balanced regime keeps several memories alive or just one, because the notebook did not print per-memory activations for the sweeps.
  - `run_experiments.py` now records them in `metrics.json`.
- **Retrieval reflects survival, not similarity.** Weighting by activation means a collapsed memory cannot be retrieved even by an exact-match query (see Results §2).
- **Why replay picked the person photo is an open question.**
  - The replay score $s = X X^\top a$ uses raw, unnormalised embeddings, so memories with larger norms, or more central positions in embedding space, get larger scores. $\beta = 3$ then sharpens the softmax.
  - Two things to check: the per-image embedding norms, and a version of $F(a)$ that uses normalised $X$.
- **$E$ only descends while replay is weak.** It plateaus in the single run and rises in the Runaway regime. That is expected, since replay is not part of $E$.
- **Small sample.** These are qualitative demonstrations of the regimes, not measurements: four images, one query, one seed.
- **The first attempt failed.** Notebook cell 0 used Wikimedia URLs without a User-Agent header. Every download failed, and the run stopped with `RuntimeError: stack expects a non-empty TensorList`. Cell 2 switched to Unsplash URLs and a browser User-Agent.
- **Cell 4 shows a stale figure.**
  - It ran at execution 9, before the successful run at execution 12. The `three_regimes.png` it re-displays came from an earlier failed run and has empty axes.
  - That image is left out of `results/`.

## Architecture

```text
+-- 1. ENCODE ------------------------------- memory_dynamics/encoder.py --+
|                                                                          |
| image URL --(requests)--> bytes --(Pillow)--> RGB image                  |
|                                                   |                      |
|    transformers: AutoProcessor + I-JEPA ViT-H/14  |                      |
|                                                   v                      |
|                                 1280-d embedding (mean over patches)     |
|                                                                          |
+---------------------------------------------------+----------------------+
                                                    |
    outputs/embeddings.pt <--(torch.save)-----------+
    (--embeddings loads it back and                 |
     skips the 2.5 GB model) ......................>|
                                                    v
+-- 2. MEMORY DYNAMICS ----------------------- memory_dynamics/memory.py --+
|                                                                          |
| store()          one memory per image: a_i = 1, decay rate alpha_i       |
|    |                                                                     |
|    v                                                                     |
| step() x 500     da/dt = - decay - diffusion - inhibition + replay       |
|    |                                                                     |
|    +--> retrieve()    score_i = cos(query, x_i) * a_i                    |
|    +--> energy()      E(a)                                               |
|                                                                          |
| randomness: NumPy RandomState(seed) -> decay rates, replay coin flip     |
|                                                                          |
+---------------------------------------------------+----------------------+
                                                    |
                                                    v
+-- 3. OUTPUTS ------------------------- plotting.py, run_experiments.py --+
|                                                                          |
| figures (matplotlib)                     metrics.json                    |
|                                                                          |
+--------------------------------------------------------------------------+
```

| Part | File | Libraries | Job |
|---|---|---|---|
| Encoder | [`memory_dynamics/encoder.py`](memory_dynamics/encoder.py) | requests, Pillow, transformers, torch | Image URL → 1280-d I-JEPA embedding |
| Memory | [`memory_dynamics/memory.py`](memory_dynamics/memory.py) | torch, NumPy | Stores embeddings; runs the equation, retrieval and energy |
| Plotting | [`memory_dynamics/plotting.py`](memory_dynamics/plotting.py) | matplotlib | Trajectory, energy, activation and regime figures |
| Runner | [`run_experiments.py`](run_experiments.py) | argparse, json + the above | Runs single run → sweep A → sweep B and writes `outputs/` |

## Repository layout

```
├── README.md
├── requirements.txt
├── run_experiments.py         # runs every experiment, writes outputs/
├── memory_dynamics/
│   ├── memory.py              # DynamicalMemory: the equation above
│   ├── encoder.py             # I-JEPA image -> 1280-d embedding
│   └── plotting.py            # figures
├── notebooks/
│   └── metacognition.ipynb    # original Colab notebook, outputs intact
├── results/                   # recorded output of the notebook run
│   ├── figures/
│   │   ├── dynamics_demo.png
│   │   ├── three_regimes_sweep_a.png
│   │   └── three_regimes_sweep_b.png
│   ├── metrics.json
│   └── notebook_output.txt    # printed output, verbatim
├── web/                       # website: Vite + React + TypeScript
│   ├── src/                   # page, components, styles
│   └── public/memories/       # thumbnails of the four photos
└── vercel.json                # tells Vercel to build web/
```

## Reproducing

```bash
pip install -r requirements.txt
python run_experiments.py
```

The first run downloads the I-JEPA ViT-H/14 checkpoint (about 2.5 GB) and the four images. It saves the embeddings to `outputs/embeddings.pt`. To re-run only the dynamics without loading the model:

```bash
python run_experiments.py --embeddings outputs/embeddings.pt
```

Other flags:

- `--seed`: default 42
- `--steps`: default 500
- `--out`: default `outputs/`

**What the script reproduces.**

- It runs the notebook's cells 2 → 3 → 5 in that order, on one random stream seeded with 42. That is also the order the notebook ran them (executions 12, 13 and 14).
- The dynamics code matches the notebook's class exactly. This was checked by running both on the same embeddings and comparing every final activation, energy, retrieval score and sweep result.
- Given the same embeddings, the script reproduces the single run and both plotted sweeps.

**What it does not reproduce.**

- Sweep B's printed numbers came from the notebook's extra re-run, so they will not match.
- Embeddings computed on other hardware or library versions can differ in the last few digits. Winner-take-all dynamics can amplify that.

## Website

[`web/`](web/) is a single-page write-up of everything above: the model, the architecture, the paper comparison and every recorded result.

- **No copied numbers.** It imports `results/metrics.json` and the three Colab figures straight from `results/`, so the page always shows the recorded run.
- **Built with** Vite, React and TypeScript. The charts are hand-written SVG and the maths uses KaTeX. There are no other runtime dependencies.

Run it locally:

```bash
cd web
npm install
npm run dev        # http://localhost:5173
npm run build      # type-checks, then writes web/dist
```

Deploy it on Vercel:

1. Import this repository at vercel.com/new.
2. Keep the defaults. The root `vercel.json` already sets the install command, the build command and the `web/dist` output.

## References

- Ghosh, Jaiswal, Gupta. *The Dynamics of Memory: Replay, Diffusion, and Attractor Formation.* MetaCognition Labs.
- Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.* CVPR 2023. Model: [`facebook/ijepa_vith14_1k`](https://huggingface.co/facebook/ijepa_vith14_1k).
