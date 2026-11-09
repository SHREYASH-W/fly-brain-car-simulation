# 🪰🏎️ Fly-Brain F1: A Fruit Fly Connectome Driving a Race Car

A closed-loop experiment that connects a **whole-brain simulation of the adult fruit fly (*Drosophila melanogaster*)** to a **simulated racing car** (Gymnasium `CarRacing-v3`).

The fly brain model is built from the FlyWire v783 connectome: **138,639 neurons** and **~15 million synaptic connections**. It runs as a persistent, spiking leaky-integrate-and-fire (LIF) network on a GPU. Each frame from the car's camera is converted into neural input, the brain is simulated for a few milliseconds, and the resulting spikes are decoded into steering, throttle, and brake commands.

> **Read this first:** this is a **closed-loop integration demo**, not a claim that a fly brain "learned to drive" or that it drives well. There is no learning, the sensory mapping is a proxy, and the car does not complete the track. See [What this is and is not](#what-this-is-and-is-not) and [Limitations](#limitations-and-honest-caveats).

![Best episode](results/videos/best_episode.gif)

*Best fly-brain episode (episode 33, total reward 35.4).*

---

## Table of contents

1. [TL;DR](#tldr)
2. [What this is and is not](#what-this-is-and-is-not)
3. [How it works](#how-it-works)
4. [Neuron mapping](#neuron-mapping)
5. [Experimental protocol](#experimental-protocol)
6. [Results](#results)
7. [Limitations and honest caveats](#limitations-and-honest-caveats)
8. [Setup and reproduction](#setup-and-reproduction)
9. [Repository layout](#repository-layout)
10. [Troubleshooting](#troubleshooting)
11. [Future work](#future-work)
12. [Acknowledgements and references](#acknowledgements-and-references)

---

## TL;DR

| | |
|---|---|
| **Connectome** | FlyWire v783, 138,639 neurons, ~15.09M connection records |
| **Neuron model** | Original PyTorch LIF implementation from [`eonsystemspbc/fly-brain`](https://github.com/eonsystemspbc/fly-brain) |
| **Environment** | Gymnasium `CarRacing-v3` (96×96 RGB observations, continuous actions) |
| **Hardware used** | Kaggle Tesla T4 GPU (≈15 GB VRAM) |
| **Coupling** | 20 neural steps (2.0 ms of brain time) per environment step |
| **Protocol** | 50 episodes × 3 conditions (fly brain, no-brain, rule-based heuristic) |
| **Headline result** | Fly-brain controller mean reward **−40.9**, vs **−53.4** (heuristic) and **−93.2** (no brain). All conditions are net-negative and none completes the track. |
| **Seed** | 42 |

---

## What this is and is not

**This is:**
- A working end-to-end pipeline in which a **full-scale connectome-based brain simulation** runs live inside a control loop with persistent neural state.
- A transparent testbed for studying sensorimotor coupling between a biological network model and an artificial environment.
- A clearly labeled, reproducible baseline comparison (fly brain vs. no brain vs. heuristic).

**This is not:**
- ❌ **Learned driving.** Connectome weights are fixed. There is no plasticity, training, or reinforcement learning.
- ❌ **A biologically faithful visual system.** A fly has no pathway for "racetrack". Sensory inputs go to **arbitrary proxy neurons** (see [Neuron mapping](#neuron-mapping)).
- ❌ **Evidence that the connectome alone solves the task.** The controller includes hand-written components (visual feature extraction, a heading-correction term, and a throttle floor) that contribute to behavior. No ablation yet isolates the brain's contribution.
- ❌ **A good racing agent.** Mean reward is negative and 0% of episodes exceed the +100 "success" threshold.

The notebook runs in `PROXY_DEMO` mode. A `BIOLOGICALLY_MAPPED` mode is described in the design notes but only the P9 forward-walking neurons have a documented biological mapping.

---

## How it works

```
 ┌──────────────┐   RGB 96×96   ┌──────────────────┐  5 visual features   ┌─────────────────┐
 │ CarRacing-v3 │ ────────────▶ │  Vision (OpenCV) │ ───────────────────▶ │ Sensory Encoder │
 │ environment  │               │  HSV road mask   │                      │ features → Hz   │
 └──────▲───────┘               └──────────────────┘                      └────────┬────────┘
        │                                                                          │ Poisson input rates
        │ steering / throttle / brake                                              ▼
 ┌──────┴───────┐   spike counts   ┌──────────────────────────────────────────────────────┐
 │ Motor Decoder│ ◀─────────────── │ Fly Brain Runtime: PyTorch LIF, 138,639 neurons,     │
 │ (smoothed)   │   (20 steps)     │ sparse CSR weights on GPU, persistent state          │
 └──────────────┘                  └──────────────────────────────────────────────────────┘
```

### 1. Environment
`gym.make("CarRacing-v3", continuous=True, domain_randomize=False)` gives 96×96×3 observations and a 3-dimensional continuous action `[steering ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1]]`. No privileged simulator state (car position, track geometry) is used.

### 2. Vision (heuristic, no learning)
`extract_features(obs)` converts the frame to HSV, thresholds a low-saturation "gray road" mask, cleans it with a morphological close, and computes five normalized features:

| Feature | Meaning |
|---|---|
| `road_visibility` | Fraction of pixels classified as road |
| `road_center_error` | Offset of the road's mean position from the image center (lower half) |
| `heading_error` | Change in road center between the near and far parts of the view |
| `left_clearance` / `right_clearance` | Distance from image center to the left/right road edge at mid-height |

### 3. Sensory encoder
Clearance features become Poisson input rates (max 100 Hz) applied to two blocks of 100 proxy neurons (left and right "visual field"). The encoder stimulates the **opposite** side from the nearer road edge, so the model is nudged to steer away from it.

### 4. Fly brain runtime
`FlyBrainRuntime` wraps the original `TorchModel` (imported from the upstream `run_pytorch.py`). The brain state is **persistent within an episode** and reset at the start of each one. Each environment step runs **20 neural time steps** (`DT` = 0.1 ms, so 2.0 ms of brain time) and records spikes at every step.

### 5. Motor decoder
Spikes in each motor group are summed over the 20-step window and normalized by `n_neurons × n_steps`:

- `steering = rate(steering_right) − rate(steering_left)`
- `throttle = max(0.2, 1.5 × rate(P9))`, so there is a **baseline forward drive of 0.2**
- `brake = rate(brake group)`

All three outputs are exponentially smoothed (α = 0.3).

### 6. Closed-loop controller
`EnhancedFlyBrainController` ties it together each step: extract features → encode → add direct P9 drive → simulate → decode → add a small heading correction (`steering −= 0.3 × heading_error`) → `env.step()`.

**P9 drive:** the two P9 neurons receive 400 Hz input when `road_visibility > 0.1` and 50 Hz otherwise. This is a direct, hand-set stimulation (not a result of the visual pathway) and is documented in the notebook as a deliberate choice to "ensure cross-embodiment actuation".

---

## Neuron mapping

| Role | Neurons | Basis |
|---|---|---|
| Throttle / forward drive | P9 left (`720575940627652358`, index 83620) and P9 right (`720575940635872101`, index 119032) | **Documented** forward-walking neurons |
| Left visual input | Indices 1000–1099 | ⚠️ **Proxy** (arbitrary) |
| Right visual input | Indices 2000–2099 | ⚠️ **Proxy** (arbitrary) |
| Steering left | Indices 3000–3049 | ⚠️ **Proxy** (arbitrary) |
| Steering right | Indices 4000–4049 | ⚠️ **Proxy** (arbitrary) |
| Brake | Indices 5000–5019 | ⚠️ **Proxy** (arbitrary) |

The proxy indices are fixed and documented so results are reproducible, but they are **not** claimed to correspond to real fly circuitry for vision, turning, or braking.

---

## Experimental protocol

Three conditions, **50 episodes each**, up to 1,000 environment steps per episode (the `CarRacing-v3` time limit):

| Condition | Description | Seeds |
|---|---|---|
| **Fly Brain** | Full pipeline above | `42 + episode` |
| **No-Brain** | Zero action every step (`[0, 0, 0]`), so the car sits still | `42 + episode + 100` |
| **Heuristic** | Rule-based: `steer = −2 × road_center_error`; throttle 0.6 and no brake if road visible, else throttle 0.2 and brake 0.5 | `42 + episode + 100` |

Statistics: independent two-sample t-tests on total episode reward, plus the Pearson correlation between mean P9 spikes per environment step and total reward across fly-brain episodes.

---

## Results

![Multi-episode analysis](results/plots/multi_episode_analysis.png)

### Performance (50 episodes per condition)

| Condition | Mean reward | Std dev | Mean steps | Success rate (reward > 100) |
|---|---:|---:|---:|---:|
| **Fly Brain** | **−40.93** | 34.86 | 1000 | 0.0% |
| Heuristic | −53.35 | 4.70 | 1000 | 0.0% |
| No-Brain | −93.17 | 0.58 | 1000 | 0.0% |

### Statistical tests

| Comparison | t | p | Verdict (α = 0.05) |
|---|---:|---:|---|
| Fly Brain vs No-Brain | 10.594 | < 0.0001 | Significant |
| Fly Brain vs Heuristic | 2.495 | 0.0143 | Significant |

### Neural activity vs. reward

![Neural correlation](results/plots/neural_correlation.png)

Pearson **r = 0.855** between mean P9 spikes per environment step and total episode reward across the 50 fly-brain episodes. See the caveat below before interpreting this.

### Observed behavior

- Fly-brain results are **widely spread** (std 34.9). Some episodes end clearly positive (best: +35.4, episode 33) and many end near −70 or worse.
- In the logged first episode, reward rises early (peaking around +18 near step 100) and then falls by a steady ≈ −10 per 100 steps. That is the signature of the car leaving the track and collecting no new tiles while the −0.1/frame penalty accumulates.
- Per-step P9 spike counts in that episode were small (0–2 spikes per environment step across both neurons).

---

## Limitations and honest caveats

Please keep these in mind when reading the numbers above.

1. **Net-negative performance in every condition.** "Better than baseline" here means *less negative*. No condition exceeds the +100 success threshold.
2. **The no-brain baseline is a low bar.** With zero action the car does not move, so it simply accumulates the time penalty (≈ −93). Any controller that drives forward on the track will beat it.
3. **No ablation isolates the brain's contribution.** The fly-brain controller also uses hand-coded vision, a heading-correction term (`−0.3 × heading_error`), and a throttle floor of 0.2. The advantage over the heuristic may come largely from those components rather than from neural dynamics.
4. **Throttle is probably dominated by the floor.** With 2 P9 neurons and 20 steps, a count of 0–2 spikes gives a normalized rate ≤ 0.05, so `1.5 × rate ≤ 0.075`, which is below the 0.2 floor. In the logged episode the P9 contribution to throttle therefore appears to be effectively overridden. This should be verified by logging pre-floor throttle.
5. **The P9–reward correlation is likely partly circular.** P9 input is set directly from `road_visibility`, so P9 activity tracks "how much road is in view", which itself tends to track reward. The r = 0.855 shows a coupling between the stimulation scheme and performance. It is **not** evidence of emergent brain-driven control.
6. **Proxy steering/brake neurons are unvalidated.** Spiking in the proxy steering and brake groups was not logged, so it is unknown how much they influence the actions.
7. **Unpaired seeds.** Fly-brain episodes use seeds `42…91`; baselines use `142…191`. Tracks differ between conditions, so comparisons are not episode-paired.
8. **Statistical caveats.** The fly-brain reward distribution is wide and likely non-normal; the t-tests use SciPy's default equal-variance assumption. Welch's t-test, a non-parametric test (Mann–Whitney U), or bootstrap confidence intervals would be more appropriate.
9. **Short neural window.** Each environment step simulates only 2 ms of brain time, so slower network dynamics have little time to develop within a step.
10. **Fixed, single configuration.** No hyperparameter sweeps, and a single seed family (42).
11. **Synapse count wording.** The notebook's first markdown cell says "5M synapses", but the loaded connectivity table has **15,091,983 rows** (connection records between neuron pairs). This README uses the loaded figure.

---

## Setup and reproduction

### Option A: Kaggle (what the original run used)

1. Create a new Kaggle notebook and enable **GPU (T4)** and **Internet** (needed once for `git clone` and `pip`).
2. Upload `fly-brain.ipynb` (or paste its cells).
3. Run all cells. The notebook:
   - installs `gymnasium[box2d]`, `imageio`, and `opencv-python-headless`,
   - clones [`eonsystemspbc/fly-brain`](https://github.com/eonsystemspbc/fly-brain) into `/kaggle/working/fly-brain`,
   - loads `2025_Completeness_783.csv` and `2025_Connectivity_783.parquet` from that repo's `data/` folder,
   - runs the three-condition protocol and writes outputs to `/kaggle/working/fly_f1_outputs/`.
4. Download the generated `fly_f1_complete_results.zip`.

Total time was roughly **50 minutes end to end on a T4** (based on notebook execution timestamps).

### Option B: Local machine

**Requirements:** Python 3.10+, an NVIDIA GPU with ~15 GB VRAM recommended (CPU works but is very slow), and a CUDA build of PyTorch.

```bash
# 1. Get the upstream model + connectome data
git clone https://github.com/eonsystemspbc/fly-brain.git
cd fly-brain

# 2. Install dependencies
pip install torch numpy pandas pyarrow scipy matplotlib \
            "gymnasium[box2d]" imageio opencv-python-headless jupyter

# 3. Copy this notebook into the repo, then edit the path constants in cell 1:
#    BASE_DIR   = Path(".")              # was /kaggle/working/fly-brain
#    DATA_DIR   = BASE_DIR / "data"
#    CODE_DIR   = BASE_DIR / "code"
#    OUTPUT_DIR = Path("./fly_f1_outputs")   # was /kaggle/working/fly_f1_outputs

# 4. Launch
jupyter lab fly-brain.ipynb
```

> **Import order matters:** `import pyarrow` **before** `import torch`, otherwise you may hit libarrow conflicts. The notebook already does this in its first code cell.

### Reproducibility notes
- Seeds: `SEED = 42` for NumPy and PyTorch (CUDA included); environment seeds are `SEED + episode` (fly brain) and `SEED + episode + 100` (baselines).
- GPU nondeterminism and stochastic Poisson input mean exact reward values may differ slightly between runs and hardware.

---

## Repository layout

Suggested structure for this repo:

```
.
├── README.md
├── fly-brain.ipynb                    # the full experiment
└── results/
    ├── experiment_results.csv         # summary table (per-condition stats)
    ├── final_summary.txt              # printed summary + integrity notes
    ├── plots/
    │   ├── multi_episode_analysis.png
    │   └── neural_correlation.png
    └── videos/
        ├── best_episode.mp4
        └── best_episode.gif
```

The notebook writes to `fly_f1_outputs/{videos,plots,data,config}/`; copy the files into `results/` before committing.

### Key components inside the notebook

| Component | Purpose |
|---|---|
| `extract_features()` | HSV road detection → 5 normalized visual features |
| `SensoryEncoder` | Features → Poisson input rates on proxy visual neurons |
| `FlyBrainRuntime` | Wraps the upstream `TorchModel`, holds persistent state, runs N neural steps |
| `MotorDecoder` | Spike counts → smoothed steering/throttle/brake |
| `EnhancedFlyBrainController` | The closed-loop episode runner |
| `run_baseline_episodes()` | No-brain and heuristic baselines |

---

## Troubleshooting

**The `.mp4` won't play in a browser, on GitHub, or on social media.**
The notebook writes video with OpenCV's `mp4v` codec at 96×96, which many browsers and platforms cannot decode and which is below some platforms' minimum size. Re-encode to H.264:

```bash
ffmpeg -i best_episode.mp4 \
  -vf "scale=720:720:flags=neighbor" \
  -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p \
  -movflags +faststart -an best_episode_h264.mp4
```
`flags=neighbor` keeps the pixels crisp when upscaling. Alternatively, use the generated GIF.

**`libarrow` / import errors.** Make sure `import pyarrow` comes before `import torch`.

**Very slow simulation.** You are probably on CPU. Check that `torch.cuda.is_available()` is `True`.

**`FileNotFoundError: Connectome data files missing!`** The upstream repo was not cloned, or `DATA_DIR` does not point to its `data/` folder.

**First weight load takes a while.** `get_weights(...)` builds and caches the sparse connectivity matrix; the first run can take a couple of minutes.

---

## Future work

- **Ablations:** run with P9 input only, no heading correction, no throttle floor, and with the proxy sensory neurons silenced, to measure what the brain actually contributes.
- **Log everything:** pre-floor throttle, spike counts in every motor group, and population activity over time.
- **Paired evaluation:** use identical track seeds across conditions and report confidence intervals or non-parametric tests.
- **Biologically grounded mapping:** replace proxy indices with identified visual-input and descending neurons (e.g., turning-related descending neurons) from FlyWire annotations.
- **Longer neural windows:** more neural steps per environment step to let slower dynamics develop.
- **Plasticity or readout learning:** train a small readout on spike activity while keeping the connectome fixed, and compare to the current untrained decoder.
- **Statistical power:** more episodes and multiple seed families.

---

## Acknowledgements and references

- **Upstream fly-brain model and data:** [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) (PyTorch LIF implementation and the FlyWire v783 processed files). Please refer to that repository for the model's primary citations and license.
- **FlyWire connectome:** Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature (2024); Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature (2024). Please verify exact citation details and data-use terms with the FlyWire consortium.
- **Environment:** [Gymnasium](https://gymnasium.farama.org/) `CarRacing-v3`.
- Built with PyTorch, OpenCV, NumPy, pandas, SciPy, Matplotlib, and imageio.

### License

Add a license of your choice for the notebook code in this repository, and check that it is compatible with the upstream repository's license and the FlyWire data terms.

---

*Experiment run on 2026-11-08 on a Kaggle Tesla T4 GPU, Python 3.13, PyTorch 2.11 (CUDA 13).*
