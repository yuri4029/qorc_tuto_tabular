# QORC tutorial: quantum optical reservoir computing on tabular network data

A self-contained tutorial that applies a **Quantum Optical Reservoir Computer (QORC)**,
following [Sakurai et al.](https://opg.optica.org/opticaq/abstract.cfm?URI=opticaq-3-3-238),
to network intrusion detection on NSL-KDD, with a zero-day-style test: attack families
that are never shown during training.

```
122 tabular features ─► PCA(12) ─► tanh phases ─► [ U ─► phase dials ─► U ] ─► 220 Fock probabilities ─► classical head
                                                    fixed Haar-random,          3 photons, 12 modes,      the only part
                                                    never trained               no-bunching readout       that learns
```

## Contents

| file | what it is |
|---|---|
| `blog.ipynb` | **the tutorial**: dataset, circuit, training, comparison against classical baselines over 3 seeds |
| `boson_sampling_basics.py` | Boson Sampling end to end with Perceval at a small size (3 photons, 6 modes): MZI decomposition, sampling, comparison with exact probabilities |
| `boson_sampling_explain.md` | the theory behind it: permanents, MZI meshes, what to compute from samples |
| `boson_sampling_instructions.md` | practical Perceval guide and troubleshooting |
| `basic_rc.py` | a classical echo-state reservoir on a sine wave, to show the reservoir idea without optics |
| `data/` | put `NSL_KDD_labeled.csv` here (see `data/README.md`) |
| `archive/` | earlier drafts, kept for reference; superseded by `blog.ipynb` |

## Running

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook blog.ipynb
```

The notebook runs on a CPU in about 3 minutes. The photonic circuit is built and simulated
with Perceval's SLOS backend; for the ~13,000 rows × 3 seeds the same probabilities are
computed with a vectorised permanent in NumPy, checked against Perceval inside the notebook.

## What the tutorial finds

Over three random reservoirs (details and numbers in Section 4 of the notebook):

* Detectors trained on **normal traffic only** generalise to unseen attack families far better
  than a classifier trained on normal + DoS (macro AUC 0.90–0.95 vs 0.56–0.74).
* With an autoencoder detector, the reservoir output, its own 12 PCA inputs, and a classical
  random Fourier lift of the same width (220) perform the same.
* With a **linear** detector, the reservoir lifts macro AUC from 0.84 to 0.94 over its inputs,
  but the classical random lift does just as well. The reservoir is a useful nonlinear feature
  map for cheap heads; nothing here shows a quantum advantage.
* R2L attacks stay below ~16% detection at a 1% false-alarm budget for every model.

## Design choices that matter

* All preprocessing (scaler, PCA, phase encoding) is fitted on normal training rows only.
* Phases are mapped into (0, π) with a smooth `tanh` squash. Mapping onto [0, 2π] would make the
  two ends of each feature identical (phases are periodic), and min-max clipping erases exactly
  the out-of-range values that anomalies produce.
* Only phase *differences* affect the output, so 12 dials carry 11 effective inputs. On very
  small circuits (e.g. 4 modes) this loses a quarter of the input.
* Every head is trained on four feature views (`quantum`, `dials`, `rff`, `raw`), so the reservoir
  only gets credit for what its own input and a classical lift of the same size cannot do.
