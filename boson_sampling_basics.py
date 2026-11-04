"""Boson Sampling with Perceval, end to end, at a size a laptop runs in seconds.

1. Draw a Haar-random unitary U.
2. Decompose U into a triangular (Reck-style) mesh of Mach-Zehnder interferometers.
3. Inject photons, sample output occupation patterns with the SLOS backend.
4. Compare the empirical frequencies with the exact probabilities.

Background: boson_sampling_explain.md and boson_sampling_instructions.md.

Run:  python boson_sampling_basics.py
"""

from collections import Counter
import math

import numpy as np
import perceval as pcvl
from perceval.algorithm import Sampler

N_PHOTONS = 3        # photons at the input
N_MODES = 6          # optical modes
N_SHOTS = 20_000     # repeated measurements
SEED = 42

# The original target size (14 photons, 60 modes, 5,000,000 shots) is far beyond exact
# simulation: the output space has C(60 + 14 - 1, 14) ~ 1e14 patterns. Perceval's
# "CliffordClifford2017" backend samples such sizes without enumerating the space.

pcvl.random_seed(SEED)
n_patterns = math.comb(N_MODES + N_PHOTONS - 1, N_PHOTONS)
print(f"{N_PHOTONS} photons in {N_MODES} modes -> {n_patterns} possible output patterns")

# 1. Target unitary
U = pcvl.Matrix.random_unitary(N_MODES)

# 2. MZI template: two 50/50 beam splitters and two phase shifters. The phases are
#    symbolic placeholders; the decomposition below solves for their numeric values.
mzi = (pcvl.BS()
       // (0, pcvl.PS(phi=pcvl.Parameter("phi_a")))
       // pcvl.BS()
       // (1, pcvl.PS(phi=pcvl.Parameter("phi_b"))))

circuit = pcvl.Circuit.decomposition(
    U, mzi,
    phase_shifter_fn=pcvl.PS,                    # component for the leftover per-mode phases
    shape=pcvl.InterferometerShape.TRIANGLE,     # Reck-style mesh: m(m-1)/2 MZIs
)
n_mzi = N_MODES * (N_MODES - 1) // 2
err = np.abs(np.array(circuit.compute_unitary()) - np.array(U)).max()
print(f"triangular mesh: {n_mzi} MZIs, max |U_circuit - U| = {err:.1e}")

# 3. Input: one photon in each of the first N_PHOTONS modes
input_state = pcvl.BasicState([1] * N_PHOTONS + [0] * (N_MODES - N_PHOTONS))
processor = pcvl.Processor("SLOS", circuit)
processor.with_input(input_state)
sampler = Sampler(processor)

counts = sampler.sample_count(N_SHOTS)["results"]       # {output pattern: count}
exact = sampler.probs()["results"]                      # {output pattern: probability}

# 4. Analysis
total = sum(counts.values())
freq = {s: c / total for s, c in counts.items()}
assert all(s.n == N_PHOTONS for s in counts), "ideal lossless circuit conserves photon number"

tvd = 0.5 * sum(abs(exact.get(s, 0.0) - freq.get(s, 0.0)) for s in set(exact) | set(freq))
collisions = sum(c for s, c in counts.items() if max(s) > 1) / total
mean_occ = np.sum([np.array(list(s)) * c for s, c in counts.items()], axis=0) / total

print(f"{total} shots, {len(counts)} distinct patterns observed")
print(f"total variation distance, empirical vs exact: {tvd:.4f}")
print(f"share of shots with two or more photons in one mode: {collisions:.3f}")
print("mean photons per output mode:", np.round(mean_occ, 3))
print("\nmost frequent patterns:")
for s, c in Counter(dict(counts.items())).most_common(5):
    print(f"  {s}  empirical {c / total:.4f}  exact {exact[s]:.4f}")
