# Boson Sampling: Explanation and Analysis

## What Boson Sampling means

Boson Sampling is a photonic sampling task. Identical photons are prepared in selected input modes, sent through a linear-optical interferometer, and detected at its output. One run (a *shot*) produces an occupation pattern: the number of photons detected in each output mode.

For example, `[1, 0, 2, 0]` means one photon in mode 0, no photon in mode 1, two photons in mode 2, and none in mode 3. For an ideal lossless interferometer, the total photon number is conserved, so the entries sum to the number of input photons.

The experiment or simulation is repeated many times. The resulting patterns are samples from a probability distribution determined by the input state and the interferometer.

## How the interferometer defines probabilities

An `m`-mode interferometer is represented by an `m × m` unitary matrix `U`. Given an input occupation pattern `T` and an output pattern `S`, each with `n` photons, the ideal probability is


$$
P(S \mid T)=\frac{\left|\operatorname{Perm}\left(U_{S,T}\right)\right|^2}
{\left(\prod_i s_i!\right)\left(\prod_j t_j!\right)}
$$


Here `U[S,T]` is an `n × n` submatrix: rows of `U` are selected according to the output modes (repeated for multiple photons in a mode), and columns according to the input modes. `Perm` denotes the matrix permanent, like a determinant but with no alternating signs.

Computing permanents becomes difficult as photon number grows. This is one reason Boson Sampling is studied as a task that may be hard to simulate classically.

## Building a unitary from MZIs

A Mach–Zehnder interferometer (MZI) is a two-mode building block, usually made from two beam splitters and phase shifters. In Perceval, `pcvl.BS()` represents a beam splitter and `pcvl.PS(phi=...)` represents a phase shifter.

For example:

```python
mzi = (
    pcvl.BS()
    // (0, pcvl.PS(phi=pcvl.Parameter("φ_a")))
    // pcvl.BS()
    // (1, pcvl.PS(phi=pcvl.Parameter("φ_b")))
)
```

The `//` operator places components in sequence. The phase parameters are symbolic placeholders in this template; they are not numeric settings yet. A circuit made from this template needs numeric phase values before it can be evaluated as a physical transformation.

When a target unitary is decomposed into a mesh of MZIs, the decomposition algorithm determines the required phase values. In a call such as:

```python
linear_circuit = pcvl.Circuit.decomposition(
    unitary,
    mzi,
    phase_shifter_fn=pcvl.PS,
    shape=pcvl.InterferometerShape.TRIANGLE,
)
```

the MZI is a reusable template. The algorithm calculates the settings needed to implement `unitary`; `phase_shifter_fn` specifies the phase-shifter component used for any remaining single-mode phases.

`InterferometerShape.TRIANGLE` selects a triangular (Reck-style) arrangement of two-mode elements for the decomposition. `RECTANGLE` is another common arrangement. They are different layouts for implementing a unitary, not different sampling rules. For an ideal, lossless implementation of the same unitary, they produce the same output probabilities. Their depth and physical loss characteristics can differ.

For `m` modes, a triangular decomposition uses `m(m−1)/2` two-mode elements. For 60 modes this is:

```text
60 × 59 / 2 = 1770
```

## Sampling and what to calculate afterward

Suppose a simulation uses 14 photons, 60 modes, and requests 5,000,000 shots. Each shot produces one 60-entry occupation pattern whose entries sum to 14 in an ideal lossless model.

After collecting samples, count how often each pattern occurs and divide by the total number of shots to estimate its probability:

```python
from collections import Counter

counts = Counter(samples)
number_of_shots = len(samples)
frequencies = {
    pattern: count / number_of_shots
    for pattern, count in counts.items()
}
```

For example, if a pattern occurs 500 times in 5,000,000 shots, its empirical frequency is `500 / 5_000_000 = 0.0001`. This is an estimate, not the exact theoretical probability; finite sampling causes statistical fluctuations.

Useful follow-up calculations include:

- **Photon-number check:** for each pattern, verify that the total occupation equals the input photon count (unless loss or postselection is part of the model).
- **Collision statistics:** count how often two or more photons occupy the same output mode.
- **Mode occupancies:** average the photon count in each output mode across shots.
- **Distribution comparison:** for a small enough problem, compare observed frequencies with theoretical probabilities. One metric is total variation distance, `½ Σₛ |pₛ − qₛ|`, where `p` is the theoretical distribution and `q` is the observed distribution.

The full output space can be very large. With many photons and modes, storing millions of distinct patterns or calculating every exact probability may be expensive. In practice, calculate summary statistics or compare a selected subset of outcomes unless the exact distribution is tractable.

## Interpreting a simulation

A Boson Sampling run does not usually produce one single “answer” to calculate. Its output is a collection of random photon patterns. The useful result is the empirical distribution and the statistics derived from it. To assess a simulator, compare those results with theory on a smaller test case, or check appropriate statistics and photon-number conservation.

