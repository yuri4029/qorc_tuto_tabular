# Boson Sampling: Practical Instructions

This guide collects the concepts and workflow discussed for building and understanding a Perceval Boson Sampling simulation. The example dimensions are 14 photons, 60 modes, and up to 5,000,000 requested shots. Start with a much smaller problem to verify your Perceval installation and code before scaling up.

## 1. Understand the quantities

- **Photon count (`n`)**: number of photons injected into the network.
- **Mode count (`m`)**: number of optical paths or modes in the interferometer; normally `m >= n` for this example.
- **Shot count (`N`)**: number of repeated measurements, not the number of photons.
- **Input occupation**: a list saying how many photons start in each mode. For 14 photons in the first 14 of 60 modes, it has fourteen `1`s followed by forty-six `0`s.
- **Output occupation**: a list with one count per output mode. For ideal lossless sampling, its entries sum to `n`.

An example output `[1, 0, 2, 0]` means one photon in mode 0, none in mode 1, two in mode 2, and none in mode 3.

## 2. Understand the probability being sampled

A linear-optical network is represented by an `m × m` unitary matrix `U`. For input occupation `T` and output occupation `S`, each containing `n` photons, the ideal probability is

$$
P(S \mid T) = \frac{\left|\operatorname{Perm}(U_{S,T})\right|^2}
{\left(\prod_i s_i!\right)\left(\prod_j t_j!\right)}.
$$

`U_{S,T}` is an `n × n` submatrix formed by selecting rows for the output modes and columns for the input modes, repeating rows or columns when a mode has multiple photons. `Perm` is the matrix permanent: like a determinant, but without alternating signs. Repeated sampling estimates probabilities; it does not return one deterministic output pattern.

## 3. Create an input state and target unitary

A minimal setup follows this pattern (check API details against the Perceval version installed in your environment):

```python
import perceval as pcvl
from perceval.algorithm import Sampler

n = 14
m = 60
number_of_shots = 5_000_000

input_state = pcvl.BasicState([1] * n + [0] * (m - n))
target_unitary = pcvl.Matrix.random_unitary(m)
```

For a simple first test, reduce `n`, `m`, and the shot count (for example, use 2–3 photons and a few modes). This makes it easier to detect API errors and inspect results.

## 4. Understand the MZI and its phase parameters

A Mach–Zehnder interferometer (MZI) is a two-mode building block made from beam splitters and phase shifters. In Perceval, `pcvl.BS()` is a beam splitter and `pcvl.PS(phi=...)` is a phase shifter. A phase shifter adds a phase to one mode; it does not itself change photon number.

```python
mzi = (
    pcvl.BS()
    // (0, pcvl.PS(phi=pcvl.Parameter("phi_a")))
    // pcvl.BS()
    // (1, pcvl.PS(phi=pcvl.Parameter("phi_b")))
)
```

The `//` operator composes components in sequence. The `Parameter` objects above are named symbolic placeholders, not numeric phase values. This template can be displayed while symbolic, but it cannot by itself define a numeric optical transformation until its parameters are assigned.

When decomposing a fully numeric target unitary, the decomposition routine uses copies of the MZI template and calculates the phase settings needed for the target. You generally do not manually set the template's placeholder values for that decomposition.

## 5. Decompose the unitary into an interferometer mesh

```python
linear_circuit = pcvl.Circuit.decomposition(
    target_unitary,
    mzi,
    phase_shifter_fn=pcvl.PS,
    shape=pcvl.InterferometerShape.TRIANGLE,
)
```

- `phase_shifter_fn=pcvl.PS` tells the decomposition which phase-shifter component to use for single-mode phases required by the implementation.
- `shape=...TRIANGLE` chooses a triangular Reck-style arrangement of two-mode elements. `RECTANGLE` is another common arrangement. These are layouts for implementing a unitary, not different Boson Sampling rules.
- A triangular decomposition of an `m`-mode unitary uses `m(m−1)/2` two-mode elements. At 60 modes, that is `60 × 59 / 2 = 1770`.
- For an ideal lossless implementation of the same unitary, the mesh shape does not change its output probabilities. Layout can affect circuit depth and the effect of component losses in a physical device.

## 6. Configure the processor and sample

A typical Perceval setup is:

```python
processor = pcvl.Processor("SLOS", linear_circuit)
processor.with_input(input_state)
sampler = Sampler(processor)
```

Use the sampling method supported by your installed Perceval version to request shots (commonly `sampler.samples(number_of_shots)`). Inspect the returned object's type and fields: some versions expose samples as a list, while others return a result structure containing counts and performance information. Adapt the analysis to the actual returned format rather than assuming it is a plain list.

**Resource warning:** SLOS is a strong simulation backend and can require substantial memory and compute as photon and mode counts grow. The number of possible occupation patterns is `binomial(m+n−1, n)`. A request for five million shots can also consume significant time and memory. First benchmark a small case, then increase sizes and shot count gradually; do not assume the 14-photon, 60-mode, 5-million-shot run will be practical on every computer.

## 7. Analyze the sampled results

Convert the returned samples to a count mapping where each occupation pattern maps to its occurrence count. If `samples` is a list of occupation patterns, for example:

```python
from collections import Counter

counts = Counter(samples)
number_of_observed_shots = sum(counts.values())
frequencies = {
    pattern: count / number_of_observed_shots
    for pattern, count in counts.items()
}
```

If Perceval returns a count mapping already, use it directly and normalize its counts instead of passing it to `Counter`.

Useful checks and summaries:

1. **Photon-number conservation:** verify `sum(pattern) == n` for each pattern in the ideal lossless case. Loss, heralding, or postselection can change what should be expected.
2. **Empirical probabilities:** a pattern occurring `c` times in `N_observed` shots has estimated probability `c / N_observed`. This estimate fluctuates because the sample is finite.
3. **Collision statistics:** count outcomes in which any mode has occupancy greater than one.
4. **Mean occupancy per mode:** average each mode's photon count across all shots.
5. **Theory comparison:** for a small tractable problem, compare measured frequencies `q(S)` against exact probabilities `p(S)`. Total variation distance is `1/2 * sum_S |p(S) - q(S)|`. Computing the full distribution may be infeasible for large instances.

## 8. Troubleshoot in order

- If `Circuit.linear_circuit(...)` raises `AttributeError`, that name is not a valid constructor in the installed Perceval API. Use the documented `Circuit.decomposition(...)` workflow when decomposing a unitary, and check the API for your installed version.
- If a method or argument on `Sampler` fails, check your installed Perceval version and inspect the returned type and API; signatures and return formats can vary by version.
- If a symbolic phase prevents unitary computation, assign numeric values or use a decomposition from a numeric target unitary.
- If the simulation is slow or runs out of memory, reduce photon/mode counts and shots. A large number of requested samples does not make exact simulation cheaper.

## Summary

Boson Sampling prepares identical photons, sends them through a unitary interferometer, and repeatedly records output occupation patterns. MZI phase settings implement the chosen unitary; `TRIANGLE` or `RECTANGLE` selects its mesh layout. After sampling, count patterns, normalize to empirical frequencies, check photon number, and compute meaningful statistics. Validate the workflow on a small instance before attempting a large simulation.
