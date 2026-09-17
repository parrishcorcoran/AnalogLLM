# Prior art — what is already done, and what is not

Compiled 2026-09-17. **Confidence caveat:** the network here blocks arxiv, nature,
PMC, IEEE and most publisher domains at the egress proxy, so this was built from
search results and abstracts, not from papers read end to end. Treat every claim
below as "worth verifying against the paper" rather than settled. Where a
distinction turns on a mechanism, go read the source before relying on it.

---

## 1. Pretrained transformers on analog hardware — DONE, on silicon

Do not pitch this as the novelty. It exists.

- **IBM, ALBERT on a 14nm analog AI inference chip.** 7.1M unique analog weights
  across 12 layers, programmed into the conductance of 28.3M devices. Average
  hardware accuracy **1.8% below the floating-point reference**, several tasks at
  full iso-accuracy. [Nature Communications](https://www.nature.com/articles/s41467-025-63794-4)
- **GPT-2 attention on analog gain cells.** A full software-to-hardware mapping of
  a *pretrained* model to non-traditional hardware, reported as reaching accuracy
  equivalent to GPT-2, with up to 5 orders of magnitude lower energy and 2 orders
  lower latency vs GPU. [arXiv 2409.19315](https://arxiv.org/pdf/2409.19315)
- **Post-training crossbar calibration on pretrained RoBERTa.** Per-crossbar
  conductance-range and DAC-range optimisation, no notable accuracy change across
  GLUE. [arXiv 2401.09859](https://arxiv.org/abs/2401.09859)

## 2. Runtime-variable precision — PUBLISHED, but by a different mechanism

This is the one that most directly overlaps the runtime dial, so read it first.

**Garg et al., "Dynamic Precision Analog Computing for Neural Networks,"**
IEEE JSTQE 2023 / [arXiv 2102.06365](https://arxiv.org/pdf/2102.06365).

They state the gap in the same terms we do — *"analog computing architectures
today only support a single, static precision, unlike digital architectures which
support programmable precision"* — and they also do per-layer precision on a
**pretrained** model without retraining weights, chosen at run time.

Their mechanism is **redundant coding**: run the same computation k times, in
different spatial channels or over multiple clock cycles, and average. Reported
89% energy reduction on ResNet50, 24% on BERT.

**Where a pulse-width dial differs, quantitatively.** Averaging k noisy repeats
cuts noise by √k, so it buys `0.5·log2(k)` bits for k× the time. A duration on a
clock of T ticks resolves T+1 levels, so it buys `log2(T)` bits for T ticks.

> For the same time budget, a pulse-width dial yields **twice the bits** that
> redundant averaging does — up to the point where the analog noise floor binds
> instead of the tick count.

That is a sharp, testable claim, and finding where the noise floor binds is
exactly what the loopback rig measures. [DERIVED — the arithmetic is ours; the
comparison should be checked against the paper's own noise model.]

## 3. ADC/DAC cost — THE central problem of the field

This is the most useful finding here, because it is where the work already sits.

- *"Peripheral circuits (ADCs, DACs, sample-and-hold stages, sense amplifiers)
  dominate both area and energy in memristive accelerator prototypes, with ADCs
  alone being the single largest contributor."*
- *"DAC-ADC noise dominates over other system-level noises."*
- Compute-aware SNR modelling that cuts ADC precision by 3 bits is reported to
  save **40–64× of ADC energy**.

[npj Unconventional Computing](https://www.nature.com/articles/s44335-025-00044-2)

An audio-jack loopback is a DAC, a wire, and an ADC. It is a bench model of the
exact component the field says dominates its cost. That is a far stronger framing
than "analog is fast," which is a claim every one of these companies already
believes harder than we do.

Related, and worth knowing: **ADC/DAC-free analog acceleration via frequency
transformation** already exists — computing in the frequency domain to avoid
conversion entirely. [arXiv 2309.01771](https://arxiv.org/pdf/2309.01771). This is
close to the "modes in the wire" idea and should be read before claiming it.

## 4. Continuous-time / all-layers-live execution — NO DIRECT HIT

Searched for continuous-time analog inference with no layer staging, all layers
settling together, on a transformer. The nearest neighbours are a different thing:

- Neural ODEs and continuous-depth networks — these *train* a continuous-depth
  model, rather than running a **pretrained discrete** network with every layer
  live and reading the answer when it settles.
- A memristive neural-ODE solver exists ([Science Advances](https://www.science.org/doi/10.1126/sciadv.adr7571)),
  again for continuous-time dynamics, not for collapsing a trained stack.
- Continual/streaming transformers address input streaming, not simultaneous
  layer execution, and report that multi-layer versions degrade because early
  layers change what later layers cached.

**This looks like the most open gap of the four.** [CONJECTURE — absence of
evidence from a snippet-only search is weak evidence of absence. Before claiming
novelty, search Google Scholar directly for "continuous-time inference pretrained
transformer analog" and check citations of the gain-cell paper.]

## 5. Commodity audio hardware as the analog substrate — NO HITS

No published work found using sound cards or audio codecs as the analog compute
medium for neural networks. Weak positive; the same caveat as above applies, and
"nobody published it" can also mean "nobody thought it was worth publishing."

## 6. Device-level problems — do not aim here

Conductance drift (stochastic, state-dependent drift exponent in PCM),
programming noise from read-write-verify schemes, device-to-device variability,
array parasitics. These are real and open, and they all need a fab. Nothing in
this repo can touch them.

---

## What this implies for positioning

The strong ground is **(3) with (2) as the sharp claim and (4) as the open one**:
work in the DAC/ADC-bounded regime the field says dominates its cost, show a
precision mechanism that is more tick-efficient than the published dynamic-
precision approach, and show a pretrained transformer surviving execution with no
layer staging.

The weak ground is speed. Mythic's chiplet holds ~30M parameters at 8-bit and
4-bit — static precision, which is exactly what Garg et al. name as the gap. A
dial is a contrarian position against both, and that is an interesting
conversation to walk into. "I beat a datacenter in my garage" is not.
