# Kimi units

Every result in this repo, restated for Kimi K3. Dollar per token/second, as
asked for.

Architecture is quoted from Moonshot's own repository, github.com/MoonshotAI/Kimi-K3:
2.8T total parameters, 104B activated per token, 93 layers (1 dense, 69 KDA +
24 Gated MLA), attention hidden dimension 7168, 96 heads, 896 experts with 16
selected and 2 shared, latent MoE dimension 3584, 3072 per expert, vocabulary
160K, context 1,048,576. **[cited]**

Everything below is arithmetic on those numbers plus numbers measured elsewhere
in this repo. Prices are marked **[est]** and are not measurements.

## The MoE shape is the friendly part

2.8T has to be **available**. Only 104B ever **flows** for a given token — 16
experts of 896. That split is the best possible shape for this machine, because
storage is the cheap axis and flow is the expensive one.

| stored, all 2.8T | |
|---|---|
| dial only, 12 bit | 4.20 TB |
| dial + lap, fp32-exact | 8.40 TB |
| fp32, for reference | 11.20 TB |

One drive. Note that dial+lap is *larger* than bf16 would be — it is exact
rather than compressed, which is the whole point of section 2 of
NOTES_AND_INTERVALS.

## What flows per token

| | K3 | gpt2 small |
|---|---|---|
| active weights | 104 B | 84.9 M |
| interval sets (params / d²) | **2,024** | 144 |

1,224x the active weights but only **14x the interval sets**, because each set
is 87x bigger. Width absorbs almost all of the growth.

## The flowing state

All 93 layers live at once, which is what the flow experiment says you can do:

| | |
|---|---|
| live notes | 93 x 7168 = 666,624 |
| glass at 100 positions/m | **6.67 km** |
| loss at 0.2 dB/km | 1.33 dB |

One spool holds the entire simultaneous state of Kimi K3.

## Streaming the weights: linear, no floor

Weights emitted by parallel cheap devices, each cycling through its share. One
USB4 drive at 40 Gbps emits 1.67 G dial+lap weights/s.

| target | drives | $/tok/s **[est]** |
|---|---|---|
| 1 tok/s | 62 | 6,240 |
| 100 tok/s | 6,240 | 6,240 |
| 1000 tok/s | 62,400 | 6,240 |

**$6,240 per token/second, flat at every scale.** Streaming is linear — no
economy of scale, and no floor either.

## Against the alternative

A GPU route has to hold all 2.8T in fast memory before it emits a single token.
At fp8 that is 2.8 TB of HBM, so 35 80-GB accelerators, so **$0.88 M before
token one**. **[est]**

| tok/s | streaming | GPU **[est]** |
|---|---|---|
| 1 | $6,240 | $875,000 |
| 10 | $6,240 | $87,500 |
| 100 | $6,240 | $8,750 |
| 1000 | $6,240 | $875 |
| 10000 | $6,240 | $438 |

The crossover is near 1,000 tok/s, and the reason is batching: a GPU amortises
each weight fetch across a batch, and streaming re-presents weights per token.

So this wins by ~140x in the regime where nobody batches — one person, one
stream, their own copy of a frontier model — and does not win at datacenter
scale. That is a position, not a verdict, and it is the axis already chosen:
cost and build time, not power.

## Stationary weights: the one-off

If the weights are stations rather than streams, nothing is re-presented and
the cost stops being per-token. The glass then sets the rate — listen =
4096/bandwidth, from section 8:

| tuning band | ceiling |
|---|---|
| 1 GHz | 0.24 M tok/s |
| 10 GHz | **2.44 M tok/s** |

Fill latency is 93 ticks, 38 us at 10 GHz, paid once rather than per token.

**Streaming cost is linear in tok/s. Stationary cost is a one-off.** Which one
is buildable is what the generator count in `experiments/weight_structure.py`
decides — a few dozen physical things per projection is a purchase, 51 M is
not. **[open]**
