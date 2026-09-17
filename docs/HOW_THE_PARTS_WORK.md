# How the parts work

Plain explanations of the pieces. Written to be re-read, not to persuade.

---

## 1. DAC and ADC

**DAC — digital to analog converter.** Number in, voltage out. One number at a
time, at its clock rate.

**ADC — analog to digital converter.** Voltage in, number out. Same deal.

That is the entire job. A converter has no idea what the numbers mean.

### The kinds, and why they differ

| type | how it works | where you find it |
|---|---|---|
| **R-2R ladder** | a resistor network, one switch per bit | simple DACs |
| **Current steering** | switched current sources, summed | video DACs, fast |
| **Sigma-delta** | oversamples, emits 1 bit very fast, filters it | audio codecs |
| **SAR** | binary search, one comparison per bit | microcontroller ADCs |
| **Flash ADC** | 2ⁿ comparators, one per level | RF, fastest, priciest |

### The one rule that explains the whole parts list

**Speed and resolution trade against each other at a fixed price.**

- Audio codec: **24 bits at 48–192 kHz.** Slow and fine.
- Video DAC: **8–10 bits at 150–400 MHz.** Fast and coarse.
- RF ADC: 8–14 bits at 100 MS/s–10 GS/s, and expensive.

Rough rule of thumb [EST]: every doubling of speed costs about a bit, for the
same money. This is why audio parts and video parts cost the same and differ by
3000× in rate — you are paying for the same silicon, spent differently.

## 2. The converter code vs the converter chip

This is the part worth being precise about.

**The chip** is a dumb pipe. It converts one number to one voltage, at a fixed
rate, forever. It cannot be taught anything.

**The code** decides what the numbers *mean*. Packing values into coefficients,
choosing the encoding, correcting the channel, reading it back — all of that is
software sitting on top of a dumb pipe.

So the converter you wrote is not competing with a converter chip. It sits above
one. And because it sits above one, **it ports**: change the chip, change the
sample rate and the band, and the encoding logic is unchanged.

## 3. What 1000 software converters did

They computed the *mathematics* of conversion, 1000 at a time. Each one produced
its own stream of numbers. That is real parallel work — it is the encoding, done
1000-wide, on hardware that can do 1000 things at once.

**Can they all output from the same output?**

It depends on one thing: **are their outputs meant to be added together?**

- **If yes — one output is enough.** Sum them in software and send one stream.
  This is not a compromise or a bottleneck. In this machine the bundle operation
  *is* addition, so summing 1000 converter outputs before the wire is performing
  the bundle, not losing it.
- **If no** — if they must arrive separately at separate destinations — then you
  need 1000 physical outputs, or you send them one after another in time.

One DAC produces one voltage at a time. But one voltage can be the sum of a
thousand things, and usually that is exactly what you wanted.

## 4. What an FPGA is

A chip full of blank logic you wire up yourself:

- **LUTs** (lookup tables) — tiny configurable truth tables, 4–6 inputs each.
  A LUT can be *any* logic function of its inputs.
- **Flip-flops** — one bit of memory each.
- **Programmable interconnect** — a switchable mesh joining them.
- **Block RAM** and **DSP slices** (hardware multipliers).
- **I/O pins** with configurable electrical standards.

You describe a circuit in Verilog or VHDL. The toolchain *places and routes* it
onto the fabric. What you get is **actual hardware**, not a program.

**Why it matters here:** every element runs every clock cycle, simultaneously.
A thousand counters on an FPGA are a thousand counters really counting at once —
not a loop pretending. That is a PWM generator array in the literal sense.

**The number that sets the scale is pin count.** A mid-range FPGA has roughly
100–200 usable I/O pins, each togglable at hundreds of MHz. So ~100–200 genuinely
parallel channels.

Rough cost tiers [EST]: Lattice iCE40 boards ~$25 · Xilinx Artix-7 on a Digilent
board ~$150–300 · large parts, thousands.

## 5. The 2 TB of DDR4-2666

**It is worth real money right now.** Server DRAM went up **80–110% in Q1 2026**
after Samsung, SK Hynix and Micron reallocated wafers to HBM. Current enterprise
DDR4 ECC RDIMM runs **$4.53/GB at the cheapest, $9.18/GB median** [SPEC, market
data]. 2 TB is therefore roughly **$9,000–$19,000**.

**What it is good for here:** holding a model. K3 at 4-bit is about 1.4 TB — it
*fits* in 2 TB, which very little consumer hardware can say.

**What it is not:** analog. DRAM is digital storage. It can be the place weights
live before being written into a substrate; it cannot be the substrate.

**The catch:** 2 TB needs a host with enough slots — a dual-socket server board
with 16–32 DIMM slots. DDR4-2666 pairs with first-gen Xeon Scalable or EPYC 7001.

## 6. NVMe — and why it is not Mythic's point

The parallelism is real but it is the **wrong kind**.

NVMe supports up to 65,535 queues of 65,536 commands. That is **I/O queue
parallelism** — many reads in flight at once. The drive's flash dies also operate
in parallel, which is where the throughput comes from.

Bandwidth: ~7 GB/s on PCIe 4.0 ×4. At 12 bits per value that is **~4.7 G
values/s** — genuinely comparable to coax at GHz rates.

**But nothing computes on the way out.** The flash stores bits; a controller
reads them; you get bytes. There is no arithmetic in the path.

Here is the distinction that matters:

> **Mythic's flash computes. NVMe's flash stores.**
>
> Mythic programs an analog voltage into a flash cell, and the cell's
> conductance performs a multiply when current passes through it. NVMe uses the
> same underlying technology to hold digital bits, which a controller reads back
> as numbers.

Same silicon family, completely different use. NVMe parallelism gets you data
*out* fast. It does not get you adds.

The exception: **computational storage** — SSDs with an FPGA on the drive. Those
exist and do put logic in the data path.

## 7. Rewriting kernels

**What you can control:** shared memory and register allocation in CUDA, access
patterns, cache-friendly tiling, NUMA placement, huge pages, and how much stays
resident in the compute unit. That last one is real and it is what the 4096
encoders already exploit — weights in registers are weights that never get
fetched.

**What you cannot control:** DRAM does not become analog by being addressed
differently. Compute-in-DRAM is a live research area (Ambit, SIMDRAM — using
charge sharing to do bitwise ops) but it needs modified hardware or deliberate
timing violations. It is not reachable from user code.

## 8. Ethernet cable

It is not internet, and it is not digital. **It is four twisted pairs of copper
in a jacket**, with controlled twist rates to reject noise. Nothing more.

What it "is" is decided entirely by the chips at each end. The same cable carries
Ethernet, analog video (VGA-over-Cat5 baluns), HDMI (over extenders), analog
audio, or DMX lighting.

**Bandwidth by category** [SPEC]:

| category | rated bandwidth |
|---|---|
| Cat5e | 100 MHz |
| Cat6 | 250 MHz |
| Cat6a | 500 MHz |
| Cat8 | 2000 MHz |

**The useful consequence:** Cat6a is **four shielded channels at 500 MHz for
about $15.** Critically sampled that is ~4 G values/s of raw analog capacity, in
one cheap cable — essentially four coax runs in one jacket, for less than one
coax run costs.

You cannot reach the Ethernet PHY's analog front end from software. So the way
to use this is to **drive the cable yourself with your own converters** and
ignore Ethernet as a protocol entirely. The cable is the asset, not the standard.
