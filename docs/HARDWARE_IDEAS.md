# Hardware ideas

Unevaluated. Saved so they don't have to be re-explained. Nothing here is
recommended, costed, or ruled out.

The point of all of them is the same: **many cheap analog channels rather than
one expensive one.**

## Parts worth looking at

- **Coax cable.** Bipolar, carries sign natively, cheap per meter.
- **A modem off Amazon** — or better, just the components out of one.
- **VGA / HDMI** — 6 channels.
- **DisplayPort** is all digital, but a DP-to-VGA adapter does what's needed.
  These are dirt cheap.
- **Arduino Pico or Nano running PIO.** ~$5 each.

## Already in hand, cost nothing

- The computer's sound card is already a converter. The encoder is **coded**,
  not bought.
- GPU: one encoder per compute unit, weights resident in the unit.

## Discussed elsewhere, not resolved

- **SSD parallelism.** 144k parallel processes, adds are free. The objection
  raised was weight transfer.

## Goal

A cheap, massive, analog LLM machine. Target: **Kimi K3.**

---

## osmo-fl2k — the cheap fast DAC, proven by other people

From Osmocom, the same people behind RTL-SDR. **$5 USB 3.0→VGA adapters** built on
the Fresco Logic FL2000 chip, driven as general-purpose DACs.

- **3 independent 8-bit channels** (R, G, B)
- **157 MS/s is the chip ceiling; ~84-100 MS/s is what people actually get**
  → **250-300 M values/s for $5**, so 50-60 M values/s per dollar
- **buy the cheap one.** $5-15 adapters advertised "USB 3.0 to VGA, 1920x1080"
  are FL2000. Expensive ones are DisplayLink and will not work at all. Brand
  is irrelevant, price is the tell. Needs a real USB 3.0 port; some units have
  the USB wired wrong and cap at USB-2 rates.
- **output only.** It cannot receive. The receive side here is the DHO804:
  12 bit, 1.25 GS/s, 25 Mpts -- better than an RTL-SDR (8 bit, 2.4 MS/s) or a
  HackRF (8 bit, 20 MS/s), and already owned.
- open source, working code: [osmocom/osmo-fl2k](https://github.com/osmocom/osmo-fl2k)
- [FL2K_2 fork](https://github.com/BM45/fl2k_2_rgb) drives all three channels;
  the original uses only red
- **the blanking-interval problem is already solved** — it picks timings that
  avoid HSYNC/VSYNC and produce a continuous sample stream. That was the killer
  in every earlier VGA-as-DAC attempt.

People transmit WBFM, GSM, UMTS and GPS with it. Fundamental to ~157 MHz,
harmonics usable to ~1.7 GHz.

### Cost per value/second

| | $/M values/s |
|---|---|
| **osmo-fl2k dongle** | **$0.011** |
| DP→VGA + capture card | $0.31 |
| ADALM2000 | $5.00 |
| Analog Discovery 3 | $7.90 |

~74 K3 tok/s of transport per $5 dongle, or about **$0.07 per tok/s** on the
output side.

### The matching input

fl2k is **TX only**. The cheap fast return path is the same trick reversed:
**USB video capture**, ~124 M pixels/s per channel, 3 channels, 8-bit. Same
repurposed video silicon, same bit depth.

Must be **uncompressed** (YUY2/RGB24, not MJPEG) and **4:4:4 or RGB**, not 4:2:2
— subsampling halves two of the three channels.

### Caveats

- **8-bit**, not 12. Two channels per value, or carry the rest in the lap counter.
- **CPU intensive** at high sample rates; the docs warn about single-board
  computers even with USB 3.0.
- Blanking intervals still exist on the capture side — usable as a free,
  hardware-guaranteed frame marker for alignment.

### What runs between them

**A VGA cable is already three 75 Ω coax runs** — R, G, B each have their own
mini-coax inside the jacket. For a plain loop that is the whole answer, ~$5.

Break out to separate BNC coax (VGA→3×BNC, ~$12) when you want to **put
something in a channel**: an attenuator, a tap, a filter, a splitter — or
different lengths per channel.

- **Termination matters.** At ~100 MS/s the cable is electrically long. VGA gear
  is 75 Ω throughout and terminates correctly on its own.
- **BNC comes in 50 Ω and 75 Ω and they look identical.** Video is 75 Ω. Mixing
  them causes reflections that will look like noise you cannot explain.
- **Delay:** at ~100 MS/s, **~2 m of coax = one sample of delay** (velocity
  factor 0.66). That is the conversion if cable length is ever used as timing.
