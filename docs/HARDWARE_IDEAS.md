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
