# Data per dollar, by medium

Theoretical maximums only. Kimi units: **dollars per token/second on K3.**

Two tags you need to read the numbers:

- **[SPEC]** — from a datasheet or a published standard.
- **[EST]** — my estimate. Prices especially; street prices move.

The K3 figure is **6.4M activations/token [EST]**. Every row divides by it, so if
that number is wrong every row moves by the same factor and **the ranking does
not change.**

---

## 1. What a link is actually made of

Every one of these, without exception, is the same three parts:

```
numbers -> [ ENCODER ] -> [ DAC ] -> [ MEDIUM ] -> [ ADC ] -> [ DECODER ] -> numbers
             code           chip       wire         chip        code
             free           $$$        cents        $$$         free
```

- **Encoder / decoder** is software. Packing values into a waveform, correcting
  the channel, reading them back. This is the part already built, and it carries
  over unchanged to any medium below — only the sample rate and the band change.
- **DAC / ADC** is the converter. **This is the only part you actually buy.**
- **Medium** is wire or fibre. Effectively free.

So yes — "digital encoder to outputter to medium" is right, with one addition:
you need an **inputter** too.

## 2. Do you need an input as well as an output?

**Yes.** A DAC alone puts voltage on a wire and nothing reads it back. Something
has to measure what the medium did. That is the ADC.

**When you can loop:** if one device has both, you loop out-to-in on the same
box. A sound card does. An SDR does. A microcontroller does.

**When you can't:** a VGA port is output only — it has a DAC and no ADC. To
close that loop you buy a capture device, which is a separate ADC. That is why
the VGA row below has two prices.

## 3. What is analog and what is digital

This is the part that decides whether a cable is usable at all.

| interface | analog or digital | channels | notes |
|---|---|---|---|
| **Audio jack** | **ANALOG** | 2 | the codec in the machine is the converter |
| **VGA** | **ANALOG** | 3 (R,G,B) | 0–0.7 V per line. plus digital sync |
| **Component (YPbPr)** | **ANALOG** | 3 | same idea, on RCA plugs |
| **Composite** | **ANALOG** | 1 | ~6 MHz, one channel |
| **DVI-I** | both | 3 analog | DVI-D is digital only |
| **HDMI** | DIGITAL | — | TMDS pairs. the *cable* is 4 shielded pairs |
| **DisplayPort** | DIGITAL | — | but a DP→VGA adapter contains a DAC |
| **TOSLINK / optical audio** | **DIGITAL** | 2 | on/off light carrying PCM bits |
| **SPDIF coax** | DIGITAL | 2 | PCM bits over coax |
| **Ethernet** | DIGITAL at the plug | — | the PHY is analog inside, not reachable |
| **Plain coax (RG-6/58)** | neither | 1 | just a wire. whatever you drive it with |

Two that matter:

**TOSLINK is digital and runs at audio rates.** Optical out on old receivers
carries PCM at 48–192 kHz. It is not a fast optical channel — it is the audio
jack, in light. Used gear won't buy speed here.

**VGA is a real analog DAC and it is fast.** The STDP3150 DP-to-VGA converter
runs a **162 MHz pixel clock with a 10-bit video DAC** [SPEC]. Three channels.
A DP→VGA dongle is the cheapest high-rate analog output that exists.

## 4. Dollars per token/second

| option | $ | values/s | bits | K3 tok/s | **$/tok/s** |
|---|---|---|---|---|---|
| ADV7123-class video DAC + FPGA | 120 | 990,000,000 | 10 | 154.7 | **0.78** |
| DP→VGA adapter + VGA capture | 115 | 373,000,000 | 10 | 58.3 | **1.97** |
| Component video out + capture | 60 | 80,000,000 | 8 | 12.5 | **4.80** |
| Coax + 65 MS/s ADC+DAC modules | 60 | 65,000,000 | 12 | 10.2 | **5.91** |
| PlutoSDR (TX+RX, 56 MHz) | 230 | 112,000,000 | 12 | 17.5 | **13.14** |
| Composite video out + capture | 25 | 12,000,000 | 8 | 1.9 | **13.33** |
| Pi Pico (PWM out, 12-bit ADC in) | 4 | 500,000 | 12 | 0.1 | **51.20** |
| Bare I2S codec chips | 6 | 384,000 | 24 | 0.1 | **100** |
| TOSLINK, used gear | 30 | 384,000 | 24 | 0.1 | **500** |
| USB audio dongle | 10 | 96,000 | 16 | 0.0 | **667** |
| USB audio interface, 18ch @ 96k | 250 | 1,728,000 | 24 | 0.3 | **926** |
| USB audio interface, 192k stereo | 120 | 384,000 | 24 | 0.1 | **2,000** |
| Laptop sound card | 0 | 96,000 | 16 | 0.0 | free |

Prices are [EST]. Rates are [SPEC] except the coax module rate, which is [EST].

**Video parts beat audio parts by 300–1000× per dollar.** That is the headline,
and it is because a video DAC runs at 148 MHz where an audio DAC runs at 48 kHz,
while costing about the same.

## 5. Parts per option

**DP→VGA (cheapest real option)**
- DP or USB-C → VGA adapter, ~$15. Contains the DAC. 3 analog channels.
- VGA capture card, ~$100. Contains the ADC.
- VGA cable, ~$5.
- Code: render values as pixels, capture, read back.

**Coax**
- ADC module + DAC module, ~$30 each at 65 MS/s.
- Something to clock them — Pico or FPGA.
- RG-6 or RG-58 and connectors, a few dollars.

**Pico**
- Pico, $4. PWM out, 12-bit ADC in, PIO for exact timing. Loops on itself.
- Nothing else. This is the cheapest complete loop that exists.

**Bare codecs**
- PCM5102 DAC breakout ~$3, PCM1808 ADC breakout ~$3.
- A Pico or FPGA to drive I2S.
- Cheapest way to get *many* audio-rate channels — far below USB interfaces.

## 6. The bit-depth column matters

The dial is 12 bits. Video is 8–10.

Options: two pixels per value, or run the dial at 10 bits and carry the rest in
the lap counter. Not resolved — noted so it isn't forgotten.

## 7. Other things not yet costed

- Powerline (HomePlug) adapters — analog OFDM on mains wiring, chipset locked.
- Cable TV tuner cards — wideband analog front end, receive only.
- FPGA dev boards with ADC/DAC Pmods.
- DVI-I, which carries the same analog RGB as VGA on a different plug.
