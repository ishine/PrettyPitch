# Separating the vibrato

The F0 residual is split into two channels, `plain` and `vibrato`. It is a fixed procedure the dataloader runs every time. The model does not learn the split. The implementation is `prettypitch/decomp.py`.

![Vibrato separation](fig/vibrato_separation.png)

## Why split them

Vibrato and expression live on different time scales, and you want to control them in different units. Trained together, reaching for more vibrato moves the expression as well.

Split, they take separate strengths at synthesis time through `plain_scale` and `vib_scale`. Pass a per-frame array and the strength can change over the course of a phrase (`notebooks/03_f0_control.ipynb`).

## The steps

The residual is `residual = log2(F0) − baseline`.

1. **Soft clip** (`compress_residual`) Up to 2 semitones nothing happens; past that, tanh saturates the excess. The scoop into a vowel is a −5 semitone lurch, it carries energy in the vibrato band, and the filter rings on it and manufactures false vibrato. Holding the amplitude down first prevents that. Real vibrato, within ±2 semitones, is untouched.

2. **Band-pass** (`extract_vibrato`) A 4.5–7.5 Hz FIR runs through `filtfilt` for zero phase. The FIR is designed in seconds, not taps (0.3 s). The band is a physical quantity, so it does not depend on the frame rate.

3. **Gain** (`make_vibrato_gain`) A 0/1 curve is zeroed over every stretch that is neither a nucleus nor `pau`, over the margins around it, and around phoneme boundaries. Smoothing it with a σ = 0.05 s Gaussian gives a gain between 0 and 1. The margins are set separately for the two sides; the bundled configuration uses 0.05 s before and 0.1 s after. Right after a consonant comes the rise of a scoop, and right after a boundary comes the head of the next vowel. Both are pitch transitions, not vibrato.

4. **Subtract** `vibrato = gain × band-pass output`, `plain = residual − vibrato`.

## Reconstruction always holds

Whatever the gain removes is absorbed by `plain`, so

```
baseline + plain + vibrato ≡ log2(F0)
```

always holds. A poor split never damages the F0. All it decides is which channel the model learns a given movement in.

## Short phrases

`filtfilt` has a minimum sequence length, 3 × taps + 1. A 0.3 s FIR at 172.27 fps is 53 taps, so any phrase shorter than 0.93 s comes out with the vibrato all zero, and that movement stays in `plain`.

The dataset's `min_sec` follows from this constraint. The default is 1.7 s, which leaves a safe margin over the 0.93 s floor.

## Main constants

| Constant | Default | Meaning |
|---|---|---|
| `VIB_BAND_HZ` | (4.5, 7.5) | The vibrato band [Hz] |
| `FIR_TIME_LEN_SEC` | 0.3 | The length of the FIR [s] |
| `VIB_COMP_THRESH_ST` | 2.0 | Where the soft clip stops being the identity [semitones] |
| `GAIN_MARGIN_SEC` | 0.10 | How far around consonants and phoneme boundaries to zero the gain |
| `GAIN_SIGMA_SEC` | 0.05 | The smoothing σ of the gain curve |

Training and inference both use the values under `decomp:` in `configs/<recipe>.yaml`. The module defaults are kept in step with `configs/ja_example.yaml`.
