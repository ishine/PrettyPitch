# How the F0 model is built

This model builds F0 from a score and a speaker. It generates three channels, not F0 itself.

![How the F0 model is built](fig/prettypitch_overview.svg)

## The sum of three layers

F0 is the sum of three layers in the log (log2) domain.

```
F0 = 2^(baseline + plain·σ_plain + vibrato·σ_vib)
```

![The three layers baseline / plain / vibrato, and v/uv](fig/f0_layers_furusato.png)

| Layer | Who makes it | What it holds |
|---|---|---|
| baseline | a fixed formula | The staircase of note pitch values, smoothed with a σ = 20 ms Gaussian. At training time the pitch is the representative value |
| plain (expression) | the model | Scoop, the fall at the end of a note, the transitions between notes, the wobble inside a note |
| vibrato | the model | Oscillation at 4.5 to 7.5 Hz |
| v/uv | the model | Voiced probability per frame. Binarised at 0.5 after generation |

The model does not generate baseline. Training and inference build it with the same formula. It is the ground the other layers sit on, and those layers learn the deviation from baseline. This ground is why the pitch roughly follows the score.

How plain and vibrato are split is described in [vibrato_separation.en.md](vibrato_separation.en.md).

## Encoder

It is a stack of ConvNeXt-1D blocks with no attention.

1. Embed the phoneme id into 64 dimensions. Pass the note pitch, a continuous forward-filled value, through a 1 → 32 linear layer. Concatenate both with the log duration, per phoneme.
2. The length regulator stretches each phoneme by its duration.
3. Concatenate a 0 → 1 ramp that gives the position inside the phoneme.
4. 1×1 conv → PReLU → 1×1 takes it to 128 dimensions, then four ConvNeXtBlock1d follow. The dilations are 1, 5, 12, 24.
5. LayerNorm and a 1×1 turn it into the 256-dimensional condition `c`.

Two terms that are constant over time are added to this. One is the speaker bank, 32 dimensions, projected to 256. The other is the style embedding, 256 dimensions. `style_mix` blends styles continuously, and a one-hot vector reproduces `style_id` exactly.

**Style moves the F0 very little.** Across the three styles of the bundled weights, the Ritsu kire, normal and soft banks, the generated F0 differs by a median of 3.5 cents, which is not audible. The three banks differ mainly in timbre; their pitch behaviour is close. Measured on the training data, the vibrato depth is about the same and only the amount of expression shrinks, by 13 to 17 per cent from kire to normal to soft. The embedding is trained: on held-out phrases, giving the correct style gives the lowest reconstruction error for all three.

## Denoiser

It is a DDPM that predicts x0 directly. The default is 10 steps with a cosine β schedule.

- The input is a noisy two-channel sequence, plain and vibrato
- A 1×1 conv takes it to 64 dimensions, then 16 GatedResBlock follow
- The dilations run 1, 2, 4, …, 128 twice
- The sum of the skip connections goes through 1×1 → PReLU → 1×1 and gives the two-channel x0

Every block takes the step embedding, 64 dimensions, and the condition `c`, 256 dimensions.

v/uv alone does not use the denoiser. It comes straight from the condition `c` through 1×1 → PReLU → 1×1 → σ.

## Choosing the sampler

`model.arch` switches it.

| arch | Default steps | Notes |
|---|---|---|
| `diffusion` | 10 | The default. The bundled weights use this. DDPM with direct x0 prediction |
| `reflow` | 1 | The same network. Integrates the velocity field with a single Euler step |

The network is the same. Only the sampler differs. reflow is slightly better on cent RMSE, but we judged by ear and made DDPM the default.

## Adjustment at synthesis time

```python
out = infer.synthesize_f0(model, cfg, score, plain_scale=1.2, vib_scale=0.5)
```

You can pass `plain_scale` and `vib_scale` as a scalar or as a per-frame array. `notebooks/03_f0_control.ipynb` shows how to vary them over the course of a phrase.

Two stages of pitch correction run after generation ([pitch_correction.en.md](pitch_correction.en.md)).

## The training targets

The targets are split in two, one for the vibrato output and one for the expression output. The representative value of each nucleus phoneme ([representative_pitch.en.md](representative_pitch.en.md)) builds the baseline. The residual splits into plain and vibrato. Those two channels are the diffusion target.

The dataloader computes all of this every time. The preprocessing shards do not store it. To change the decomposition parameters, just edit the config. You do not need to redo preprocessing.
