# guide — reference documents

**日本語**: [README.md](README.md)

How PrettyPitch works, one subject per document. The figures live in `fig/`.

## The whole system

| Document | Contents |
|---|---|
| [svs_pipeline.en.md](svs_pipeline.en.md) | Score to waveform. Which model does what, and what is learned against what is a fixed rule |

## F0

| Document | Contents |
|---|---|
| [f0_model.en.md](f0_model.en.md) | The F0 model. The three layers baseline, plain and vibrato, plus v/uv, and how the conditioning flows |
| [representative_pitch.en.md](representative_pitch.en.md) | Estimating the representative pitch: how a continuous MIDI value replaces the note pitch. It is used both to label the training data and as the target of the pitch correction |
| [vibrato_separation.en.md](vibrato_separation.en.md) | Splitting the residual into plain and vibrato: soft clip, band, gain |
| [pitch_correction.en.md](pitch_correction.en.md) | How the two correction stages work at synthesis time |

## Tools

| Document | Contents |
|---|---|
| [tuner.en.md](tuner.en.md) | The web app for tuning the vibrato extraction and the representative-pitch hyperparameters |

## Durations

| Document | Contents |
|---|---|
| [duration_model.en.md](duration_model.en.md) | The consonant-duration model and its post-processing, in detail |
