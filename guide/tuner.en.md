# The tuner — seeing the training hyperparameters

A web app for tuning the hyperparameters of the vibrato extraction and the representative-pitch estimate. Both are **computed by the dataloader on every training step**, not fixed during preprocessing. Phrases are laid out one after another, and changing a value redraws them on the spot.

```bash
pip install -e ".[tuner]"
python vibrato_separation_tuner/app.py        # http://127.0.0.1:8017
```

`--port 8017` changes the port. It reads preprocessed datasets (`data/<db>/shard.npz` and `metadata.json`), so run `preprocess.run` first.

**What it draws is exactly what training computes.** It calls `prettypitch.decomp` and `prettypitch.pitch` directly, so what you see here is what the model is trained on.

There are two tabs.

## Vibrato separation

![The vibrato separation tab](fig/tuner_vibrato_separation.png)

This shows what the hyperparameters of the plain/vibrato split actually do ([vibrato_separation.en.md](vibrato_separation.en.md)).

The upper pane overlays the F0 and `plain`; the lower one shows the vibrato and the gain curve. Stretches where the gain is zero carry an orange band, so you can see at a glance how much has been excluded around consonants and phoneme boundaries.

| Setting | What it changes |
|---|---|
| band [Hz] | The band counted as vibrato. Default 4.5–7.5 |
| FIR window [s] | The length of the filter. The effective response and the minimum phrase length appear at the top right |
| cons / boundary margin | How far around consonants and phoneme boundaries the gain is zeroed, separately for each side |
| gain σ [s] | The smoothing of the gain curve |
| softclip | The threshold and strength that keep a scoop from becoming false vibrato |

`vib RMS` is the amplitude for that phrase, and `clip` is the fraction of frames the soft clip acted on. Once you settle on values, copy them into `decomp:` in `configs/ja_example.yaml`. To vary them per database, use `decomp.per_db`.

## Representative values

![The representative value tab](fig/tuner_pitch_representative.png)

This shows how the representative value of each nucleus phoneme is decided ([representative_pitch.en.md](representative_pitch.en.md)).

Horizontal lines over the F0 curve mark the representative value of each nucleus. The number above a line is the deviation from equal temperament in cents, and a nucleus where the arch rule fired is drawn green. The orange columns are the weights, so you can read which frames decided the value.

| Setting | What it changes |
|---|---|
| sigma_st | The σ of w_t. Smaller discards scoops more aggressively |
| w_d | What counts as "moving". `slope` (the default) is the magnitude of the F0 slope; `blaauw` is the deviation from a smoothed curve, that is, how jagged it is |
| edge window / lead / trail | The shape of the edge taper. It can be asymmetric |
| peak: max len / apex−median / prominence | The conditions under which the arch rule fires on a short note |
| peak band / offset | How much of the region near the apex is averaged |

The `preset` buttons switch between three settings. `real` is the plain setting that measures jaggedness, `slope` measures the slope instead, and `+ peak` adds the arch rule for short notes on top of that.

**`+ peak` is the setting the bundled weights were trained with.** It reproduces the values under `pitch:` in `configs/ja_example.yaml`.

## Shared controls

- **Data folder** starts out holding `data/`. Change it only to look somewhere else. Emptying it makes the app scan the directory you launched from, so do not clear it.
- **Scan DBs** rebuilds the list. Only directories that have both `shard.npz` and `metadata.json` appear, and you can select one at a time.
- **Order** switches between `alphabetical` and `shuffled`. **reshuffle** is a separate button that draws a new order. To find a particular phrase, type part of its name into **Search**.
- After changing a value, press **Run separation** or **Compute values** to redraw.

You never have to preprocess again. The shard holds only the raw phonemes, F0 and uv; both the decomposition and the representative values are computed here every time.
