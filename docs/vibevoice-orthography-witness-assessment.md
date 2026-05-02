# VibeVoice ASR Orthography Assessment

Date: 2026-04-28

## Recommendation

`microsoft/VibeVoice-ASR` should be kept as an orthography-only STT witness for Zazaki.

Current role:

- use as a second or third opinion for A.K. orthographic transcription review
- prioritize it for another training run when the next larger reviewed orthography dataset is ready
- do not treat it as an IPA model
- do not make it the primary blind STT model yet

## Current Model

Base model:

- `%Z_MODELS_DIR%\speech_to_text_ipa\base\microsoft-vibevoice-asr`

Current LoRA adapter:

- `%Z_MODELS_DIR%\speech_to_text_ipa\experiments\vibevoice-asr-ak-orthography-lora-v001`

Training data:

- `%Z_WORKSPACE%\Speech to Text IPA\training_data\vibevoice_asr_ak_orthography_v001`
- A.K.-only, `speaker_user_id = 8`
- source manifest:
  `%Z_WORKSPACE%\Speech to Text IPA\training_data\genc-palu-recording_text-20260409-071931-filtered\manifest_text.tsv`
- train examples: `1711`
- eval examples: `404`

Training setup:

- LoRA / QLoRA
- 4-bit base-model loading
- `r=8`, `alpha=16`, dropout `0.05`
- 3 epochs
- SDPA attention, not FlashAttention

Training result:

- completed cleanly
- final train loss: `0.8948`
- runtime: about `1:08:19`
- durable log:
  `%Z_MODELS_DIR%\speech_to_text_ipa\logs\vibevoice-asr-ak-orthography-lora-v001-20260428-082256.log`

## Initial Held-Out Smoke Eval

Eval output:

- `%Z_MODELS_DIR%\speech_to_text_ipa\evaluations\vibevoice-asr-ak-orthography-lora-v001\eval_first25_predictions.jsonl`
- `%Z_MODELS_DIR%\speech_to_text_ipa\evaluations\vibevoice-asr-ak-orthography-lora-v001\eval_first25_summary.json`

First 25 held-out A.K. orthography eval rows:

| Model | Normalized WER | Normalized CER | Exact match |
| --- | ---: | ---: | ---: |
| MMS orthography checkpoint-1282 | `16.7%` | `4.3%` | `68%` |
| Whisper medium orthography checkpoint-855 | `26.9%` | `6.9%` | `44%` |
| VibeVoice ASR LoRA v001 | `27.8%` | `7.4%` | `52%` |

Interpretation:

- VibeVoice is not ahead of the current MMS orthography model.
- On this small smoke split it is roughly in the same broad range as the available Whisper medium orthography witness.
- It produces valid structured JSON output with timestamps and content.
- It sometimes gets rows right that MMS and/or Whisper miss, so it has practical review value as an additional orthography witness.

Useful examples:

- `Êyi qwim.` -> `Êyi qwim.`
- `Êya yo balışnawa.` -> `Êya yo balışnawa.`
- `Êya yo herra maya.` -> `Êya yo herra maya.`

Observed weaknesses:

- vowel and diacritic details, especially `ê/e`, `û/u`, `ı/i`
- word spacing and short function words
- occasional over-normalization toward more common-looking forms
- slower model startup than MMS, especially when the base model is read from the Windows model directory

## How To Use It Now

Use VibeVoice ASR when the task benefits from another orthographic witness, especially:

- MMS and Whisper disagree
- the row needs word-boundary evidence
- the row needs punctuation or capitalization suggestions
- the expected output is practical orthography, not IPA
- a reviewer wants a third independent model shape before deciding manually

Do not use it for:

- direct `recording_ipa`
- narrow phonetic decisions
- automatic imports without human review
- replacing the current MMS orthography model as the primary blind STT model

## Priority For Next Dataset

When the next larger reviewed orthography dataset is ready, VibeVoice should be a high-priority continuation candidate.

Recommended next run:

1. Run a full 404-sample eval for the current v001 adapter before changing anything.
2. Compare against MMS and Whisper on exactly the same eval manifest.
3. Train a new VibeVoice run on the larger reviewed orthography dataset.
4. Evaluate checkpoints during training instead of assuming the final checkpoint is best.
5. Try a larger LoRA rank, probably `r=16`, before trying much longer training.
6. Include controlled context fields such as word title, recording type, and speaker/dialect note, because VibeVoice may gain more from prompt context than MMS does.

The likely upside is not that VibeVoice immediately replaces MMS. The more realistic goal is a strong second or third opinion model that catches different orthographic errors and improves review triage.

## Reusable Scripts

Dataset builder:

- `%Z_WORKSPACE%\Speech to Text IPA\prepare_vibevoice_asr_dataset.py`

Training script:

- `%Z_WORKSPACE%\Speech to Text IPA\train_vibevoice_asr_lora_local.py`

A.K. orthography training launcher:

- `%Z_WORKSPACE%\Speech to Text IPA\run_vibevoice_asr_ak_orth_lora_train_wsl.sh`
- `%Z_WORKSPACE%\Speech to Text IPA\start_vibevoice_asr_ak_orth_lora_train_tmux.sh`

Eval script and launcher:

- `%Z_WORKSPACE%\Speech to Text IPA\evaluate_vibevoice_asr_lora.py`
- `%Z_WORKSPACE%\Speech to Text IPA\run_vibevoice_asr_ak_orth_eval_wsl.sh`

