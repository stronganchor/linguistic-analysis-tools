# razhan Whisper / DOLMA Warm-Start Assessment

Date: `2026-04-14`

## Repos

- Model: [razhan/whisper-base-zza](https://hf.co/razhan/whisper-base-zza)
- Dataset: [razhan/DOLMA-speech](https://hf.co/datasets/razhan/DOLMA-speech)

## Main Conclusions

### `razhan/whisper-base-zza`

This is a real ASR model checkpoint, so it is technically usable as a warm-start source in our Whisper pipeline.

But it is a **low-priority experiment only**, not a strong base candidate.

Reasons:

- model card reports very poor eval:
  `WER 1.2574`, `CER 1.0132`
- training log shows only `5` total optimizer steps
- it was fine-tuned from `openai/whisper-base` on the tiny `zazaki` slice of `razhan/DOLMA-speech`
- tokenizer/config are standard Whisper; no dedicated `zza` language token was added
- `generation_config.json` still says `language: persian`, so the standalone checkpoint is misconfigured for direct inference

Important nuance:

- our own Whisper scripts explicitly set `forced_decoder_ids` with `language=None, task="transcribe"`
- so if we trained from this checkpoint inside our codebase, we would override the bad default generation prompt
- that makes it **technically loadable**, but not empirically promising

Best use case if tested at all:

- orthography-target Whisper experiment only
- not IPA-target
- probably only worth trying for a smaller / lighter secondary witness model, not as a mainline training branch

### `razhan/DOLMA-speech`

This is **not** a warm-start model. It is only a training dataset.

The `zazaki` slice is weak as a pre-adaptation source:

- train: `201`
- test: `50`
- mean duration: about `3.94s`
- train speaker mix: `178/201` clips from `SPEAKER_19`
- test speaker mix: `45/50` clips from `SPEAKER_19`
- all observed clips are `male`
- samples look like scripted/read children’s-story material, not broad conversational/narrative speech

So:

- it is too small to be a compelling standalone pretraining stage
- it is too speaker-skewed to be a strong acoustic warm-up set
- it is domain-mismatched relative to our C.K. narrative adaptation goals

## Recommended Priority

### Worth trying?

- `razhan/whisper-base-zza`:
  `maybe`, but low priority and orthography-only
- `razhan/DOLMA-speech` as standalone pre-adaptation stage:
  `no`
- `razhan/DOLMA-speech` as a tiny auxiliary dataset mixed into a larger orthography run:
  `maybe`, but low expected payoff

### Not recommended

- using either repo as a main candidate for `STT -> IPA`
- using `DOLMA-speech` as if it were a model
- prioritizing this over current stronger local bases and current C.K. adaptation branches

## Practical Recommendation

If we want one constrained experiment, the only one I would consider is:

1. start from `razhan/whisper-base-zza`
2. train on our approved orthography dataset only
3. compare against a direct `openai/whisper-base` warm start on the same manifest

That would answer the real question cleanly:

- does the tiny prior Zazaki orthography adaptation help at all beyond plain Whisper base?

I would not spend time on a `DOLMA-speech -> our data` two-stage pipeline unless the first direct comparison shows an actual gain.
