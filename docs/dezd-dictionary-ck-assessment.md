# DEZD Dictionary C.K. Assessment

Date: `2026-04-14`

## Verdict

`Manidar/DEZD-Dictionary-Kirmancki` is not a viable **STT base model** for our pipeline.

Reason:

- it is a Hugging Face **dataset**, not an acoustic ASR checkpoint
- it contains text dictionary entries, not speech/audio encoder weights
- so it cannot be warm-started into `Wav2Vec2/MMS` or `Whisper` the way our current STT training scripts expect

The viable use is narrower:

- use DEZD as a **lexicon / orthography prior** for CTC beam decoding
- possibly use it later for post-correction or phrase-ranking
- not as direct pretraining for `STT -> IPA`

## What Was Added

- DEZD unigram builder:
  [build_dezd_orthography_unigrams.py]($Z_WORKSPACE/Speech%20to%20Text%20IPA/build_dezd_orthography_unigrams.py)
- text-CTC beam/unigram eval support:
  [evaluate_wav2vec2_ctc_text_model.py]($Z_WORKSPACE/Speech%20to%20Text%20IPA/evaluate_wav2vec2_ctc_text_model.py)
- generated DEZD unigram list:
  [dezd_simple_unigrams.txt]($Z_WORKSPACE/Speech%20to%20Text%20IPA/external-data/dezd-dictionary-kirmancki-20260414/dezd_simple_unigrams.txt)
- build summary:
  [summary.json]($Z_WORKSPACE/Speech%20to%20Text%20IPA/external-data/dezd-dictionary-kirmancki-20260414/summary.json)
- apples-to-apples CK decode comparison:
  [summary.json]($Z_WORKSPACE/Speech%20to%20Text%20IPA/evals/ck-text-dezd-beam-compare-20260414/summary.json)

## CK Orthography Results

Held-out eval split:

- approved `C.K.+A.K. anchor` orthography eval manifest:
  [manifest_text_ck_plus_ak_anchor.tsv]($Z_WORKSPACE/Speech%20to%20Text%20IPA/training_data/ck-adaptation-tier1-20260410/manifest_text_ck_plus_ak_anchor.tsv)
- eval size: `10`

### Untouched A.K. base orthography MMS

- model:
  [checkpoint-1282]($Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-orthography-zza-simple-v001-20260409-071931/checkpoint-1282)
- greedy:
  `CER 0.28117`, exact `0.10`
- DEZD beam:
  `CER 0.28117`, exact `0.10`
- effect:
  one row changed only by spacing; aggregate metrics did not move

### Best C.K.-adapted orthography MMS

- model:
  [checkpoint-249]($Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-orthography-ck-tier1-ak-anchor-v001-20260411/checkpoint-249)
- greedy:
  `CER 0.20538`, exact `0.10`
- DEZD beam:
  `CER 0.20538`, exact `0.10`
- effect:
  no row changed at all

## Practical Conclusion

DEZD did not produce a measurable gain on the current held-out C.K. orthography eval.

So:

- do **not** treat `Manidar/DEZD-Dictionary-Kirmancki` as a candidate acoustic warm-start
- do **not** prioritize a DEZD-driven retraining branch for `STT -> ortho`
- if DEZD is used further, it should be for:
  `1.` larger-split lexicon/post-correction experiments
  `2.` phrase-ranking in mining/review tools
  `3.` possible CK wordlist support outside the acoustic trainer

## Existing First-Checkpoint Reference

The earlier C.K. orthography adaptation already shows that a first checkpoint is not a reliable decision point by itself:

- base native on held-out C.K. eval:
  [ck-text-base-native-20260411-0848/eval_metrics.json]($Z_WORKSPACE/Speech%20to%20Text%20IPA/evals/ck-text-base-native-20260411-0848/eval_metrics.json)
  `CER 0.28117`
- first saved C.K.+A.K. checkpoint `41`:
  [ck-text-tier1-checkpoint41-vs-base-20260411-0842/metrics.json]($Z_WORKSPACE/Speech%20to%20Text%20IPA/evals/ck-text-tier1-checkpoint41-vs-base-20260411-0842/metrics.json)
  `CER 1.0`
- later best checkpoint `249`:
  [ck-text-checkpoint249-20260411-1026/eval_metrics.json]($Z_WORKSPACE/Speech%20to%20Text%20IPA/evals/ck-text-checkpoint249-20260411-1026/eval_metrics.json)
  `CER 0.20538`

That earlier run recovered strongly after a very bad first checkpoint, which is another reason not to over-interpret a DEZD-style idea from the first save alone.
