# Recommended C.K. STT Models

Model storage note:

- these model roots now live under `%Z_MODELS_DIR%\speech_to_text_ipa`
- set `Z_STT_MODELS_DIR` if you want the scripts to use a different external model directory

These are the current best models for transcribing C.K.'s speech.

## IPA

- Use model root:
  `%Z_MODELS_DIR%\speech_to_text_ipa\alternatives\mms-1b-all-ipa-ck-tier1-ak-anchor-v001-20260410`
- Provenance:
  best checkpoint during training was `checkpoint-180`
- Practical result:
  this C.K.+A.K. anchor IPA model modestly beat the untouched base IPA model on the held-out C.K. eval split

### Secondary IPA Witness

- Whisper medium C.K. IPA witness:
  `%Z_MODELS_DIR%\speech_to_text_ipa\high-accuracy\whisper-medium-ipa-ck-ak-reviewed-v002-bf16-20260412-1631`
- Best checkpoint in that run:
  `checkpoint-288`
- Practical result:
  this Whisper medium IPA run did not beat the best C.K. MMS IPA model on `CER`, but it improved substantially over its first checkpoint and reached `5/11` exact matches on the held-out C.K. eval split, so it is worth keeping as a second-opinion witness for C.K. transcription review

## Orthography

- Use model root:
  `%Z_MODELS_DIR%\speech_to_text_ipa\alternatives\mms-1b-all-orthography-ck-tier1-ak-anchor-v001-20260411`
- Provenance:
  best checkpoint during training was `checkpoint-249`
- Practical result:
  this C.K.+A.K. anchor orthography model beat the untouched base orthography model on the held-out C.K. eval split
- Whisper-medium orthography note:
  the C.K.-focused Whisper medium orthography experiment `whisper-medium-orth-ck-ak-v001-bf16-20260412-1646` was stopped early and is not recommended as a second-opinion witness, because its held-out C.K. orthography eval stayed clearly worse than the MMS orthography model

## Important Note

For transcription, use the run root directories above rather than the checkpoint folders alone.

Reason:
- the checkpoint folders do not contain the full tokenizer/vocabulary bundle
- the run roots do contain the full bundle needed for inference
- the training runs were configured with `load_best_model_at_end`, so the run roots are the correct deployed inference targets

## A.K. Generalization Check

These C.K.-adapted models are better for C.K., but they are not better general models.

- On the held-out A.K. eval split, the C.K.-adapted IPA model was worse than the base IPA model
- On the held-out A.K. eval split, the C.K.-adapted orthography model was worse than the base orthography model

Practical consequence:

- use these adapted models specifically for C.K.
- keep the original A.K.-oriented base models for A.K. and broader mixed-speaker work
