# MMS-300m Zaza Orthography -> IPA

## Current Status

This experiment path is currently deprioritized.

What happened:

- `mms-300m-zza-cv25-orthography-v001` collapsed to near-blank decoding with best `CER 0.9988382899628253`
- `mms-300m-zza-cv25-orthography-v002-unfrozen-lr1e4` showed the same blank-collapse behavior
- `xls-r-300m-zza-cv25-orthography-v001-unfrozen-lr1e4` also collapsed the same way
- direct `xls-r-300m-ipa-genc-palu-v001-20260408` on the IPA dataset also failed, with best `CER 0.9488829988640667`

Current recommendation:

- do not continue this Common Voice orthography warm-start branch unchanged
- prefer the `MMS 1B` IPA path documented in `PROJECT_STATUS_AND_RESULTS.md`

This experiment has two stages:

1. Common Voice Zaza orthography warm-start
2. IPA fine-tuning on the reviewed A.K. Genç-Palu manifest

## Expected Common Voice archive

Place the Mozilla Data Collective archive here:

`%Z_WORKSPACE%\Speech to Text IPA\external-data\common-voice\common-voice-scripted-speech-25-0-zaza-4baca5e5.tar.gz`

If you have a Mozilla Data Collective API key and have already agreed to the dataset terms in the web UI, you can also download it with:

`python "%Z_WORKSPACE%\Speech to Text IPA\experimental-tools\download_mdc_dataset.py" --dataset-id cmn29lne80160o107scm9ustt --output-path "%Z_WORKSPACE%\Speech to Text IPA\external-data\common-voice\common-voice-scripted-speech-25-0-zaza-4baca5e5.tar.gz"`

That script reads the API key from `MDC_API_KEY`.

## Stage 1

Windows:

`%Z_WORKSPACE%\Speech to Text IPA\experimental-tools\train_mms_300m_zza_cv25_stage1.cmd`

WSL:

`$Z_WORKSPACE/Speech to Text IPA/experimental-tools/train_mms_300m_zza_cv25_stage1_wsl.sh`

Outputs:

- model: `%Z_MODELS_DIR%\speech_to_text_ipa\alternatives\mms-300m-zza-cv25-orthography-v001`
- manifest: `%Z_WORKSPACE%\Speech to Text IPA\training_data\common-voice-zza-25-orthography\manifest_text.tsv`

## Stage 2

Windows:

`%Z_WORKSPACE%\Speech to Text IPA\experimental-tools\train_mms_300m_zza_cv25_to_ipa_stage2.cmd`

WSL:

`$Z_WORKSPACE/Speech to Text IPA/experimental-tools/train_mms_300m_zza_cv25_to_ipa_stage2_wsl.sh`

Output:

- `%Z_MODELS_DIR%\speech_to_text_ipa\alternatives\mms-300m-zza-cv25-to-ipa-v001`

## Notes

- Stage 1 uses plain orthography targets and lowercases them.
- Stage 2 keeps the current IPA training/eval path unchanged, so the result can be compared directly against the current `189`-example eval split.
- The local `facebook-mms-300m` base checkpoint in this workspace is not adapter-enabled, so this branch does not pass `--target-lang`.
