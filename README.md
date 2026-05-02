# Linguistic Analysis Tools

This repository collects reusable code and process notes for linguistic
analysis workflows. The initial material comes from a Zazaki speech-to-text,
transcription-review, and interlinearization project.

The work started with a small manually transcribed IPA corpus. We trained local
STT models, used the best model to transcribe more recordings, manually
corrected those predictions, rebuilt the training set, and repeated that loop.
Once the models reached a useful accuracy level and more data was no longer
producing large gains, the workflow shifted toward multi-witness review:

- two IPA-target STT witnesses
- three orthography-target STT witnesses
- comparison of witnesses in a shared orthographic space
- best-guess IPA and orthography proposals
- standard checks for recurrent phonetic and notation issues
- dictionary and same-speaker family cross-checks
- interlinear matching with confidence scores and human correction loops

The project was built around Central Zazaki / Genç-Palu material, but the
workflow is meant to be useful for other low-resource language projects where
the key problem is not just training a model, but building an evidence trail for
reviewable transcriptions.

## Repository Layout

```text
docs/
  project-method.md                  Narrative of the overall workflow
  ll-tools-integration.md             How Language Learner Tools fit the loop
  publication-checklist.md            What still needs review before public release
  *-current-workspace.md              Historical/current workspace notes

src/stt/
  Training, evaluation, serving, and batch transcription scripts

src/interlinearization/
  Local interlinearizer, correction memory, and review UI

src/dictionaries/
  Dictionary conversion, QA, and LL Tools import builders

src/acoustic_analysis/
  Acoustic review helpers, including vowel and bilabial-trill tools

examples/
  Small example manifests and interlinear review files
```

## Core Workflow

1. Record and export audio with stable metadata.
2. Manually transcribe a small seed set in IPA.
3. Train several STT model families.
4. Pick the best model by held-out CER/exact-match and practical review quality.
5. Use it to transcribe more audio.
6. Correct those predictions by hand.
7. Retrain and repeat until accuracy improvements flatten.
8. Switch to multi-witness review for bulk work.
9. Use model agreement, dictionary evidence, family context, and acoustic checks
   to produce best-guess IPA and orthography.
10. Feed reviewed orthography into interlinearization and correct the
    interlinear guesses through a lightweight review UI.

See [docs/project-method.md](docs/project-method.md) for the full project
narrative.

## Language Learner Tools

The upstream data collection and review surface was the Language Learner Tools
WordPress plugin. LL Tools provided structured audio collection, recording
types, speaker metadata, IPA and orthography fields, export/import bundles, and
review workflows on the website.

This repository does not include the full WordPress plugin. It documents the
integration points and keeps the reusable research-side scripts that consumed or
produced LL Tools-compatible data.

See [docs/ll-tools-integration.md](docs/ll-tools-integration.md).

## Data And Models

Large or sensitive artifacts are intentionally not included in normal git:

- raw audio and video
- full LL Tools exports
- trained model checkpoints
- extracted bundle directories
- virtual environments and caches
- large generated transcription outputs

For a public release, publish reusable model artifacts through a model host such
as Hugging Face or through GitHub Releases, and include only small
permission-safe examples in this repository.

## Status

This is an initial curation pass from a working research directory. Some scripts
still reflect their original project context and may need path/config cleanup
before they are turnkey for a new user.

Before making the repository public, work through
[docs/publication-checklist.md](docs/publication-checklist.md).
