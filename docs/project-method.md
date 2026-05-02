# How This Project Was Done

This project combined manual linguistic review, local STT training, AI-assisted
comparison, acoustic checking, dictionary support, and interlinearization. The
main lesson is that useful low-resource transcription work came from an
iterative review system, not from a single model run.

## Phase 1: Bootstrap From Manual IPA

The first loop was deliberately simple:

1. Manually transcribe a seed set of recordings in IPA.
2. Train a few STT models against that seed set.
3. Evaluate them on held-out examples.
4. Pick the best practical model.
5. Use that model to transcribe more audio.
6. Correct the new IPA predictions by hand.
7. Add the corrected rows to the training set.
8. Train again and repeat.

This bootstrapping loop mattered because the model did not need to be perfect at
first. It only needed to become good enough to reduce the cost of producing the
next batch of reviewed data.

The recurring evaluation metrics were character error rate and exact match, but
the final decision was not purely numerical. A model that made reviewable,
consistent mistakes could be more useful than a model with a slightly better
score but harder-to-diagnose behavior.

## Website Data And Review Layer

Language Learner Tools was the operational layer around this loop. The recorder
UI collected audio inside a wordset/category/recording-type structure, with
speaker metadata attached from the beginning. The export tools produced STT
training bundles with `metadata.csv`, `metadata.jsonl`, and audio files. Local
scripts in this repository trained models, generated predictions, and built
metadata update files. LL Tools then imported accepted changes back into the
site, usually marking imported IPA changes as needing Transcription Manager
review.

That website loop mattered because the model work was never detached from the
review surface. Reviewers could search IPA or orthography, filter to rows that
still needed review, use an IPA helper keyboard, inspect known-rule failures,
convert IPA to orthography after checking rules, and keep corrected data tied to
the same stable recording IDs used in later exports.

See [ll-tools-integration.md](ll-tools-integration.md) for a summary of the LL
Tools recorder, transcription manager, import/export, and automation surfaces.

## Phase 2: Train Multiple Useful Witnesses

After the first IPA models became useful, the project expanded beyond one
winner. Different models were trained or evaluated for different transcription
targets:

- IPA-target MMS
- IPA-target Whisper
- orthography-target MMS
- orthography-target Whisper
- auxiliary orthography witnesses such as VibeVoice ASR or prompted OpenAI
  transcription

The goal shifted from "which single model should we trust?" to "which collection
of witnesses gives the best evidence for a human or AI-assisted reviewer?"

That was especially important because IPA and practical orthography expose
different error patterns. An IPA model may preserve a narrow phonetic detail
that an orthography model normalizes away. An orthography model may produce
better word boundaries or phrase shapes when the IPA output is noisy.

## Phase 3: Multi-Witness STT Review

Once additional training stopped giving spectacular accuracy jumps, the main
workflow moved to five-witness review:

- two IPA witnesses
- three orthography witnesses

The IPA predictions were converted into a practical orthographic comparison
space so all witnesses could be compared against one another. The comparison
step produced reviewer-facing evidence rather than automatic truth.

Rows were easiest when all witnesses agreed or when disagreement was only a
trivial spelling/normalization difference. Rows stayed in review when:

- model families split against each other
- IPA and orthography agreed internally but disagreed after conversion
- only one witness supported a marked segment
- a model looked plausible but contradicted same-speaker family evidence
- the decision depended on narrow phonetic detail

For each row, the target output was a best-guess IPA transcription and a
best-guess practical orthography transcription, with enough evidence preserved
to explain why the guess was accepted or why it still needed human review.

## Phase 4: Standard Procedures For Known Failure Modes

The project accumulated standard checks for recurring problems. These checks
were important because a consensus vote is not enough when every model shares a
blind spot.

Examples of standard procedures:

- tie-bar misuse or missing affricate handling
- invalid aspiration marking
- overconfident laryngeal or pharyngeal notation
- confusing nearby vowels or rhotics
- deciding between sounds with waveform and audio evidence
- comparing a target row against same-speaker family recordings
- checking dictionary and category-image context before accepting a surface form

The practical rule was: use models for triage and evidence, but use targeted
phonetic procedures for decisions models are not qualified to make.

## Phase 5: Dictionary And Family Cross-Checks

Dictionary resources were not treated as automatic authorities. They were used
as witnesses:

- to find likely lexical forms
- to identify cognate or related word families
- to catch impossible-looking model outputs
- to support orthography decisions
- to prioritize manual review rows

Same-speaker family context was often more valuable than a raw dictionary match.
If a speaker used the same stem across several category recordings or sentence
contexts, that family evidence could support a transcription even when the
current row was noisy.

## Phase 6: Interlinearization

Reviewed orthography then fed into a local interlinearization prototype.

The interlinearizer combined:

- dictionary-backed token lookup
- a seed lexicon of approved analyses
- conservative morphology rules
- phrase memories
- confidence scores
- review-candidate reports
- a browser review UI for corrections

The important design decision was to keep uncertainty visible. Unknown or
low-confidence tokens were not forced into polished-looking output. They were
highlighted for review, and approved corrections were saved back into scoped
correction files for the next run.

The interlinear review loop followed the same philosophy as the STT loop:

1. Generate a best guess.
2. Preserve confidence and source evidence.
3. Correct the highest-value uncertain items.
4. Save approved corrections.
5. Regenerate.
6. Repeat until the output is stable enough to publish or use in lessons.

## What Made The Workflow Work

The useful pattern was not simply "use AI." It was:

- keep stable IDs and metadata attached to every recording
- separate IPA from orthography targets
- train and compare multiple model families
- convert outputs into a shared comparison space
- preserve model disagreement for review
- use acoustic analysis for narrow phonetic decisions
- use dictionaries as supporting evidence, not as automatic truth
- let human-approved corrections become durable training or review memory
- avoid importing unreviewed model output as if it were final

That combination made the system useful even when no single component was
perfect.
