# Language Learner Tools Integration

This project depended on the Language Learner Tools WordPress plugin for the
data collection and review side of the workflow.

The plugin is not included in this repository, but the research scripts here
expect the same general data model:

- wordsets and categories
- words with stable IDs
- audio entries with stable recording IDs
- recording types such as isolation, introduction, question, or sentence
- speaker metadata
- recording-level IPA fields
- recording-level practical orthography fields
- review flags for transcriptions that need checking
- export/import bundles for batch processing

## Recording Collection

LL Tools provided the website surface for collecting recordings in context.
That mattered because recordings were tied to words, categories, recording
types, and speakers at capture time. The STT scripts could then work from
structured exports rather than from loose audio files.

That structure made it possible to:

- filter by speaker
- train only on reviewed recordings
- separate isolation words from phrases or questions
- compare related recordings in the same lexical family
- avoid accidentally mixing speakers in a speaker-specific model

## Transcription Review

The website also served as the review surface for IPA and orthography metadata.

The basic loop was:

1. Export LL Tools data.
2. Run STT or multi-witness scripts locally.
3. Build an import or review sheet.
4. Review or correct the transcription fields.
5. Import accepted metadata back into LL Tools.
6. Preserve review flags for anything uncertain.

The key principle was that model output should not silently bypass review. If a
recording's IPA or orthography changed, the row should remain traceable and
reviewable.

## Import And Export Format

The copied workspace note
[ll-tools-import-format-current-workspace.md](ll-tools-import-format-current-workspace.md)
describes the local import/update assumptions used during this project.

The reusable idea is simple:

- keep stable `word_id` and `recording_id` values
- update only the fields you intend to change
- leave blank values as "no change" when building metadata updates
- keep extra evidence columns for review even if the importer ignores them
- preserve review state rather than treating every batch update as final

## Why LL Tools Was Useful

For this project, LL Tools was not just a storage layer. It provided the missing
operational pieces around STT:

- a consistent way to gather recordings
- a way to organize recordings by lesson/category/wordset
- a reviewable transcription field for IPA
- a reviewable transcription field for orthography
- export bundles that local scripts could process
- import paths for accepted corrections
- a website-side view where learners and reviewers could benefit from the
  cleaned data

The scripts in this repository are easiest to adapt if a new project has a
similar structured export: audio files plus stable metadata linking recordings
to words, speakers, target transcription fields, and review state.
