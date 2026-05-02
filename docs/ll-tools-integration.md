# Language Learner Tools Integration

This project depended on the Language Learner Tools WordPress plugin for the
data collection and review side of the workflow.

Public plugin repository: <https://github.com/stronganchor/language-learner-tools>

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

## How The Pieces Fit

The website and this repository worked as two halves of one loop:

1. LL Tools collected audio and stored it with word, category, wordset,
   recording type, speaker, and review metadata.
2. LL Tools exported audio plus `metadata.csv` / `metadata.jsonl` for local STT
   and analysis work.
3. Scripts in this repository prepared manifests, trained or evaluated models,
   generated IPA and orthography predictions, compared witnesses, and built
   import/update files.
4. LL Tools imported accepted metadata updates back into the website while
   preserving review flags for uncertain rows.
5. The Transcription Manager and lesson/editor UIs supplied the human review
   surface before the next export, retrain, or interlinearization pass.

This let the website stay the operational source of truth while the heavier
model training, acoustic analysis, dictionary QA, and interlinearization scripts
stayed outside WordPress.

## Recorder UI

LL Tools provided the website surface for collecting recordings in context. The
`[audio_recording_interface]` shortcode renders the browser recorder for users
with recording access. The queue is built from words, images, prompt cards, and
missing `recording_type` entries inside the assigned wordset.

That structure made it possible to:

- filter by speaker
- train only on reviewed recordings
- separate isolation words from phrases or questions
- compare related recordings in the same lexical family
- avoid accidentally mixing speakers in a speaker-specific model

The recorder saves uploads as `word_audio` records with a parent word, recording
type, `speaker_user_id`, recording date, audio path, and optional processing
state. Prompt-card audio uses the same recorder UI but routes to prompt-card
metadata instead of creating ordinary word audio.

The upload handler also hardens the collection workflow:

- validates audio uploads before storing them
- deduplicates by word, recording type, speaker, and upload hash
- uses a short server-side lock to prevent rapid duplicate submits
- can auto-process recordings when configured
- can preserve original audio for wordsets that opt into that behavior

Manager-facing recorder tools live on the front-end wordset settings page rather
than requiring wp-admin. The `Recorder Access` tool assigns, upgrades, invites,
or unassigns recorder users for a wordset. The `Recorder Queues` tool lets a
manager inspect each assigned recorder's live queue and hide or restore items
for that recorder without leaving the wordset page.

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

## Transcription Manager

The LL Tools Transcription Manager is the main admin review surface for
recording-level text and transcription fields. It is wordset-scoped and includes
four related tabs:

- `Letter to IPA Map`: inferred letter-to-transcription mappings with manual
  overrides.
- `IPA Special Characters`: the symbol inventory used by a wordset plus helper
  keyboard characters.
- `Search`: written-text and transcription search with inline edits.
- `IPA to Orthography`: inferred conversion rules, contradiction review,
  approved exceptions, and bulk fill for missing written text.

The search tab is especially important for STT review. It can search written
text, transcription text, or both; filter to rows that need review; filter to
flagged typo/check-rule issues; and optionally require exact letters plus
diacritics. It also supports IPA context patterns and regex-style searches, so
reviewers can find recurring issues such as aspiration marking, tie-bar use, or
nearby sound confusions.

Inline edits update `recording_text` and `recording_ipa` directly on the
recording. The row stays visible until the reviewer intentionally changes the
filter, search, or page, which keeps long review queues stable. Review state can
also be re-flagged when a row is only partially fixed.

The IPA-to-orthography tab was useful after the project had enough reviewed
pairs. It learns candidate conversion rules from saved IPA/text pairs, lets the
reviewer hide bad auto rules or add manual overrides, surfaces contradictions,
and then bulk-fills missing written text only after the rules have been checked.

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

## Export Tools

The LL Tools export page provided three different research-facing export paths:

- `Export`: a full bundle zip with `data.json` plus media. Full mode includes
  words, audio, categories, images, source wordsets, and related metadata.
- `Export Word Text (CSV)`: a broad text export for dictionary, translation,
  and review work. It can include word text, translations, example sentences,
  and per-recording text, translation, and transcription fields.
- `Export STT Training Data`: the main bridge into this repository's STT
  scripts. It builds a zip with `metadata.csv`, `metadata.jsonl`, and an
  `audio/` folder.

The STT export lets the operator choose the training target field:
`recording_text`, `recording_ipa`, `recording_translation`, `word_title`, or
`word_translation`. It defaults to reviewed-only rows so recordings still marked
as needing transcription review are excluded from training unless the operator
intentionally widens the export.

The STT metadata includes the fields the local scripts need for filtering,
training, evaluation, and later imports:

- audio path and public audio URL
- target text plus `text_field`
- wordset, word, recording, category, and recording-type identifiers
- speaker user ID
- `recording_text`, `recording_translation`, and `recording_ipa`
- `review_status` / `needs_review`
- duration and MIME type

In this repository, scripts such as `prepare_lltools_ipa_dataset.py` and
`prepare_lltools_text_dataset.py` turn those exports into train/eval manifests.
Training and evaluation scripts then consume those manifests to build IPA-target
or orthography-target models.

## Import Tools

The metadata update importer is the return path from the research scripts back
to LL Tools. It accepts CSV, JSON, JSONL, or NDJSON and updates existing words
and recordings in place. It is intentionally conservative:

- it requires stable IDs where possible
- blank values mean "leave unchanged"
- `clear_fields` is required to delete values
- extra evidence columns are ignored
- `text` plus `text_field` can update whichever exported text field was used
- preview shows row counts, warnings, errors, and before/after examples before
  anything is applied
- undo is recorded through the import history
- imported IPA changes are marked as needing Transcription Manager review by
  default

That last point is central to the workflow. Model predictions, AI comparisons,
or batch corrections can be imported quickly, but they do not become invisible
"truth" just because they were imported. They re-enter the website review loop.

## Automation Surface

For larger Codex/operator runs, LL Tools also exposes REST automation under
`/wp-json/ll-tools/v1`. The import routes can preview a bundle, start an import
job, process it in batches, discard a paused import, and fetch durable result
JSON. This avoided fragile wp-admin form replay when moving generated bundles or
metadata-heavy updates between local analysis and a live site.

## Why LL Tools Was Useful

For this project, LL Tools was not just a storage layer. It provided the missing
operational pieces around STT:

- a consistent way to gather recordings
- a way to organize recordings by lesson/category/wordset
- manager-facing recorder assignment and live queue inspection
- a reviewable transcription field for IPA
- a reviewable transcription field for orthography
- a Transcription Manager for search, inline correction, IPA helper symbols,
  review flags, and IPA-to-orthography conversion
- export bundles that local scripts could process
- import paths for accepted corrections
- REST automation for repeatable operator and Codex workflows
- a website-side view where learners and reviewers could benefit from the
  cleaned data

The scripts in this repository are easiest to adapt if a new project has a
similar structured export: audio files plus stable metadata linking recordings
to words, speakers, target transcription fields, and review state.
