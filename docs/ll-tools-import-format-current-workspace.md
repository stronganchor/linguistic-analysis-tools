# LL Tools Import Format

Use one `CSV`, `JSON`, or `JSONL` file of metadata changes.

## Stable identifiers

- Keep stable identifiers in every row.
- Use `word_id` when possible.
- Use `recording_id` for recording fields when possible.
- If `recording_id` is unavailable, identify a recording with:
- `recording_slug + word_id/word_slug`
- or `recording_type + word_id/word_slug` when that word has exactly one recording of that type
- For full export zips:
- word IDs come from `words[].origin_id`
- recording IDs come from `words[].audio_entries[].origin_id`

## Editable fields

The importer accepts these editable fields:

- `word_title`
- `word_translation`
- `word_example_sentence`
- `word_example_sentence_translation`
- `recording_text`
- `recording_translation`
- `recording_ipa`
- `speaker_name`

## Update behavior

- Blank values mean `leave unchanged`.
- To delete a value, put the field name in `clear_fields`.
- Extra columns are ignored.
- You can start from an exported STT `metadata.csv` or `metadata.jsonl` file and only change the fields you want to fix.
- If you keep the STT export columns `text` + `text_field`, changing `text` updates whichever field `text_field` names:
- `recording_text`
- `recording_ipa`
- `recording_translation`
- `word_title`
- `word_translation`
- If the same field for the same item appears more than once, rows are processed top to bottom and the last changed value wins.

