# Dictionary Pipeline

This directory now has a review-first pipeline for Harun Turgut plus a shared LL Tools import builder.

## Harun review seed

Run:

```bash
python3 Dictionaries/build_harun_review_pipeline.py
```

Primary outputs under `Dictionaries/tsv_output/`:

- `harun_source_blocks.tsv`
- `harun_entries_review.tsv`
- `harun_priority_manifest.tsv`
- `harun_translation_memory.tsv`
- `harun_source_metadata.json`

Authoritative cleaned Harun outputs:

- Source-faithful corrected master review TSV: `Dictionaries/tsv_output/harun_entries_review.tsv`
- LL Tools-ready public import TSV: `Dictionaries/tsv_output/lltools_import/lltools_dictionary_harun_public.tsv`
- Older seed files such as `Palu - Bingöl Harun Turgut - corrected.tsv` are now superseded by the review-first outputs above.

Supporting review assets under `Dictionaries/tsv_output/harun_review/`:

- `harun_review_queue.xlsx`
- `page_batches/page_###.tsv`
- `page_images/harun_page_###.png`

Notes:

- Review starts from the original Harun TSV, not the machine-expanded corrected TSV.
- The old corrected TSV is used only for hints such as likely split blocks and affix/example clusters.
- Public Harun import should stay filtered to `review_status=verified_pdf`.

## Genç export

Run:

```bash
python3 Dictionaries/build_genc_dictionary_tsv.py
```

Primary outputs under `Dictionaries/tsv_output/`:

- `Genç Abdurahman Yilmaz ZAZACA TURKCE SOZLUK.tsv`
- `Genç Abdurahman Yilmaz ZAZACA TURKCE SOZLUK.summary.json`

Notes:

- This exporter is specific to the 2015 Bingöl University thesis PDF.
- It preserves the shared TSV columns used by the other dictionary exports and adds Genç-specific metadata such as `raw_pos`, `normalized_pos`, `source_abbreviations`, and `source_pdf_page`.
- Wrapped entry starts, source abbreviations before or after POS tags, and the small number of missing-colon OCR cases are normalized into one entry block per source headword.

## LL Tools import TSVs

Run:

```bash
python3 Dictionaries/build_lltools_dictionary_import_tsv.py
```

Outputs under `Dictionaries/tsv_output/lltools_import/`:

- `lltools_dictionary_harun_draft.tsv`
- `lltools_dictionary_harun_public.tsv`
- `lltools_dictionary_dezd_public.tsv`
- `lltools_dictionary_combined_draft.tsv`
- `lltools_dictionary_combined_public.tsv`
- `lltools_dictionary_sources.tsv`

`harun_public` only includes `verified_pdf` rows. The Harun review is now complete in the current local dataset, so `lltools_dictionary_harun_public.tsv` is the cleaned/corrected import-ready export.

## Local MT enrichment

Run:

```bash
python3 Dictionaries/enrich_dictionary_mt.py --limit 100
```

This updates `harun_translation_memory.tsv` in place using:

- `quickmt/quickmt-tr-en`
- `quickmt/quickmt-en-de`

The MT step is intentionally isolated from the public import builder. Review the MT fields before relying on them in site output.
