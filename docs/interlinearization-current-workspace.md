# Zazaki Interlinearization Prototype

This folder is the first local pass at a FLEx/Paratext-style interlinear workflow for A.K.'s approved speech texts.

## Decision

Use a local command-line pipeline first, with optional FLEx export as a secondary path.

FLEx is a good manual linguistic workbench, especially for morpheme parsing and a growing approved-analysis inventory. It is not the best first automation surface for our current LL Tools data because the import path is mainly reliable for baseline text/free translation, while generated morpheme/gloss lines still need careful handling.

Paratext's Interlinearizer is useful as a model: make statistical guesses, let a human correct them, then learn from approved choices. Its workflow is tied to registered Paratext projects and verse-oriented translation/back-translation work, so it is not the natural home for A.K. story audio or future LL Tools display.

The local path keeps the useful parts:
- dictionary-backed suggestions from Harun/DEZD/LL Tools TSV exports
- a small editable seed memory for recurring approved analyses
- confidence and source labels on every token
- review-candidate TSVs for the next human correction loop
- JSON output that can later be rendered in LL Tools
- SFM output for FLEx experiments

## Morphology Rules

The interlinearizer now has a small rule layer before it gives up on a token:

- suffixes: `-a` feminine singular, `-n/-en/-ın/-in` imperfective on verbs, `-z/-iz` also, `-da/-d/-t` locative, `-ûn/-û` oblique plural, `-wo` w-buffered conjunction, plus conservative handling of `-o/-ê/-i/-ı`.
- prefixes: `nı-/nu-/n-` negative and `b-/bı-` subjunctive.
- verbal `-o` is treated as `1SG` in A.K. verb forms rather than masculine agreement.
- for verbs with separate present/subjunctive and past stems, the MORPH row shows the stem actually present in the surface form while the LEMMA row stays on the dictionary/citation form, usually the infinitive.
- orthography variants: `tp/db` are checked against `tw/dw` and `tu/du`; `we/wa/wı` forms are checked against `o/ö/û/ü/u` alternatives.
- final vowel raising: final `â`, `i`, and `ê` are also tried as lowered `a`, `ı`, and `e` when looking for the base.
- phrase memories can match through suffix stripping, so an approved expression such as `dûn pwırı şın` can be recognized in inflected surface forms.
- `ın/ına`-type demonstratives are tagged `DEM`; proximity is glossed separately as `PROX`, `PROX.M.SG`, or `PROX.F.SG`.
- `lirû` is now `lir-û`: `lir` "young" plus reduced oblique plural `-û`.
- `lıngûna` is `lıng-ûn-a`: `lıng` "foot", oblique plural `-ûn`, plus the closing marker in the `pê ... -a` circumpositional phrase.
- `rwe` and `hâri` are treated as nonverbal elements (`NVE`) when they participate in light-verb/multiword predicates.
- lexicalized multiword predicates such as `dûn pwırı şın`, `dûn hâri`, and `nûn rwe` stay visible word-by-word and also appear in the phrase list with their idiomatic meaning.
- `piyor/piya` are separated as `all` and `together`; the phrase `piyor piya` is also recognized as `all together`.
- `mıda` is analyzed as `mı-da` (`me-LOC`), with comitative "with me" force in the mill-going context.
- `ho/ha/hê` are treated as progressive or present-actuality particles (`PROG`) that make an ongoing present reading explicit with imperfective verbs.
- `beno` in `ğele xwe beno arye` is `be-n-o`: `take/carry-IPFV-1SG`.

## Files

- `auto_interlinearize.py`: command-line interlinearizer.
- `build_ak_category_texts.py`: builds one interlinear text per A.K. LL Tools category from the latest STT metadata export.
- `apply_ak_category_high_confidence_pass.py`: upserts approved high-confidence corrections from dictionary/context/image review of the generated A.K. category texts.
- `find_typo_candidates.py`: read-only likely-transcription-typo report; the refined pass skips pure surface-vowel/diacritic alternations because many are normal raising in this dataset.
- `lexicon_seed.tsv`: first-pass editable analysis memory.
- `review_server.py`: local browser review UI and save API.
- `review_ui/`: browser UI assets.
- `review_corrections.tsv`: saved review corrections. Draft rows preserve work-in-progress; approved rows are loaded by `auto_interlinearize.py`.
- `review_cell_notes.tsv`: notes tied to exact line/token/row/morpheme cells in the Text view. Resolved rows preserve the review history but no longer highlight the cell.
- `sources/ak_categories/*.tsv`: generated category-level source texts, one recording transcription per line.
- `sources/ak_category_texts_manifest.tsv`: manifest of generated A.K. category texts and coverage counts.
- `outputs/ak_category_high_confidence_pass_20260430.tsv`: audit of the broad auto-accepted category-text correction passes.
- `outputs/ak_uncertain_image_contact_20260430.jpg`: contact sheet used to inspect linked category images for ambiguous A.K. words.
- `outputs/*.interlinear.json`: structured line/token output.
- `outputs/*.interlinear.md`: human-readable aligned view.
- `outputs/*.tokens.tsv`: flat token table.
- `outputs/*.review_candidates.tsv`: low-confidence/unknown tokens sorted by frequency.
- `outputs/*.flex.sfm`: SFM export for FLEx experiments.

## Current Texts

The review UI now includes the two corrected A.K. story alignment bundles plus category-level texts from the April 28 STT metadata export:

- 119 category texts from A.K. recordings, one text per LL Tools category.
- 2,804 recording transcription lines; 2,188 reviewed lines and 616 `needs_review` lines.
- 7,570 category-text tokens; 6,572 matched; 5,989 high-confidence after the image-backed broad high-confidence correction pass and the first user-reviewed typo-candidate pass.
- Little Red Hen: 697 tokens; 697 matched; 689 high-confidence.
- Story of the Mouse: 407 tokens; 407 matched; 401 high-confidence.

The story texts remain the strongest starting point because repeated story vocabulary is better covered by the seed memory and dictionary. The category texts are broader vocabulary review surfaces; expect more yellow/red low-confidence cells there.

## Rerun

```bash
python3 interlinearization/auto_interlinearize.py \
  --alignment-tsv "Speech to Text IPA/training_data/ak-little-red-hen-edited-20260416-alignment/line_alignment.tsv" \
  --lesson-id ak_little_red_hen \
  --title "AK - Little Red Hen"

python3 interlinearization/auto_interlinearize.py \
  --alignment-tsv "Speech to Text IPA/training_data/ak-story-of-the-mouse-final-20260415-shorter-ending-alignment/line_alignment.tsv" \
  --lesson-id ak_story_of_the_mouse \
  --title "AK - Story of the Mouse"

python3 interlinearization/build_ak_category_texts.py

python3 interlinearization/apply_ak_category_high_confidence_pass.py
python3 interlinearization/build_ak_category_texts.py
```

## Review UI

```bash
python3 interlinearization/review_server.py
```

Open `http://127.0.0.1:8765/`.

The UI saves every review-panel edit to `review_corrections.tsv` as a draft. Use `Approve & Next` to mark a form as approved; approved rows override the first-pass seed memory during the next interlinearizer run. The Text view supports direct row/cell notes from any interlinear cell, saving exact cell notes in `review_cell_notes.tsv`. Yellow/red highlighted cells also offer an `Accept guess` action. MORPH cells keep morpheme editing through the same click menu, and existing hyphen buttons still delete a boundary directly. The `Regenerate` button reruns the selected text with approved review corrections loaded.

## Next Loop

1. Review high-frequency forms in the browser UI.
2. Approve confirmed forms so they become the next-run correction memory.
3. Regenerate and compare confidence counts.
4. Once stable, add an LL Tools renderer for the JSON schema rather than trying to make WordPress parse FLEx exports.
