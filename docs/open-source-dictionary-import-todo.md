# Zazacaogren Open-Source Import TODO

This note turns the open-source overlap review into a practical curation queue for `zazacaogren`.

The goal is not to bulk-import everything. The goal is to review source material line by line and only merge entries that add real value to the existing dictionary.

## Working rule

Only merge a candidate if it does at least one of these:

- adds a headword that is not already in `zazacaogren`
- adds a clearly different sense for an existing headword
- adds a useful idiom, proverb, fixed phrase, or multiword expression
- adds a high-value variant spelling or dialect form that users are likely to search for
- adds good etymology or lexical notes worth showing on an existing entry

Do not merge a candidate if it only does one of these:

- repeats an obvious translation of a common word already covered
- adds a bare synonym with no extra sense distinction or usage value
- adds noisy morphology, inflected forms, or spellchecker-only material
- adds historical or OCR-damaged material without clear modern value
- adds phrasebook content that belongs in a phrasebook, not the core dictionary

## Approximate overlap summary

These percentages are source-side overlap against the current `zazacaogren` export after light normalization.

- Kaikki / Wiktionary Zazaki: about `55%` overlap, about `467` likely novel forms
- ku.wiktionary etymology pages: about `76%` overlap, about `52` likely novel forms
- ku.wiktionary Persian/Kurmanci/Zazaki comparison page: about `88%` overlap, about `24` likely novel forms
- ku.wiktionary Hawrami -> Zazaki glossary: about `86%` overlap, about `2` likely novel forms
- Wikivoyage phrasebook: about `34%` overlap, about `54` likely novel items, but many are phrasebook-only
- ASJP Zazaki/Kirmanjki forms: about `35%` overlap, about `44` likely novel forms, but mostly basic list material
- Latham 1856 vocabulary: about `11%` overlap, but mostly historical spelling and legacy forms
- KurdishHunspell Zazaki: about `16%` overlap, but this is not an entry-ready dictionary; it is mostly a raw lexicon/morphology source

## Priority queue

### 1. Kaikki first

Reason:

- best balance of structure, modernity, and licensing
- largest pool of plausible new dictionary material
- likely to contain entries that can be reviewed and merged with less cleanup than the other sources

What to do:

- review Kaikki candidate headwords that are not already in `zazacaogren`
- prefer entries with usable glosses, multiple senses, or clearly distinct meanings
- ignore junk rows, formatting artifacts, and trivial duplicates

Merge rule:

- if a Kaikki entry is just another obvious translation of a common word already covered, skip it
- if it adds a missing lemma, missing sense, or useful lexical note, merge it

### 2. ku.wiktionary etymology pages second

Reason:

- smaller source, but high-value enrichment
- useful both for missing headwords and for improving existing entries

What to do:

- review novel headwords first
- then use the remaining pages to enrich existing `zazacaogren` entries with etymology notes

Merge rule:

- do not import etymology pages as plain duplicate dictionary entries
- use them to improve entry quality where they add something users do not already have

### 3. ku.wiktionary comparison page third

Reason:

- low incremental breadth, but useful for variant mapping
- may help with alternate spellings, synonym grouping, and search aliases

What to do:

- review the novel items
- then mine the page for variant forms of entries already in the site

Merge rule:

- add only if the form is a useful searchable variant or fills a real gap

### 4. KurdishHunspell later, as a mining source

Reason:

- huge raw lexicon
- very low clean overlap because it includes many forms that should not be direct public entries

What to do:

- do not bulk-import Hunspell rows into the public dictionary
- use it to generate a filtered review queue of clean lemmas absent from `zazacaogren`
- prefer nouns, verbs, adjectives, and proper lexical items
- exclude punctuation, digits, obvious inflections, and low-value morphology artifacts

Merge rule:

- only merge a Hunspell lemma after manual review against existing entries and after checking that it adds real user-facing value

### 5. Wikivoyage, ASJP, and historical sources only after the above

Reason:

- useful, but lower priority for the main dictionary

Wikivoyage:

- good for phrasebook or beginner-content expansion
- not a core dictionary source

ASJP:

- useful as a small core-vocabulary check
- not rich enough to drive major dictionary growth

Latham 1856 and similar historical items:

- useful for historical notes, legacy spellings, and occasional missing forms
- should be treated as manual curation only

## Suggested review workflow

### Phase 1: Kaikki review list

1. Build a candidate list of Kaikki forms not already in `zazacaogren`.
2. Normalize obvious spelling and punctuation noise.
3. Review candidates one by one.
4. Keep only entries that add a missing lemma, missing sense, idiom, or useful lexical note.
5. Record rejected rows so they are not reviewed repeatedly.

### Phase 2: Etymology enrichment pass

1. Review novel ku.wiktionary etymology headwords.
2. For overlapping entries, attach etymology only when it improves the existing entry.
3. Skip etymology notes that are too weak, too speculative, or too noisy.

### Phase 3: Variant and alias pass

1. Review the ku comparison page.
2. Add searchable aliases and variant spellings where they help users.
3. Do not create duplicate public entries just for a minor variant.

### Phase 4: Hunspell gap mining

1. Build a filtered lemma-only candidate list from Hunspell.
2. Remove punctuation, digits, inflected forms, and obvious morphology-only rows.
3. Review the remainder line by line.
4. Merge only the best candidates.

## Acceptance checklist for each candidate

- Is this already covered by an existing `zazacaogren` entry?
- If yes, does it add a different sense, not just another translation?
- Is the spelling clean enough for a public dictionary?
- Is the form likely to be searched by a learner or speaker?
- Is the source reliable enough for a public-facing entry?
- Does this belong in the core dictionary, or should it live in phrasebook/history/notes instead?

If the answer is mostly no, skip it.

## Immediate next tasks

- [ ] Build a Kaikki-only candidate review list of likely novel entries
- [ ] Review that list manually and keep only high-value additions
- [ ] Build a ku.wiktionary etymology enrichment list for existing entries
- [ ] Build a small ku comparison-page alias list
- [ ] Leave Hunspell for a later filtered-lemma pass, not a bulk merge

## Files to use

- Existing dictionary snapshot:
  `Dictionaries/zazacaogren-master-dictionary/ll-tools-dictionary-qa-pass-20260416.json`

- Open-source source inventory:
  `Dictionaries/open_sources_20260416/README.md`

- Highest-priority sources:
  `Dictionaries/open_sources_20260416/wikimedia/kaikki-zazaki/kaikki.org-dictionary-Zazaki.jsonl`
  `Dictionaries/open_sources_20260416/wikimedia/ku-wiktionary/etimolojiya-peyven-zazaki/`
  `Dictionaries/open_sources_20260416/wikimedia/ku-wiktionary/pevek-berhevdana-farisi-kurmanci-zazaki-inglizi.raw.wikitext`

- Later mining source:
  `Dictionaries/open_sources_20260416/kurdishhunspell-zazaki/zza/zza-Latn.dic`
