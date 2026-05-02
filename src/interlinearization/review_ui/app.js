const state = {
  texts: [],
  lessonId: "",
  lesson: null,
  queue: [],
  interlinear: [],
  filtered: [],
  cellNotes: [],
  selectedKey: "",
  filter: "needs",
  view: "text",
  saveTimer: null,
  lineSaveTimers: {},
  activeCellMenu: null,
};

const els = {
  workspace: document.querySelector(".workspace"),
  queuePanel: document.querySelector(".queue-panel"),
  reviewPanel: document.querySelector(".review-panel"),
  textPanel: document.getElementById("textPanel"),
  textLines: document.getElementById("textLines"),
  reviewViewBtn: document.getElementById("reviewViewBtn"),
  textViewBtn: document.getElementById("textViewBtn"),
  lessonSelect: document.getElementById("lessonSelect"),
  regenerateBtn: document.getElementById("regenerateBtn"),
  subtitle: document.getElementById("subtitle"),
  searchInput: document.getElementById("searchInput"),
  queueList: document.getElementById("queueList"),
  editor: document.getElementById("editor"),
  emptyState: document.getElementById("emptyState"),
  formTitle: document.getElementById("formTitle"),
  formMeta: document.getElementById("formMeta"),
  saveState: document.getElementById("saveState"),
  correctedFormInput: document.getElementById("correctedFormInput"),
  lemmaInput: document.getElementById("lemmaInput"),
  glossEnInput: document.getElementById("glossEnInput"),
  glossTrInput: document.getElementById("glossTrInput"),
  posInput: document.getElementById("posInput"),
  confidenceInput: document.getElementById("confidenceInput"),
  noteInput: document.getElementById("noteInput"),
  saveDraftBtn: document.getElementById("saveDraftBtn"),
  approveBtn: document.getElementById("approveBtn"),
  candidateGloss: document.getElementById("candidateGloss"),
  candidateDetails: document.getElementById("candidateDetails"),
  systemNote: document.getElementById("systemNote"),
  contexts: document.getElementById("contexts"),
};

const ABBREVIATION_GLOSSARY = {
  "1SG": "first-person singular - the speaker as one person; case labels such as DIR or OBL specify the pronoun form.",
  "1PL": "first-person plural - the speaker plus one or more others, roughly 'we'.",
  "2SG": "second-person singular - one addressee, roughly singular 'you'.",
  "2PL": "second-person plural - more than one addressee, roughly plural 'you'.",
  "2F.SG": "second-person feminine singular - one feminine addressee, roughly feminine singular 'you'.",
  "2M.SG": "second-person masculine singular - one masculine addressee, roughly masculine singular 'you'.",
  "3SG": "third-person singular - one referent outside the speaker/addressee pair.",
  "3PL": "third-person plural - more than one referent outside the speaker/addressee pair.",
  "3F.SG": "third-person feminine singular - one feminine referent, roughly 'she' or feminine 'it'.",
  "3M.SG": "third-person masculine singular - one masculine referent, roughly 'he' or masculine 'it'.",
  "F": "feminine - grammatical feminine gender.",
  "M": "masculine - grammatical masculine gender.",
  "F.SG": "feminine singular - one feminine referent or feminine agreement target.",
  "M.SG": "masculine singular - one masculine referent or masculine agreement target.",
  "M.SG.COP": "masculine singular copula - a linking 'be/is' element with masculine singular agreement.",
  "PL": "plural - more than one referent.",
  "PL.OBL": "plural oblique - a plural form in oblique case.",
  "DIR": "direct case - the unmarked/direct noun or pronoun form; not equivalent to subject in split-ergative systems.",
  "OBL": "oblique case - a non-direct noun/pronoun form used in governed, possessive, or postpositional environments.",
  "POSS": "possessive - marks a possessor relation, roughly 'my', 'your', 'his/her', or 'their'.",
  "REFL": "reflexive - refers back to a participant in the clause, roughly 'self'.",
  "REFL/POSS": "reflexive/possessive - a form used for self-reference and possessive relations.",
  "DAT": "dative - marks a recipient, goal, or indirect-object-like relation.",
  "LOC": "locative case - marks location, roughly 'in', 'at', or 'on'.",
  "NEG": "negative polarity - marks negation of the verb, predicate, or clause.",
  "IPFV": "imperfective aspect - presents an event as ongoing, habitual, present, or future rather than complete.",
  "IPFV.PL": "imperfective plural - imperfective aspect with plural agreement or plural-triggered vowel raising.",
  "IMPF": "imperfective aspect - presents an event as ongoing, habitual, present, or future rather than complete.",
  "SBJV": "subjunctive mood - marks desired, potential, commanded, or dependent action.",
  "OPT": "optative mood - marks wished or intended action, often 'let...' or a command-like form.",
  "COP": "copula - a linking element like 'be', 'is', or 'are'.",
  "COP.NEG": "negative copula - a negative linking element, roughly 'is not' or 'are not'.",
  "DEM": "demonstrative - a word like 'this' or 'that' pointing to a referent in discourse or space.",
  "PROX": "proximal deixis - marks something near the speaker or current discourse center, roughly 'this'.",
  "PROG": "progressive aspect - marks an event as ongoing at the reference time, roughly 'is/are doing'.",
  "EZ": "ezafe - an attributive/linking marker connecting a noun with a modifier or possessor.",
  "RAISE": "vowel raising - a phonological change where the final vowel is raised, often tied to plural or absorbed ezafe marking here.",
  "RAISED": "raised vowel - a surface form whose final vowel has been raised.",
  "AGR": "agreement - morphology that matches features such as person, number, or gender.",
  "ASP": "aspect - grammatical category describing the internal time shape of an event.",
  "MOOD": "mood - grammatical category marking speaker attitude such as command, wish, or possibility.",
  "POL": "polarity - grammatical positive or negative value of a clause or predicate.",
  "CASE": "case - grammatical marking of a noun or pronoun's role or relation.",
  "CL": "clitic - a bound grammatical element that behaves partly like a word and partly like an affix.",
  "CIRC": "circumpositional marker - one part of a two-sided adposition such as pê ... -a, where the relation is expressed around the noun phrase.",
  "NUM": "number - grammatical marking such as singular or plural.",
  "PHON": "phonological process - a sound-shape change rather than a lexical morpheme.",
  "BUF": "buffer consonant - a phonological consonant inserted between vowels rather than a lexical morpheme.",
  "DER": "derivational morphology - word-forming morphology that changes or extends a stem's lexical category or meaning.",
  "ADJZ": "adjectivizer - derivational morphology that forms an adjective or adjective-like word.",
  "N": "noun - a word used for a person, place, thing, or concept.",
  "V": "verb - a word used for an action, event, process, or state.",
  "V.PRT": "verbal particle - a particle-like element functioning with a verb.",
  "PRT": "particle - a short grammatical/discourse word that does not fit neatly into noun, verb, or adjective classes.",
  "PRON": "pronoun - a word standing for a noun phrase, such as 'I', 'you', or 'they'.",
  "DET": "determiner - a word that specifies a noun, such as 'this', 'that', or 'a'.",
  "POST": "postposition - an adposition that follows its complement, similar to English prepositions in function.",
  "PART": "particle - a short grammatical/discourse word that does not fit neatly into noun, verb, or adjective classes.",
  "ADV": "adverb - a word modifying a verb, adjective, clause, or another adverb.",
  "ADJ": "adjective - a word modifying or describing a noun.",
  "INTJ": "interjection - an exclamation or response word.",
  "CONJ": "conjunction - a word or clitic linking words, phrases, or clauses.",
  "PHRASE": "phrase - a multi-word expression analyzed as a unit.",
  "LEX": "lexical item - a stored dictionary-like word or expression.",
  "NVE": "nonverbal element - a predicate component that combines with a light verb or lexical verb.",
};

const ABBREVIATION_KEYS = Object.keys(ABBREVIATION_GLOSSARY).sort((left, right) => right.length - left.length);

const PREFIX_MORPHEME_HINTS = {
  "bı": { gloss: "SBJV", pos: "MOOD" },
  "b": { gloss: "SBJV", pos: "MOOD" },
  "nı": { gloss: "NEG", pos: "POL" },
  "nu": { gloss: "NEG", pos: "POL" },
  "nê": { gloss: "NEG", pos: "POL" },
  "ne": { gloss: "NEG", pos: "POL" },
  "n": { gloss: "NEG", pos: "POL" },
};

const SUFFIX_MORPHEME_HINTS = {
  "iz": { gloss: "also", pos: "CL" },
  "z": { gloss: "also", pos: "CL" },
  "da": { gloss: "LOC", pos: "CASE" },
  "de": { gloss: "LOC", pos: "CASE" },
  "d": { gloss: "LOC", pos: "CASE" },
  "t": { gloss: "LOC", pos: "CASE" },
  "ûn": { gloss: "PL.OBL", pos: "NUM" },
  "û": { gloss: "PL.OBL", pos: "NUM" },
  "ın": { gloss: "IPFV", pos: "ASP" },
  "in": { gloss: "IPFV", pos: "ASP" },
  "en": { gloss: "IPFV", pos: "ASP" },
  "n": { gloss: "IPFV", pos: "ASP" },
  "a": { gloss: "F.SG", pos: "AGR" },
  "wo": { gloss: "and", pos: "CONJ" },
  "o": { gloss: "1SG", pos: "AGR" },
  "ê": { gloss: "RAISE/EZ", pos: "PHON" },
  "i": { gloss: "RAISED/OBL", pos: "CASE" },
  "ı": { gloss: "OBL", pos: "CASE" },
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function displayGloss(token) {
  return token.corrected_gloss_en || token.display_gloss || token.gloss_en || "?";
}

function displayLemma(token) {
  return token.corrected_lemma || token.lemma || "?";
}

function displayPos(token) {
  return token.corrected_pos || token.pos || "?";
}

function displayForm(token) {
  return token.corrected_form || token.form || "";
}

function wordDisplayForm(token) {
  return `${token.prefix_punct || ""}${displayForm(token)}${token.suffix_punct || ""}`;
}

function morphemeSummary(token) {
  const morphemes = token.morphemes || [];
  if (!morphemes.length) return "";
  const forms = morphemes.map((m) => m.form).filter(Boolean).join("-");
  const glosses = morphemes.map((m) => m.gloss).filter(Boolean).join("-");
  if (forms && glosses) return `${forms} = ${glosses}`;
  return forms || glosses;
}

function correctionValue(form, field, fallback = "") {
  const correction = form.correction || {};
  if (correction[field]) return correction[field];
  const candidate = form.candidate || {};
  if (field === "note") return correction.note || "";
  if (field === "confidence") {
    return correction.confidence || (candidate.confidence ? String(Math.max(candidate.confidence, 0.97)) : "0.97");
  }
  if (field === "gloss_en") return correction.gloss_en || candidate.gloss_en || (candidate.display_gloss === "?" ? "" : candidate.display_gloss || "");
  return correction[field] || candidate[field] || fallback;
}

function statusFor(form) {
  return (form.correction && form.correction.status) || form.status || "unreviewed";
}

function queueGloss(form) {
  const correction = form.correction || {};
  if (correction.gloss_en || correction.gloss_tr) return correction.gloss_en || correction.gloss_tr;
  return displayGloss(form.candidate || {});
}

function queueFormLabel(item) {
  const corrected = item.correction?.corrected_form || item.candidate?.corrected_form || "";
  if (corrected && corrected !== item.form) return `${item.form} -> ${corrected}`;
  return item.form;
}

async function loadTexts() {
  const payload = await api("/api/texts");
  state.texts = payload.texts || [];
  els.lessonSelect.innerHTML = state.texts.map((text) => (
    `<option value="${escapeHtml(text.lesson_id)}">${escapeHtml(text.title)}</option>`
  )).join("");
  state.lessonId = state.texts[0]?.lesson_id || "";
  if (state.lessonId) {
    els.lessonSelect.value = state.lessonId;
    await loadState(state.lessonId);
  }
}

async function loadState(lessonId) {
  const payload = await api(`/api/state?lesson_id=${encodeURIComponent(lessonId)}`);
  state.lessonId = lessonId;
  state.lesson = payload.lesson;
  state.queue = payload.queue || [];
  state.interlinear = payload.interlinear || [];
  state.cellNotes = payload.cell_notes || [];
  const summary = state.lesson.summary || {};
  els.subtitle.textContent = `${state.lesson.title} · ${summary.matched_tokens || 0}/${summary.tokens || 0} matched · corrections save to ${payload.corrections_path}`;
  state.selectedKey = pickInitialSelection();
  render();
}

function pickInitialSelection() {
  const current = state.queue.find((item) => item.key === state.selectedKey);
  if (current && matchesFilter(current)) return current.key;
  const first = state.queue.find((item) => matchesFilter(item)) || state.queue[0];
  return first ? first.key : "";
}

function matchesFilter(item) {
  const filter = state.filter;
  const status = statusFor(item);
  const query = els.searchInput.value.trim().toLowerCase();
  if (query) {
    const haystack = `${item.form} ${queueGloss(item)} ${item.candidate?.lemma || ""}`.toLowerCase();
    if (!haystack.includes(query)) return false;
  }
  if (filter === "all") return true;
  if (filter === "needs") return status === "unreviewed";
  return status === filter;
}

function render() {
  renderView();
  if (state.view === "text") {
    renderText();
    return;
  }
  state.filtered = state.queue.filter(matchesFilter);
  renderQueue();
  renderEditor();
}

function renderView() {
  const textMode = state.view === "text";
  els.workspace.classList.toggle("text-mode", textMode);
  els.textPanel.classList.toggle("hidden", !textMode);
  els.reviewViewBtn?.classList.toggle("active", !textMode);
  els.textViewBtn?.classList.toggle("active", textMode);
}

function renderQueue() {
  if (!state.filtered.length) {
    els.queueList.innerHTML = `<div class="empty-state">No matching forms</div>`;
    return;
  }
  els.queueList.innerHTML = state.filtered.map((item) => {
    const status = statusFor(item);
    const active = item.key === state.selectedKey ? " active" : "";
    return `
      <button class="queue-item${active}" type="button" data-key="${escapeAttr(item.key)}">
        <span class="queue-main">
          <span class="queue-form">${escapeHtml(item.form)}</span>
          ${queueFormLabel(item) !== item.form ? `<span class="queue-correction">${escapeHtml(queueFormLabel(item))}</span>` : ""}
          <span class="queue-gloss">${escapeHtml(queueGloss(item))}</span>
          <span class="status ${escapeAttr(status)}">${escapeHtml(status)}</span>
        </span>
        <span class="queue-count">${item.count}</span>
      </button>
    `;
  }).join("");
  for (const button of els.queueList.querySelectorAll(".queue-item")) {
    button.addEventListener("click", () => {
      state.selectedKey = button.dataset.key;
      render();
    });
  }
}

function selectedForm() {
  return state.queue.find((item) => item.key === state.selectedKey) || null;
}

function renderEditor() {
  const form = selectedForm();
  if (!form) {
    els.editor.classList.add("hidden");
    els.emptyState.classList.remove("hidden");
    return;
  }
  els.emptyState.classList.add("hidden");
  els.editor.classList.remove("hidden");
  const candidate = form.candidate || {};
  const status = statusFor(form);
  els.formTitle.textContent = form.form;
  els.formMeta.textContent = `${form.count} occurrence${form.count === 1 ? "" : "s"} · ${status} · lowest confidence ${Number(form.lowest_confidence || 0).toFixed(2)}`;
  els.correctedFormInput.value = correctionValue(form, "corrected_form");
  els.lemmaInput.value = correctionValue(form, "lemma");
  els.glossEnInput.value = correctionValue(form, "gloss_en");
  els.glossTrInput.value = correctionValue(form, "gloss_tr");
  els.posInput.value = correctionValue(form, "pos");
  els.confidenceInput.value = correctionValue(form, "confidence");
  els.noteInput.value = correctionValue(form, "note");
  els.candidateGloss.textContent = displayGloss(candidate);
  const morph = morphemeSummary(candidate);
  els.candidateDetails.textContent = [
    displayLemma(candidate),
    displayPos(candidate),
    candidate.match_type || "unknown",
    candidate.confidence ?? 0,
    morph,
  ].filter(Boolean).join(" · ");
  els.systemNote.textContent = candidate.review_note ? `System note: ${candidate.review_note}` : "";
  els.saveState.textContent = form.correction?.updated_at ? `Saved ${form.correction.updated_at}` : "Not saved";
  els.saveState.className = "save-state";
  renderContexts(form);
}

function renderContexts(form) {
  els.contexts.innerHTML = form.contexts.map((context) => {
    const tokens = (context.tokens || []).map((token, index) => {
      const position = index + 1;
      const selected = context.token_start
        ? position >= context.token_start && position <= context.token_end
        : position === context.token_index;
      const low = Number(token.confidence || 0) < 0.84;
      const label = displayGloss(token);
      const morph = morphemeSummary(token);
      const title = morph ? `${label} · ${morph}` : label;
      return `<span class="token${selected ? " selected" : ""}${low ? " low" : ""}" title="${escapeAttr(title)}">${tokenLabel(token)}</span>`;
    }).join(" ");
    return `
      <article class="context">
        <div class="context-head">
          <span>Line ${escapeHtml(context.line_id)}</span>
          <span>${escapeHtml(timeRange(context))}</span>
        </div>
        <div class="context-text">${tokens}</div>
      </article>
    `;
  }).join("");
}

function tokenLabel(token) {
  const form = token.form || "";
  const corrected = token.corrected_form || "";
  if (corrected && corrected !== form) {
    return `${escapeHtml(token.prefix_punct || "")}${escapeHtml(form)}<span class="token-correction">->${escapeHtml(corrected)}</span>${escapeHtml(token.suffix_punct || "")}`;
  }
  return escapeHtml(wordDisplayForm(token));
}

function tokenMorph(token) {
  const morphemes = token.morphemes || [];
  if (!morphemes.length) return displayForm(token) || "?";
  return morphemes.map((m) => m.form).filter(Boolean).join("-") || displayForm(token) || "?";
}

function tokenMorphGloss(token) {
  const morphemes = token.morphemes || [];
  if (!morphemes.length) return displayGloss(token);
  return morphemes.map((m) => m.gloss).filter(Boolean).join("-") || displayGloss(token);
}

function tokenMorphemes(token) {
  const existing = Array.isArray(token.morphemes) ? token.morphemes : [];
  const morphemes = existing
    .map((m) => ({
      form: String(m.form || ""),
      gloss: String(m.gloss || ""),
      pos: String(m.pos || ""),
    }))
    .filter((m) => m.form || m.gloss || m.pos);
  if (morphemes.length) return morphemes;
  return [{
    form: displayForm(token) || "?",
    gloss: displayGloss(token) || "?",
    pos: displayPos(token) || "",
  }];
}

function tokenColumnCount(token) {
  return Math.max(1, tokenMorphemes(token).length);
}

function renderText() {
  if (!state.interlinear.length) {
    els.textLines.innerHTML = `<div class="empty-state">No interlinear lines available</div>`;
    return;
  }
  els.textLines.innerHTML = state.interlinear.map((line) => renderTextLine(line)).join("");
  bindLineEditors();
  bindNoteableCells();
  bindMorphemeEditors();
}

function renderTextLine(line) {
  const tokens = line.tokens || [];
  return `
    <article class="text-line">
      <div class="text-line-head">
        <span>Line ${escapeHtml(line.id)}</span>
        <span>${escapeHtml(timeRange(line))}</span>
      </div>
      <textarea class="text-line-original line-text-editor" data-line-id="${escapeAttr(line.id)}" rows="${lineEditorRows(line.text || "")}">${escapeHtml(line.text || "")}</textarea>
      <div class="interlinear-grid">
        <table class="interlinear-table">
          <tbody>
            ${renderSpanningRow("WORD", line, tokens, wordDisplayForm)}
            ${renderMorphemeRow("MORPH", line, tokens, "form")}
            ${renderSpanningRow("LEMMA", line, tokens, displayLemma)}
            ${renderMorphemeRow("GLOSS", line, tokens, "gloss")}
            ${renderSpanningRow("POS", line, tokens, displayPos)}
            ${renderPhraseRow(line, tokens)}
          </tbody>
        </table>
      </div>
    </article>
  `;
}

function lineEditorRows(text) {
  return Math.min(8, Math.max(2, String(text || "").split(/\n/).length + 1));
}

function renderSpanningRow(label, line, tokens, valueFn) {
  return `
    <tr>
      <th scope="row">${escapeHtml(label)}</th>
      ${tokens.map((token, index) => {
        const span = tokenColumnCount(token);
        const value = valueFn(token) || "?";
        return `<td class="noteable-cell token-span token-start token-end${certaintyClass(token, value)}${cellNoteClass(line.id, index + 1, label, "")}" colspan="${span}" ${cellReviewAttributes(line.id, index + 1, label, "", value)}>${renderTextCell(value, label)}</td>`;
      }).join("")}
    </tr>
  `;
}

function renderMorphemeRow(label, line, tokens, field) {
  return `
    <tr>
      <th scope="row">${escapeHtml(label)}</th>
      ${tokens.map((token, tokenIndex) => {
        const morphemes = tokenMorphemes(token);
        return morphemes.map((morpheme, morphIndex) => {
          const isStart = morphIndex === 0 ? " token-start" : "";
          const isEnd = morphIndex === morphemes.length - 1 ? " token-end" : "";
          const value = field === "gloss" ? morpheme.gloss || "?" : morpheme.form || "?";
          const hyphen = field === "form" && morphIndex < morphemes.length - 1
            ? `<button class="hyphen-delete" type="button" title="Delete morpheme boundary" aria-label="Delete morpheme boundary" data-line-id="${escapeAttr(line.id)}" data-token-index="${tokenIndex + 1}" data-morph-index="${morphIndex}">-</button>`
            : "";
          const classes = field === "form" ? "morph-cell editable" : field === "gloss" ? "morph-cell gloss-cell editable" : "morph-cell";
          const issueClass = certaintyClass(token, value);
          const noteClass = cellNoteClass(line.id, tokenIndex + 1, label, morphIndex);
          return `<td class="noteable-cell ${classes}${isStart}${isEnd}${issueClass}${noteClass}" ${cellReviewAttributes(line.id, tokenIndex + 1, label, morphIndex, value)}><span class="morph-text">${renderTextCell(value, label)}</span>${hyphen}</td>`;
        }).join("");
      }).join("")}
    </tr>
  `;
}

function certaintyClass(token, value = "") {
  const text = String(value ?? "").trim();
  const approved = token.review_correction?.status === "approved";
  const confidence = Number(token.corrected_confidence || token.confidence || 0);
  const unknown = !text || text === "?" || displayGloss(token) === "?" || displayLemma(token) === "?";
  if (unknown || token.match_type === "unknown") return " unknown-cell reviewable-cell";
  if (approved) return "";
  if (confidence > 0 && confidence < 0.84) return " low-certainty reviewable-cell";
  if (["dictionary_folded", "orthography_variant", "prefix_morphology", "suffix_morphology"].includes(String(token.match_type || ""))) {
    return " low-certainty reviewable-cell";
  }
  return "";
}

function cellReviewAttributes(lineId, tokenIndex, rowLabel, morphIndex, currentValue) {
  return [
    `data-line-id="${escapeAttr(lineId)}"`,
    `data-token-index="${escapeAttr(tokenIndex)}"`,
    `data-row-label="${escapeAttr(rowLabel)}"`,
    `data-morph-index="${escapeAttr(morphIndex)}"`,
    `data-current-value="${escapeAttr(currentValue)}"`,
  ].join(" ");
}

function noteKeyParts(note) {
  return [
    String(note.lesson_id || state.lessonId),
    String(note.line_id || ""),
    String(note.token_index || ""),
    String(note.row_label || ""),
    String(note.morph_index ?? ""),
  ];
}

function sameCellNote(note, lineId, tokenIndex, rowLabel, morphIndex) {
  const parts = noteKeyParts(note);
  return (
    parts[0] === String(state.lessonId)
    && parts[1] === String(lineId)
    && parts[2] === String(tokenIndex)
    && parts[3] === String(rowLabel)
    && parts[4] === String(morphIndex ?? "")
  );
}

function cellNoteFor(lineId, tokenIndex, rowLabel, morphIndex) {
  return state.cellNotes.find((note) => (
    sameCellNote(note, lineId, tokenIndex, rowLabel, morphIndex)
    && String(note.status || "open") !== "resolved"
  )) || null;
}

function cellNoteClass(lineId, tokenIndex, rowLabel, morphIndex) {
  return cellNoteFor(lineId, tokenIndex, rowLabel, morphIndex) ? " noted-cell" : "";
}

function upsertLocalCellNote(note) {
  const index = state.cellNotes.findIndex((existing) => sameCellNote(
    existing,
    note.line_id,
    note.token_index,
    note.row_label,
    note.morph_index ?? "",
  ));
  if (index >= 0) {
    state.cellNotes[index] = note;
  } else {
    state.cellNotes.push(note);
  }
}

function renderPhraseRow(line, tokens) {
  const phrases = (line.phrase_matches || [])
    .map((phrase) => {
      const start = Number(phrase.start_token);
      const end = Number(phrase.end_token);
      return { ...phrase, start, end };
    })
    .filter((phrase) => (
      Number.isInteger(phrase.start)
      && Number.isInteger(phrase.end)
      && phrase.start >= 1
      && phrase.end >= phrase.start
      && phrase.start <= tokens.length
    ))
    .sort((left, right) => left.start - right.start || left.end - right.end);

  if (!phrases.length) return "";

  const cells = [];
  let cursor = 1;
  for (const phrase of phrases) {
    if (phrase.start < cursor) continue;
    cells.push(...renderEmptyPhraseCells(tokens, cursor, phrase.start - 1));
    const end = Math.min(phrase.end, tokens.length);
    const span = tokenColumnRangeCount(tokens, phrase.start, end);
    const forms = (phrase.forms || []).join(" ");
    const pos = phrase.pos ? ` (${phrase.pos})` : "";
    const label = phraseLabel(phrase);
    const title = `${forms}: ${label}${pos}`;
    cells.push(`<td class="noteable-cell phrase-cell token-start token-end" colspan="${span}" title="${escapeAttr(title)}" ${cellReviewAttributes(line.id, phrase.start, "PHRASE", "", label)} data-note-form="${escapeAttr(forms)}">${renderWithAbbreviationTooltips(label)}</td>`);
    cursor = end + 1;
  }
  cells.push(...renderEmptyPhraseCells(tokens, cursor, tokens.length));

  return `
    <tr class="phrase-row">
      <th scope="row">PHRASE</th>
      ${cells.join("")}
    </tr>
  `;
}

function phraseLabel(phrase) {
  const lemma = String(phrase.lemma || "").trim();
  const gloss = String(phrase.display_gloss || phrase.gloss_en || "?").trim();
  if (lemma && lemma !== "?") return `${lemma} = ${gloss}`;
  return gloss || "?";
}

function renderEmptyPhraseCells(tokens, startToken, endToken) {
  const cells = [];
  for (let tokenIndex = startToken; tokenIndex <= endToken && tokenIndex <= tokens.length; tokenIndex += 1) {
    cells.push(`<td class="phrase-empty" colspan="${tokenColumnCount(tokens[tokenIndex - 1])}"></td>`);
  }
  return cells;
}

function tokenColumnRangeCount(tokens, startToken, endToken) {
  let count = 0;
  for (let tokenIndex = startToken; tokenIndex <= endToken && tokenIndex <= tokens.length; tokenIndex += 1) {
    count += tokenColumnCount(tokens[tokenIndex - 1]);
  }
  return Math.max(1, count);
}

function timeRange(context) {
  if (!context.start_sec || !context.end_sec) return "";
  return `${context.start_sec}s-${context.end_sec}s`;
}

function collectCorrection(status) {
  const form = selectedForm();
  if (!form) return null;
  return {
    lesson_id: state.lessonId,
    form: form.form,
    corrected_form: els.correctedFormInput.value,
    lemma: els.lemmaInput.value,
    gloss_en: els.glossEnInput.value,
    gloss_tr: els.glossTrInput.value,
    pos: els.posInput.value,
    confidence: els.confidenceInput.value,
    note: els.noteInput.value,
    status,
    line_ids: form.contexts.map((context) => context.line_id),
  };
}

function sameForm(left, right) {
  return String(left || "").trim().toLowerCase() === String(right || "").trim().toLowerCase();
}

function applyCorrectionToToken(token, correction) {
  if (!sameForm(token.form, correction.form)) return;
  token.review_correction = correction;
  token.corrected_form = correction.corrected_form || "";
  const morphemes = parseCorrectionMorphemes(correction);
  if (morphemes.length) {
    token.morphemes = morphemes;
  }
  if (correction.status === "approved") {
    token.corrected_lemma = correction.lemma || "";
    token.corrected_gloss_en = correction.gloss_en || "";
    token.corrected_gloss_tr = correction.gloss_tr || "";
    token.corrected_pos = correction.pos || "";
    token.corrected_confidence = Number(correction.confidence || 0.97);
  }
}

function parseCorrectionMorphemes(correction) {
  const value = correction?.morphemes;
  if (!value) return [];
  const parsed = Array.isArray(value) ? value : safeJsonParse(value, []);
  if (!Array.isArray(parsed)) return [];
  return parsed
    .map((m) => ({
      form: String(m.form || ""),
      gloss: String(m.gloss || ""),
      pos: String(m.pos || ""),
    }))
    .filter((m) => m.form || m.gloss || m.pos);
}

function safeJsonParse(value, fallback) {
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function applySavedCorrection(correction) {
  for (const line of state.interlinear) {
    for (const token of line.tokens || []) {
      applyCorrectionToToken(token, correction);
    }
  }
  for (const item of state.queue) {
    if (sameForm(item.form, correction.form)) {
      item.correction = correction;
      item.status = correction.status;
      applyCorrectionToToken(item.candidate || {}, correction);
    }
  }
}

async function saveCorrection(status) {
  const payload = collectCorrection(status);
  if (!payload) return;
  els.saveState.textContent = "Saving";
  els.saveState.className = "save-state saving";
  try {
    const result = await api("/api/corrections", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    const form = selectedForm();
    if (form) {
      form.correction = result.correction;
      form.status = result.correction.status;
    }
    applySavedCorrection(result.correction);
    els.saveState.textContent = `Saved ${result.correction.updated_at}`;
    els.saveState.className = "save-state";
    renderQueue();
    if (state.view === "text") renderText();
  } catch (error) {
    els.saveState.textContent = error.message;
    els.saveState.className = "save-state error";
  }
}

function findLine(lineId) {
  return state.interlinear.find((line) => String(line.id) === String(lineId)) || null;
}

function findToken(lineId, tokenIndex) {
  const line = findLine(lineId);
  const index = Number(tokenIndex) - 1;
  if (!line || index < 0) return null;
  return (line.tokens || [])[index] || null;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function splitAtCharacter(value, offset) {
  const chars = Array.from(value);
  return [chars.slice(0, offset).join(""), chars.slice(offset).join("")];
}

function stripOuterGlossPart(gloss, part, side) {
  if (!gloss || !part || gloss === "?") return gloss || "?";
  if (side === "prefix" && gloss.startsWith(`${part}-`)) {
    return gloss.slice(part.length + 1) || "?";
  }
  if (side === "suffix" && gloss.endsWith(`-${part}`)) {
    return gloss.slice(0, -(part.length + 1)) || "?";
  }
  return gloss;
}

function morphemeHint(form, kind, token) {
  const key = String(form || "").toLowerCase();
  if (kind === "prefix") return PREFIX_MORPHEME_HINTS[key] || null;
  if (key === "o" && displayPos(token) !== "V") {
    return { gloss: "M.SG.COP", pos: "COP" };
  }
  return SUFFIX_MORPHEME_HINTS[key] || null;
}

function inferSplitMorphemes(morpheme, left, right, token, morphIndex = 0) {
  const leftAsSuffix = morphemeHint(left, "suffix", token);
  if (morphIndex > 0 && leftAsSuffix) {
    const rightHint = morphemeHint(right, "suffix", token);
    return [
      { form: left, gloss: leftAsSuffix.gloss, pos: leftAsSuffix.pos },
      { form: right, gloss: rightHint?.gloss || stripOuterGlossPart(morpheme.gloss, leftAsSuffix.gloss, "prefix"), pos: rightHint?.pos || morpheme.pos || displayPos(token) },
    ];
  }
  const prefixHint = morphemeHint(left, "prefix", token);
  if (prefixHint) {
    return [
      { form: left, gloss: prefixHint.gloss, pos: prefixHint.pos },
      {
        form: right,
        gloss: stripOuterGlossPart(morpheme.gloss, prefixHint.gloss, "prefix"),
        pos: morpheme.pos || displayPos(token),
      },
    ];
  }
  const suffixHint = morphemeHint(right, "suffix", token);
  if (suffixHint) {
    return [
      {
        ...morpheme,
        form: left,
        gloss: stripOuterGlossPart(morpheme.gloss, suffixHint.gloss, "suffix"),
      },
      { form: right, gloss: suffixHint.gloss, pos: suffixHint.pos },
    ];
  }
  const glossParts = String(morpheme.gloss || "").split("-");
  if (glossParts.length >= 2) {
    return [
      { ...morpheme, form: left, gloss: glossParts.shift() || "?" },
      { form: right, gloss: glossParts.join("-") || "?", pos: morpheme.pos || displayPos(token) },
    ];
  }
  return [
    { ...morpheme, form: left },
    { form: right, gloss: "?", pos: morpheme.pos || displayPos(token) },
  ];
}

function glossForMorphemes(morphemes) {
  return morphemes.map((m) => m.gloss).filter(Boolean).join("-");
}

function mergeGloss(left, right) {
  return [left, right].filter(Boolean).join("-");
}

function noteForMorphemeEdit(token) {
  const existing = token.review_correction?.note || "";
  if (existing) return existing;
  const note = token.review_note || "";
  return note ? `${note} Morpheme segmentation adjusted in the Text view.` : "Morpheme segmentation adjusted in the Text view.";
}

async function saveMorphemeEdit(token, lineId, morphemes) {
  token.morphemes = morphemes;
  const gloss = glossForMorphemes(morphemes) || displayGloss(token);
  const result = await api("/api/corrections", {
    method: "POST",
    body: JSON.stringify({
      lesson_id: state.lessonId,
      form: token.form,
      corrected_form: token.corrected_form || "",
      lemma: displayLemma(token),
      gloss_en: gloss,
      gloss_tr: token.corrected_gloss_tr || token.gloss_tr || "",
      pos: displayPos(token),
      confidence: String(Math.max(Number(token.corrected_confidence || token.confidence || 0), 0.97)),
      note: noteForMorphemeEdit(token),
      status: "approved",
      line_ids: [lineId],
      morphemes,
    }),
  });
  applySavedCorrection(result.correction);
  renderText();
  renderQueue();
  if (selectedForm()) renderEditor();
  await regenerateCurrentText();
}

async function regenerateCurrentText() {
  const lessonId = state.lessonId;
  const selectedKey = state.selectedKey;
  const view = state.view;
  els.subtitle.textContent = `${state.lesson?.title || lessonId} · saved; regenerating interlinear output`;
  const result = await api("/api/regenerate", {
    method: "POST",
    body: JSON.stringify({ lesson_id: lessonId }),
  });
  if (!result.ok) {
    throw new Error(result.stderr || result.stdout || "Regeneration failed");
  }
  await loadState(lessonId);
  state.view = view;
  state.selectedKey = selectedKey;
  render();
}

async function splitMorphemeAtCell(cell, event) {
  const token = findToken(cell.dataset.lineId, cell.dataset.tokenIndex);
  if (!token) return;
  const morphemes = tokenMorphemes(token);
  const morphIndex = Number(cell.dataset.morphIndex);
  const morpheme = morphemes[morphIndex];
  const form = morpheme?.form || "";
  const chars = Array.from(form);
  if (chars.length < 2) return;
  const textEl = cell.querySelector(".morph-text");
  const rect = textEl ? textEl.getBoundingClientRect() : null;
  const ratio = rect && rect.width ? (event.clientX - rect.left) / rect.width : 0.5;
  const offset = clamp(Math.round(ratio * chars.length), 1, chars.length - 1);
  const [left, right] = splitAtCharacter(form, offset);
  const split = inferSplitMorphemes(morpheme, left, right, token, morphIndex);
  const next = [
    ...morphemes.slice(0, morphIndex),
    ...split,
    ...morphemes.slice(morphIndex + 1),
  ];
  await saveMorphemeEdit(token, cell.dataset.lineId, next);
}

async function deleteMorphemeBoundary(button) {
  const token = findToken(button.dataset.lineId, button.dataset.tokenIndex);
  if (!token) return;
  const morphemes = tokenMorphemes(token);
  const morphIndex = Number(button.dataset.morphIndex);
  if (morphIndex < 0 || morphIndex >= morphemes.length - 1) return;
  const left = morphemes[morphIndex];
  const right = morphemes[morphIndex + 1];
  const merged = {
    form: `${left.form || ""}${right.form || ""}`,
    gloss: mergeGloss(left.gloss, right.gloss),
    pos: left.pos || right.pos || "",
  };
  const next = [
    ...morphemes.slice(0, morphIndex),
    merged,
    ...morphemes.slice(morphIndex + 2),
  ];
  await saveMorphemeEdit(token, button.dataset.lineId, next);
}

async function editMorphemeGloss(cell) {
  const token = findToken(cell.dataset.lineId, cell.dataset.tokenIndex);
  if (!token) return;
  const morphemes = tokenMorphemes(token);
  const morphIndex = Number(cell.dataset.morphIndex);
  const morpheme = morphemes[morphIndex];
  if (!morpheme) return;
  const nextGloss = window.prompt("Gloss", morpheme.gloss || "");
  if (nextGloss === null) return;
  morphemes[morphIndex] = { ...morpheme, gloss: nextGloss.trim() || "?" };
  await saveMorphemeEdit(token, cell.dataset.lineId, morphemes);
}

function bindLineEditors() {
  for (const editor of els.textLines.querySelectorAll(".line-text-editor")) {
    const line = findLine(editor.dataset.lineId);
    editor.dataset.originalText = line?.text || "";
    editor.addEventListener("input", () => {
      editor.classList.toggle("dirty", editor.value !== editor.dataset.originalText);
      if (editor.value.includes("\n")) {
        scheduleLineTextSave(editor, 350);
      }
    });
    editor.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        scheduleLineTextSave(editor, 350);
      }
    });
    editor.addEventListener("blur", () => {
      scheduleLineTextSave(editor, 0);
    });
  }
}

function scheduleLineTextSave(editor, delay) {
  const lineId = editor.dataset.lineId;
  if (!lineId) return;
  clearTimeout(state.lineSaveTimers[lineId]);
  state.lineSaveTimers[lineId] = setTimeout(() => {
    saveLineText(editor).catch((error) => {
      els.subtitle.textContent = error.message;
      editor.classList.add("error");
    });
  }, delay);
}

async function saveLineText(editor) {
  const lineId = editor.dataset.lineId;
  const line = findLine(lineId);
  const text = editor.value.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  if (!line || !text || text === line.text) return;
  editor.classList.add("saving");
  els.subtitle.textContent = `${state.lesson?.title || state.lessonId} · saving line ${lineId}`;
  await api("/api/line-text", {
    method: "POST",
    body: JSON.stringify({
      lesson_id: state.lessonId,
      line_id: lineId,
      text,
    }),
  });
  await regenerateCurrentText();
}

function bindNoteableCells() {
  for (const cell of els.textLines.querySelectorAll(".noteable-cell")) {
    cell.addEventListener("click", (event) => {
      if (event.target.closest(".hyphen-delete")) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      showCellReviewMenu(cell, event);
    });
  }
}

function closeCellReviewMenu() {
  if (state.activeCellMenu) {
    state.activeCellMenu.remove();
    state.activeCellMenu = null;
  }
}

function cellCanAccept(token) {
  return (
    token
    && displayLemma(token) !== "?"
    && displayGloss(token) !== "?"
    && displayPos(token) !== "?"
    && token.match_type !== "unknown"
  );
}

function cellCanSplit(cell) {
  if (cell.dataset.rowLabel !== "MORPH") return false;
  if (cell.classList.contains("gloss-cell")) return false;
  const token = findToken(cell.dataset.lineId, cell.dataset.tokenIndex);
  if (!token) return false;
  const morphemes = tokenMorphemes(token);
  const morpheme = morphemes[Number(cell.dataset.morphIndex)];
  return Array.from(morpheme?.form || "").length > 1;
}

function showCellReviewMenu(cell, event) {
  const token = findToken(cell.dataset.lineId, cell.dataset.tokenIndex);
  if (!token) return;
  closeCellReviewMenu();

  const currentNote = cellNoteFor(
    cell.dataset.lineId,
    cell.dataset.tokenIndex,
    cell.dataset.rowLabel,
    cell.dataset.morphIndex ?? "",
  );
  const canAccept = cell.classList.contains("reviewable-cell") && cellCanAccept(token);
  const canSplit = cellCanSplit(cell);
  const canEditGloss = cell.dataset.rowLabel === "GLOSS";
  const menu = document.createElement("div");
  menu.className = "cell-review-menu";
  menu.innerHTML = `
    <div class="cell-review-title">
      <strong>${escapeHtml(wordDisplayForm(token))}</strong>
      <span>${escapeHtml(cell.dataset.rowLabel || "CELL")} · ${escapeHtml(cell.dataset.currentValue || "?")}</span>
    </div>
    ${currentNote ? `<div class="cell-review-note">${escapeHtml(currentNote.note || "")}</div>` : ""}
    <div class="cell-review-actions">
      ${cell.classList.contains("reviewable-cell") ? `<button type="button" class="primary" data-action="accept"${canAccept ? "" : " disabled"}>${canAccept ? "Accept guess" : "Cannot accept ?"}</button>` : ""}
      <button type="button" data-action="note">${currentNote ? "Edit note" : "Add note"}</button>
      ${canSplit ? `<button type="button" data-action="split">Add hyphen</button>` : ""}
      ${canEditGloss ? `<button type="button" data-action="gloss">Edit gloss</button>` : ""}
    </div>
  `;
  document.body.appendChild(menu);
  state.activeCellMenu = menu;

  const rect = cell.getBoundingClientRect();
  const menuRect = menu.getBoundingClientRect();
  const left = clamp(event.clientX || rect.left, 8, window.innerWidth - menuRect.width - 8);
  const top = clamp(rect.bottom + 6, 8, window.innerHeight - menuRect.height - 8);
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;

  const acceptButton = menu.querySelector('[data-action="accept"]');
  const noteButton = menu.querySelector('[data-action="note"]');
  const splitButton = menu.querySelector('[data-action="split"]');
  const glossButton = menu.querySelector('[data-action="gloss"]');
  acceptButton?.addEventListener("click", async (buttonEvent) => {
    buttonEvent.stopPropagation();
    try {
      await acceptCellGuess(cell);
    } catch (error) {
      els.subtitle.textContent = error.message;
    }
  });
  noteButton?.addEventListener("click", async (buttonEvent) => {
    buttonEvent.stopPropagation();
    try {
      await saveCellNoteFromPrompt(cell);
    } catch (error) {
      els.subtitle.textContent = error.message;
    }
  });
  splitButton?.addEventListener("click", async (buttonEvent) => {
    buttonEvent.stopPropagation();
    try {
      closeCellReviewMenu();
      await splitMorphemeAtCell(cell, event);
    } catch (error) {
      els.subtitle.textContent = error.message;
    }
  });
  glossButton?.addEventListener("click", async (buttonEvent) => {
    buttonEvent.stopPropagation();
    try {
      closeCellReviewMenu();
      await editMorphemeGloss(cell);
    } catch (error) {
      els.subtitle.textContent = error.message;
    }
  });
}

async function acceptCellGuess(cell) {
  const lineId = cell.dataset.lineId;
  const token = findToken(lineId, cell.dataset.tokenIndex);
  if (!cellCanAccept(token)) return;
  const morphemes = tokenMorphemes(token);
  const gloss = morphemes.length > 1 ? glossForMorphemes(morphemes) : displayGloss(token);
  const acceptanceNote = `Accepted highlighted ${cell.dataset.rowLabel || "cell"} guess in line ${lineId}.`;
  const existingNote = token.review_correction?.note || "";
  const result = await api("/api/corrections", {
    method: "POST",
    body: JSON.stringify({
      lesson_id: state.lessonId,
      form: token.form,
      corrected_form: token.corrected_form || "",
      lemma: displayLemma(token),
      gloss_en: gloss,
      gloss_tr: token.corrected_gloss_tr || token.gloss_tr || "",
      pos: displayPos(token),
      confidence: String(Math.max(Number(token.corrected_confidence || token.confidence || 0), 0.97)),
      note: existingNote ? `${existingNote} ${acceptanceNote}` : acceptanceNote,
      status: "approved",
      line_ids: [lineId],
      morphemes,
    }),
  });
  applySavedCorrection(result.correction);
  closeCellReviewMenu();
  els.subtitle.textContent = `${wordDisplayForm(token)} accepted; regenerating interlinear output`;
  renderQueue();
  await regenerateCurrentText();
}

async function saveCellNoteFromPrompt(cell) {
  const token = findToken(cell.dataset.lineId, cell.dataset.tokenIndex);
  if (!token) return;
  const existing = cellNoteFor(
    cell.dataset.lineId,
    cell.dataset.tokenIndex,
    cell.dataset.rowLabel,
    cell.dataset.morphIndex ?? "",
  );
  const promptText = [
    `${wordDisplayForm(token)} / ${cell.dataset.rowLabel || "CELL"} / ${cell.dataset.currentValue || "?"}`,
    "Note for this exact interlinear column:",
  ].join("\n");
  const note = window.prompt(promptText, existing?.note || "");
  if (note === null) return;
  const cleaned = note.trim();
  if (!cleaned) return;
  const result = await api("/api/cell-note", {
    method: "POST",
    body: JSON.stringify({
      lesson_id: state.lessonId,
      line_id: cell.dataset.lineId,
      token_index: cell.dataset.tokenIndex,
      row_label: cell.dataset.rowLabel || "",
      morph_index: cell.dataset.morphIndex ?? "",
      form: cell.dataset.noteForm || token.form || "",
      current_value: cell.dataset.currentValue || "",
      note: cleaned,
      status: "open",
    }),
  });
  upsertLocalCellNote(result.note);
  cell.classList.add("noted-cell");
  closeCellReviewMenu();
  els.subtitle.textContent = `Saved note for ${wordDisplayForm(token)} line ${cell.dataset.lineId}`;
}

function bindMorphemeEditors() {
  for (const button of els.textLines.querySelectorAll(".hyphen-delete")) {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        await deleteMorphemeBoundary(button);
      } catch (error) {
        els.subtitle.textContent = error.message;
      }
    });
  }
}

function queueAutosave() {
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => {
    const form = selectedForm();
    const status = statusFor(form || {}) === "approved" ? "approved" : "draft";
    saveCorrection(status);
  }, 650);
}

function moveNext() {
  const index = state.filtered.findIndex((item) => item.key === state.selectedKey);
  const next = state.filtered[index + 1] || state.filtered[0];
  if (next) {
    state.selectedKey = next.key;
    render();
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, "&#96;");
}

function renderTextCell(value, label) {
  if (label === "GLOSS" || label === "POS") {
    return renderWithAbbreviationTooltips(value);
  }
  return escapeHtml(value);
}

function abbreviationBoundary(value) {
  return !value || /[^A-Za-z0-9]/.test(value);
}

function abbreviationSpan(label, description) {
  const tooltip = `${label}: ${description}`;
  return `<span class="ling-abbr" tabindex="0" title="${escapeAttr(tooltip)}" aria-label="${escapeAttr(tooltip)}">${escapeHtml(label)}</span>`;
}

function renderWithAbbreviationTooltips(value) {
  const raw = String(value ?? "");
  let html = "";
  let index = 0;
  while (index < raw.length) {
    const key = ABBREVIATION_KEYS.find((candidate) => (
      raw.startsWith(candidate, index)
      && abbreviationBoundary(raw[index - 1])
      && abbreviationBoundary(raw[index + candidate.length])
    ));
    if (key) {
      html += abbreviationSpan(key, ABBREVIATION_GLOSSARY[key]);
      index += key.length;
    } else {
      html += escapeHtml(raw[index]);
      index += 1;
    }
  }
  return html || "?";
}

els.lessonSelect.addEventListener("change", async () => {
  await loadState(els.lessonSelect.value);
});

els.regenerateBtn.addEventListener("click", async () => {
  els.regenerateBtn.disabled = true;
  els.regenerateBtn.textContent = "Running";
  try {
    await api("/api/regenerate", {
      method: "POST",
      body: JSON.stringify({ lesson_id: state.lessonId }),
    });
    await loadState(state.lessonId);
  } finally {
    els.regenerateBtn.disabled = false;
    els.regenerateBtn.textContent = "Regenerate";
  }
});

els.searchInput.addEventListener("input", () => {
  state.selectedKey = "";
  state.selectedKey = pickInitialSelection();
  render();
});

for (const button of document.querySelectorAll(".segments button")) {
  button.addEventListener("click", () => {
    for (const other of document.querySelectorAll(".segments button")) other.classList.remove("active");
    button.classList.add("active");
    state.filter = button.dataset.filter;
    state.selectedKey = pickInitialSelection();
    render();
  });
}

els.reviewViewBtn?.addEventListener("click", () => {
  state.view = "review";
  render();
});

els.textViewBtn?.addEventListener("click", () => {
  state.view = "text";
  render();
});

for (const input of [els.correctedFormInput, els.lemmaInput, els.glossEnInput, els.glossTrInput, els.posInput, els.confidenceInput, els.noteInput]) {
  input.addEventListener("input", queueAutosave);
}

els.saveDraftBtn.addEventListener("click", () => saveCorrection("draft"));
els.approveBtn.addEventListener("click", async () => {
  await saveCorrection("approved");
  moveNext();
});

document.addEventListener("click", (event) => {
  if (!state.activeCellMenu) return;
  if (state.activeCellMenu.contains(event.target)) return;
  if (event.target.closest?.(".noteable-cell")) return;
  closeCellReviewMenu();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeCellReviewMenu();
});

loadTexts().catch((error) => {
  els.subtitle.textContent = error.message;
});
