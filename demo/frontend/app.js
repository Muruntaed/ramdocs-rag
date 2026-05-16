"use strict";

const API = "./api";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

const state = {
  versions: [],
  datasets: [],
  questions: [],
  selectedVersion: null,
  selectedDataset: null,
};

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const t = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status} ${t}`);
  }
  return r.json();
}

async function init() {
  try {
    [state.versions, state.datasets] = await Promise.all([
      fetchJSON(`${API}/versions`),
      fetchJSON(`${API}/datasets`),
    ]);
  } catch (e) {
    $("status").textContent = "Failed to load demo data: " + e.message;
    return;
  }
  state.selectedDataset = (state.datasets.find((d) => d.default) || state.datasets[0]).id;
  renderDatasets();
  renderVersions();
  await loadQuestions();

  $("run").addEventListener("click", run);
  $("question").addEventListener("change", onQuestionChange);
  $("dataset").addEventListener("change", onDatasetChange);
  $("dataset-load").addEventListener("click", onDatasetLoad);
}

function renderDatasets() {
  const sel = $("dataset");
  sel.innerHTML = "";
  for (const d of state.datasets) {
    const opt = el("option", null, `${d.label}  (${d.n_questions} Q)`);
    opt.value = d.id;
    if (d.id === state.selectedDataset) opt.selected = true;
    sel.appendChild(opt);
  }
}

function renderVersions() {
  const wrap = $("versions");
  wrap.innerHTML = "";
  for (const v of state.versions) {
    const card = el("div", "version-card");
    card.dataset.id = v.id;
    card.append(el("div", "label", v.label), el("div", "desc", v.description));
    card.addEventListener("click", () => {
      state.selectedVersion = v.id;
      for (const c of wrap.children) c.classList.toggle("selected", c.dataset.id === v.id);
    });
    wrap.appendChild(card);
  }
  if (state.versions.length) {
    state.selectedVersion = state.versions[state.versions.length - 1].id;
    wrap.children[state.versions.length - 1].classList.add("selected");
  }
}

async function loadQuestions() {
  try {
    state.questions = await fetchJSON(`${API}/questions?dataset=${encodeURIComponent(state.selectedDataset)}`);
  } catch (e) {
    $("status").textContent = "Failed to load questions: " + e.message;
    state.questions = [];
  }
  renderQuestions();
  onQuestionChange();
}

function renderQuestions() {
  const sel = $("question");
  sel.innerHTML = "";
  for (const q of state.questions) {
    const opt = el(
      "option",
      null,
      `${q.id}  ·  [${q.category}]  ${q.question.slice(0, 60)}${q.question.length > 60 ? "…" : ""}`
    );
    opt.value = q.id;
    sel.appendChild(opt);
  }
}

function onQuestionChange() {
  const id = $("question").value;
  const q = state.questions.find((x) => x.id === id);
  $("question-text").value = q ? q.question : "";
}

async function onDatasetChange() {
  state.selectedDataset = $("dataset").value;
  await loadQuestions();
}

async function onDatasetLoad() {
  const url = $("dataset-url").value.trim();
  const status = $("dataset-status");
  if (!url) {
    status.textContent = "Paste a URL first.";
    return;
  }
  $("dataset-load").disabled = true;
  status.textContent = "Loading…";
  try {
    const ds = await fetchJSON(`${API}/datasets/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    state.datasets = await fetchJSON(`${API}/datasets`);
    state.selectedDataset = ds.id;
    renderDatasets();
    await loadQuestions();
    status.textContent = `Loaded ${ds.n_questions} questions.`;
  } catch (e) {
    status.textContent = "Failed: " + e.message;
  } finally {
    $("dataset-load").disabled = false;
  }
}

async function run() {
  const version = state.selectedVersion;
  const question_id = $("question").value;
  if (!version || !question_id) return;

  const btn = $("run");
  const status = $("status");
  btn.disabled = true;
  status.textContent = "Running pipeline…";
  $("output").hidden = true;

  const overrideText = $("question-text").value.trim();
  const preset = state.questions.find((q) => q.id === question_id);
  const payload = { version, question_id, dataset: state.selectedDataset };
  if (preset && overrideText && overrideText !== preset.question) {
    payload.question_text_override = overrideText;
  }

  const t0 = performance.now();
  try {
    const resp = await fetchJSON(`${API}/answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent = `Done in ${elapsed}s`;
    renderResult(resp);
    $("output").hidden = false;
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderResult(r) {
  const ab = $("answer-block");
  ab.innerHTML = "";

  const fa = r.final_answer;
  if (fa.abstained || !fa.variants.length) {
    ab.appendChild(el("div", "answer-primary abstained", "Pipeline abstained."));
  } else {
    ab.appendChild(el("div", "answer-primary", fa.variants[0].answer));
    if (fa.variants.length > 1) {
      const wrap = el("div", "variants");
      for (const v of fa.variants.slice(1)) {
        const card = el("div", "variant", v.answer);
        const meta = el(
          "div",
          "v-meta",
          `confidence ${v.confidence.toFixed(2)} · supporting: ${(v.supporting_doc_ids || []).join(", ") || "—"}`
        );
        card.appendChild(meta);
        wrap.appendChild(card);
      }
      ab.appendChild(wrap);
    }
  }

  const meta = el("div", "answer-meta");
  meta.append(
    metaItem("latency", `${r.latency_s}s`),
    metaItem("cost", `$${r.total_cost_usd.toFixed(4)}`),
    metaItem("LLM calls", r.llm_calls),
    metaItem("pool", `${r.retrieval.pool_size} docs`),
    metaItem("rejected", fa.rejected_doc_ids.length || "0")
  );
  ab.appendChild(meta);

  if (fa.explanation) {
    const ex = el("div", "variant", fa.explanation);
    ex.style.background = "var(--panel-2)";
    ab.appendChild(ex);
  }

  renderTrace(r);
}

function metaItem(label, value) {
  const s = el("span");
  s.append(document.createTextNode(`${label}: `), Object.assign(el("strong"), { textContent: value }));
  return s;
}

function renderTrace(r) {
  const tb = $("trace-block");
  tb.innerHTML = "";

  tb.appendChild(
    traceStep({
      node: "retrieval",
      title: `Retrieval — ${r.retrieval.pool_size} docs in pool`,
      meta: r.retrieval.doc_ids.join(", "),
    })
  );

  for (let i = 0; i < r.trace.length; i++) {
    const s = r.trace[i];
    const cost = s.cost_usd ? `$${s.cost_usd.toFixed(5)}` : "$0";
    const tokens = `${s.tokens_in}→${s.tokens_out}`;
    tb.appendChild(
      traceStep({
        node: s.node,
        title: `${i + 1}. ${s.schema_name} · ${s.model}`,
        meta: `${s.latency_s}s · ${tokens} tok · ${cost}${s.error ? " · ERROR" : ""}`,
        step: s,
      })
    );
  }
}

function traceStep({ node, title, meta, step }) {
  const det = el("details", "step");
  const sum = el("summary");
  sum.append(el("span", `node-badge node-${node}`, node), el("span", null, title), el("span", "meta", meta));
  det.appendChild(sum);

  if (step) {
    const body = el("div", "body");
    body.append(
      el("div", "kv", "input (prompt)"),
      Object.assign(el("pre"), { textContent: step.input_preview }),
      el("div", "kv", "output (raw)"),
      Object.assign(el("pre"), { textContent: step.output_preview })
    );
    if (step.output && Object.keys(step.output).length) {
      body.append(el("div", "kv", "parsed JSON"), Object.assign(el("pre"), { textContent: JSON.stringify(step.output, null, 2) }));
    }
    if (step.error) {
      body.append(el("div", "kv error-tag", "error"), Object.assign(el("pre"), { textContent: step.error }));
    }
    det.appendChild(body);
  }
  return det;
}

init();
