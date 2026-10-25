// Front-end for the Review Insight demo. Talks to the FastAPI backend in app.py.
const LABELS = ["negative", "neutral", "positive"];
const EXAMPLES = {
  "Mixed": "Love the colour and it fits perfectly, but delivery took three weeks and the box arrived crushed.",
  "Positive": "Absolutely love these earbuds! Sound quality is amazing for the price and the battery lasts all day.",
  "Neutral": "It's okay. Does the job, nothing special, and a bit overpriced for what you get.",
  "Negation": "Not worth the money. I would not recommend this to anyone.",
  "Safety issue": "The charger started smoking after two days and melted the cable. Dangerous! I want a refund.",
  "Counterfeit": "This is a cheap knock off, not the genuine brand advertised. Seller ignored my messages.",
};
const MODEL_NAMES = { bert: "DistilBERT", logreg: "TF-IDF + Logistic Regression" };

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const sentimentColor = (l) => `var(--${l.slice(0, 3)})`;
const ALERT_ICON = `<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 3l10 18H2z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v5M12 18v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;

let model = "bert";
let batchData = null;
let lastFile = null;

// ---------------------------------------------------------------- tooltip
const tip = $("#tooltip");
document.addEventListener("mousemove", (e) => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.hidden = true; return; }
  tip.innerHTML = t.dataset.tip;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(e.clientX + 14, innerWidth - r.width - 8)}px`;
  tip.style.top = `${Math.min(e.clientY + 16, innerHeight - r.height - 8)}px`;
});

// ---------------------------------------------------------------- tabs & model switch
$$(".tab").forEach((b) => b.addEventListener("click", () => {
  $$(".tab").forEach((t) => t.classList.toggle("active", t === b));
  $$(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${b.dataset.tab}`));
}));

$$(".seg").forEach((b) => b.addEventListener("click", () => {
  model = b.dataset.model;
  $$(".seg").forEach((s) => { s.classList.toggle("active", s === b); s.setAttribute("aria-checked", s === b); });
  if ($("#single-result .verdict")) analyzeSingle();
  if (lastFile) runBatch(lastFile);
}));

// ---------------------------------------------------------------- single review
const examplesBox = $(".examples");
Object.entries(EXAMPLES).forEach(([name, text]) => {
  const c = document.createElement("button");
  c.className = "chip"; c.textContent = name;
  c.addEventListener("click", () => {
    $("#review").value = text;
    $$(".chip", examplesBox).forEach((x) => x.classList.toggle("active", x === c));
    analyzeSingle();
  });
  examplesBox.appendChild(c);
});

$("#analyze-btn").addEventListener("click", analyzeSingle);
$("#review").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyzeSingle(); });

async function analyzeSingle() {
  const text = $("#review").value.trim();
  if (!text) return;
  const btn = $("#analyze-btn");
  btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Analyzing`;
  try {
    const res = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    renderSingle(await res.json());
  } catch (err) {
    $("#single-result").innerHTML = `<div class="empty error">Error: ${esc(err.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Analyze";
  }
}

function renderSingle(r) {
  const probs = LABELS.map((l) => `
    <div class="prob-row" data-tip="p(${l}) = ${pct(r.probabilities[l], 1)}">
      <span class="name">${l}</span>
      <div class="track"><div class="fill" style="width:${pct(r.probabilities[l], 1)};background:${sentimentColor(l)}"></div></div>
      <span class="val">${pct(r.probabilities[l])}</span>
    </div>`).join("");

  // Word importance: shade words that support the prediction in the predicted class colour,
  // words that push against it in grey. Opacity scales with importance.
  const maxImp = Math.max(0.05, ...r.explanation.map((w) => Math.abs(w.importance)));
  const predSoft = r.sentiment === "negative" ? "227,73,72" : r.sentiment === "positive" ? "42,120,214" : "140,139,133";
  const words = r.explanation.map((w) => {
    const a = Math.min(1, Math.abs(w.importance) / maxImp) * 0.55;
    const style = w.importance > 0
      ? `background:rgba(${predSoft},${a})`
      : `text-decoration:underline dotted rgba(140,139,133,${Math.min(1, a * 2.5)});text-decoration-thickness:2px;text-underline-offset:5px`;
    const dir = w.importance > 0 ? "supports" : "pushes against";
    return `<span class="w" style="${style}" data-tip="<b>${esc(w.word)}</b><br>${dir} “${r.sentiment}”<br>removing it changes p by ${(-w.importance * 100).toFixed(1)} pts">${esc(w.word)}</span>`;
  }).join(" ");

  const aspects = r.aspects.length
    ? r.aspects.map((a) => `
        <div class="aspect">
          <strong>${esc(a.aspect)}</strong>
          <span><span class="tag ${a.sentiment}">${a.sentiment}</span></span>
          <span class="quote">“${esc(a.sentence)}”</span>
        </div>`).join("")
    : `<div class="muted">No specific product aspect detected.</div>`;

  const alert = r.risk_flags.length ? `
    <div class="alert">${ALERT_ICON}
      <div><strong>Escalate</strong> — urgency score ${r.urgency.toFixed(2)}<br>
      ${r.risk_flags.map((f) => `<span class="flag">${esc(f)}</span>`).join("")}</div>
    </div>` : "";

  $("#single-result").innerHTML = `
    <div class="verdict">
      <span class="label"><span class="swatch" style="background:${sentimentColor(r.sentiment)};width:14px;height:14px"></span>${r.sentiment}</span>
      <span class="conf">${pct(r.confidence)} confidence · ${MODEL_NAMES[r.model]}</span>
    </div>
    ${alert}
    <div class="section-h">Class probabilities</div>
    ${probs}
    <div class="section-h">Why? Word importance</div>
    <div class="explain">${words}</div>
    <div class="explain-legend">
      <span><i style="background:rgba(${predSoft},.5)"></i>supports “${r.sentiment}”</span>
      <span><i style="background:none;border-bottom:2px dotted rgba(140,139,133,.9);border-radius:0;height:6px"></i>pushes against it</span>
      <span>Hover a word for details (occlusion analysis)</span>
    </div>
    <div class="section-h">Aspect-level sentiment</div>
    ${aspects}`;
}

// ---------------------------------------------------------------- batch
const dz = $("#dropzone"), fileInput = $("#file-input");
dz.addEventListener("click", () => fileInput.click());
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault(); dz.classList.remove("drag");
  if (e.dataTransfer.files[0]) runBatch(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => fileInput.files[0] && runBatch(fileInput.files[0]));
$("#sample-btn").addEventListener("click", async () => {
  const blob = await (await fetch("/static/sample_reviews.csv")).blob();
  runBatch(new File([blob], "sample_reviews.csv", { type: "text/csv" }));
});
$("#text-col").addEventListener("change", (e) => lastFile && runBatch(lastFile, e.target.value));

async function runBatch(file, textCol) {
  lastFile = file;
  $("#batch-result").innerHTML = `<div class="card empty"><div><span class="spinner"></span>Analyzing ${esc(file.name)} with ${MODEL_NAMES[model]}…</div></div>`;
  const fd = new FormData();
  fd.append("file", file); fd.append("model", model);
  if (textCol) fd.append("text_col", textCol);
  try {
    const res = await fetch("/api/batch", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    batchData = await res.json();
    batchData.fileName = file.name;
    const sel = $("#text-col");
    sel.innerHTML = batchData.columns.map((c) => `<option ${c === batchData.text_col ? "selected" : ""}>${esc(c)}</option>`).join("");
    $(".col-select").hidden = false;
    renderBatch();
  } catch (err) {
    $("#batch-result").innerHTML = `<div class="card empty error">Error: ${esc(err.message)}</div>`;
  }
}

function stackedBar(counts, total, withLabels = true) {
  return LABELS.map((l) => {
    const share = total ? counts[l] / total : 0;
    if (!share) return "";
    const ink = l === "neutral" ? "var(--text)" : "#fff";
    return `<div style="flex:${share};background:${sentimentColor(l)};color:${ink}"
      data-tip="<b>${l}</b>: ${counts[l]} (${pct(share, 1)})">${withLabels && share >= 0.12 ? pct(share) : ""}</div>`;
  }).join("");
}

function renderBatch() {
  const d = batchData, sc = d.sentiment_counts;
  const escalations = d.rows.filter((r) => r.risk_flags.length && r.sentiment === "negative");

  const kpi = (label, value, sub, color) => `
    <div class="kpi"><div class="k-label">${color ? `<span class="swatch" style="background:${color}"></span>` : ""}${label}</div>
    <div class="k-value">${value}</div><div class="k-sub">${sub}</div></div>`;

  // Pain-point map: aspects sorted by share of negative mentions
  const aspects = Object.entries(d.aspects)
    .map(([a, c]) => ({ a, c, n: LABELS.reduce((s, l) => s + c[l], 0) }))
    .filter((x) => x.n >= 2)
    .sort((x, y) => y.c.negative / y.n - x.c.negative / x.n);
  const painRows = aspects.map((x) => `
    <div class="pp-row">
      <span>${esc(x.a)}</span>
      <div class="stacked">${stackedBar(x.c, x.n)}</div>
      <span class="n" data-tip="${x.n} sentences mention this aspect">${x.n}</span>
    </div>`).join("") || `<div class="muted">Not enough aspect mentions.</div>`;

  const queue = [...d.rows].sort((a, b) => b.urgency - a.urgency).filter((r) => r.risk_flags.length).slice(0, 6);
  const queueHtml = queue.map((r) => `
    <div class="queue-item">
      <div class="meta"><span class="score">${r.urgency.toFixed(2)}</span>
        ${r.risk_flags.map((f) => `<span class="flag">${esc(f)}</span>`).join("")}
        <span class="tag ${r.sentiment}">${r.sentiment}</span></div>
      <div>${esc(r.text.length > 170 ? r.text.slice(0, 170) + "…" : r.text)}</div>
    </div>`).join("") || `<div class="muted">No safety, fraud or defect issues detected.</div>`;

  $("#batch-result").innerHTML = `
    <div class="kpis">
      ${kpi("Reviews analysed", d.n.toLocaleString(), `${esc(d.fileName)} · column “${esc(d.text_col)}”`)}
      ${kpi("Negative", pct(sc.negative / d.n), `${sc.negative} reviews`, "var(--neg)")}
      ${kpi("Neutral", pct(sc.neutral / d.n), `${sc.neutral} reviews`, "var(--neu)")}
      ${kpi("Positive", pct(sc.positive / d.n), `${sc.positive} reviews`, "var(--pos)")}
      ${kpi("To escalate", escalations.length, "negative + safety/fraud/defect flag")}
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Pain-point map · sentiment by aspect</div>
        <div class="muted" style="margin:-4px 0 12px">Sentence-level sentiment for every sentence mentioning an aspect, most negative first. Right column = mentions.</div>
        ${painRows}
        <div class="legend">${LABELS.map((l) => `<span><span class="swatch" style="background:${sentimentColor(l)}"></span>${l}</span>`).join("")}</div>
      </div>
      <div class="card">
        <div class="card-title">Urgent-issue queue</div>
        <div class="muted" style="margin:-4px 0 6px">Ranked by p(negative) × risk-keyword weight.</div>
        ${queueHtml}
      </div>
    </div>
    <div class="card" style="margin-top:20px">
      <div class="table-head">
        <div class="card-title" style="margin:0">All reviews</div>
        <div class="filters">
          ${["all", ...LABELS, "flagged"].map((f) => `<button class="chip ${f === "all" ? "active" : ""}" data-filter="${f}">${f}</button>`).join("")}
          <input class="search" id="search" placeholder="Search text…">
          <button class="secondary" id="download">Download scored CSV</button>
        </div>
      </div>
      <div class="table-wrap"><table>
        <thead><tr><th>Sentiment</th><th>Conf.</th><th>Review</th><th>Aspects</th><th>Flags</th></tr></thead>
        <tbody id="rows"></tbody>
      </table></div>
      <div class="muted" id="row-note" style="margin-top:8px"></div>
    </div>`;

  let filter = "all";
  const draw = () => {
    const q = $("#search").value.toLowerCase();
    const rows = d.rows.filter((r) =>
      (filter === "all" || r.sentiment === filter || (filter === "flagged" && r.risk_flags.length)) &&
      (!q || r.text.toLowerCase().includes(q)));
    $("#rows").innerHTML = rows.slice(0, 150).map((r) => `
      <tr>
        <td><span class="tag ${r.sentiment}">${r.sentiment}</span></td>
        <td class="num">${pct(r.confidence)}</td>
        <td class="txt">${esc(r.text)}</td>
        <td><div class="mini-tags">${Object.entries(r.aspects).map(([a, s]) => `<span class="tag ${s}">${esc(a)}</span>`).join("")}</div></td>
        <td>${r.risk_flags.map((f) => `<span class="flag">${esc(f)}</span>`).join("")}</td>
      </tr>`).join("");
    $("#row-note").textContent = `Showing ${Math.min(rows.length, 150)} of ${rows.length} matching reviews.`;
  };
  $$("[data-filter]").forEach((b) => b.addEventListener("click", () => {
    filter = b.dataset.filter;
    $$("[data-filter]").forEach((x) => x.classList.toggle("active", x === b));
    draw();
  }));
  $("#search").addEventListener("input", draw);
  $("#download").addEventListener("click", downloadCsv);
  draw();
}

function downloadCsv() {
  const rows = batchData.rows;
  const cols = Object.keys(rows[0]);
  const cell = (v) => {
    if (Array.isArray(v)) v = v.join("; ");
    else if (v && typeof v === "object") v = Object.entries(v).map(([k, s]) => `${k}: ${s}`).join("; ");
    v = String(v ?? "");
    return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v;
  };
  const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => cell(r[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = batchData.fileName.replace(/\.csv$/i, "") + "_scored.csv";
  a.click();
}

analyzeSingle();
