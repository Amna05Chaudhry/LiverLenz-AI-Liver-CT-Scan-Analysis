// LiverLenz AI front-end (vanilla JS, talks to the Flask JSON API).
const $ = (id) => document.getElementById(id);
const state = { file: null, example: null, result: null, slice: null, history: [] };

// ------------------------------------------------------------ helpers
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function md(text) {   // minimal, safe markdown: escape first, then bold/italic/code/lists/paragraphs
  let html = "", inList = false;
  for (const raw of esc(text).split(/\n/)) {
    const l = raw.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>").replace(/`([^`]+)`/g, '<code class="code">$1</code>');
    const m = l.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/);
    if (m) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${m[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    html += l.trim() ? `<p>${l}</p>` : "";
  }
  return html + (inList ? "</ul>" : "");
}
function showError(msg) { const b = $("errorBox"); b.textContent = msg; b.classList.toggle("hidden", !msg); }
function setBusy(busy, label) {
  $("analyzeBtn").disabled = busy || !(state.file || state.example);
  $("analyzeSpin").classList.toggle("hidden", !busy);
  $("analyzeLabel").textContent = busy ? (label || "Analysing…") : "Analyse scan";
}
function size(l) {
  if (l.max_diameter_mm) return `${(l.max_diameter_mm / 10).toFixed(1)} cm` + (l.volume_ml != null ? ` · ${l.volume_ml < 0.1 ? "<0.1" : l.volume_ml.toFixed(1)} mL` : "");
  return `${l.area_px.toLocaleString()} px`;
}

// ------------------------------------------------------------ theme + status
$("themeBtn").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("ll-theme", dark ? "dark" : "light"); } catch (e) {}
});
fetch("/api/status").then((r) => r.json()).then((s) => {
  const b = $("llmBadge");
  if (s.llm.online) {
    b.className = "inline-flex whitespace-nowrap items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-800 dark:border-brand-700/50 dark:bg-brand-900/30 dark:text-brand-200";
    b.innerHTML = `<span class="h-2 w-2 rounded-full bg-brand-500"></span><span>LLM: ${esc(s.llm.model)}</span>`;
  } else {
    b.className = "inline-flex whitespace-nowrap items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-500 dark:border-slate-700 dark:text-slate-400";
    b.innerHTML = `<span class="h-2 w-2 rounded-full bg-slate-400"></span><span>LLM offline</span>`;
    b.title = "No LLM_API_KEY set: report and chat use templates built from the measurements.";
  }
  if (!s.models_available) showError("Model files are missing from models/. See the README (Quick start) to get them.");
}).catch(() => {});

// ------------------------------------------------------------ examples
fetch("/api/examples").then((r) => r.json()).then((list) => {
  $("examples").innerHTML = list.map((e) => `
    <button type="button" class="example" data-file="${esc(e.file)}" title="${esc(e.description || e.title)}">
      <img src="/examples/${encodeURIComponent(e.thumb || e.file)}" alt="${esc(e.title)}" loading="lazy" />
      <span>${esc(e.title)}</span>
    </button>`).join("");
  document.querySelectorAll(".example").forEach((btn) => btn.addEventListener("click", () => {
    selectExample(btn.dataset.file, list.find((e) => e.file === btn.dataset.file));
    analyze();
  }));
});
function selectExample(file, meta) {
  state.example = file; state.file = null; $("fileInput").value = "";
  document.querySelectorAll(".example").forEach((b) => b.classList.toggle("selected", b.dataset.file === file));
  $("fileName").textContent = `Example: ${meta ? meta.title : file}`;
  $("spacing").value = (meta && meta.pixel_spacing) || "";
  setBusy(false);
}

// ------------------------------------------------------------ upload
function selectFile(f) {
  if (!f) return;
  state.file = f; state.example = null;
  document.querySelectorAll(".example").forEach((b) => b.classList.remove("selected"));
  $("fileName").textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
  showError(""); setBusy(false);
}
$("fileInput").addEventListener("change", (e) => selectFile(e.target.files[0]));
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("border-brand-600", "bg-brand-50"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("border-brand-600", "bg-brand-50"); }));
dz.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));
$("analyzeBtn").addEventListener("click", analyze);

// ------------------------------------------------------------ analyse
async function analyze() {
  if (!state.file && !state.example) return;
  const isVol = (state.file?.name || state.example || "").match(/\.nii(\.gz)?$/i);
  showError(""); setBusy(true, isVol ? "Segmenting volume…" : "Analysing…");
  $("emptyState").classList.add("hidden"); $("results").classList.remove("hidden");
  ["imgOriginal", "imgOverlay", "imgBoxes"].forEach((id) => { $(id).removeAttribute("src"); $(id).classList.add("skeleton"); });
  const fd = new FormData();
  if (state.file) fd.append("file", state.file); else fd.append("example", state.example);
  fd.append("pixel_spacing", $("spacing").value);
  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({ error: `Server error (${r.status})` }));
    if (!r.ok || data.error) throw new Error(data.error || `Server error (${r.status})`);
    state.result = data;
    renderResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong.");
    if (!state.result) { $("results").classList.add("hidden"); $("emptyState").classList.remove("hidden"); }
  } finally {
    ["imgOriginal", "imgOverlay", "imgBoxes"].forEach((id) => $(id).classList.remove("skeleton"));
    setBusy(false);
  }
}

function renderSlice(p) {
  state.slice = p;
  $("imgOriginal").src = p.images.original; $("imgOverlay").src = p.images.overlay; $("imgBoxes").src = p.images.boxes;
  const st = p.stats, S = state.result.summary;
  $("sliceRange").value = p.slice;
  $("sliceLabel").textContent = `slice ${p.slice + 1} / ${st.n_slices}`;
  const items = [
    ["Lesions (scan)", S.n_lesions],
    ["Tumour burden", `${S.tumour_burden_pct.toFixed(1)}%`],
    [S.tumour_volume_ml != null ? "Tumour volume" : "Tumour area (slice)", S.tumour_volume_ml != null ? `${S.tumour_volume_ml.toFixed(1)} mL` : (st.tumour_area_cm2 != null ? `${st.tumour_area_cm2} cm²` : `${st.tumour_area_px.toLocaleString()} px`)],
    [S.liver_volume_ml != null ? "Liver volume" : "Liver area (slice)", S.liver_volume_ml != null ? `${Math.round(S.liver_volume_ml)} mL` : (st.liver_area_cm2 != null ? `${st.liver_area_cm2} cm²` : `${st.liver_area_pct}% of slice`)],
  ];
  if (S.n_slices > 1) items.push(["This slice", st.tumour_detected ? `${st.lesions_on_slice.length} lesion(s)` : "no tumour"], ["Tumour on slice", st.tumour_area_cm2 != null ? `${st.tumour_area_cm2} cm²` : `${st.tumour_area_px} px`]);
  $("statsGrid").innerHTML = items.map(([k, v]) => `<div class="stat"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
}

function renderResult(d) {
  const S = d.summary;
  $("resultMeta").textContent = `${d.filename} · ${S.n_slices} slice(s) · ${S.original_size[0]}×${S.original_size[1]} px · ${(d.inference_ms / 1000).toFixed(1)} s`;
  // slider
  $("sliceBar").classList.toggle("hidden", S.n_slices < 2);
  $("sliceRange").max = S.n_slices - 1;
  $("lesionTicks").innerHTML = S.lesions.map((l) => {
    const left = 100 * l.slices[0] / Math.max(S.n_slices - 1, 1), w = Math.max(100 * (l.slices[1] - l.slices[0]) / Math.max(S.n_slices - 1, 1), 0.6);
    return `<span class="tick" style="left:${left}%;width:${w}%" data-slice="${l.slice}" title="Lesion ${l.id}"></span>`;
  }).join("");
  document.querySelectorAll(".tick").forEach((t) => t.addEventListener("click", () => goSlice(+t.dataset.slice)));
  renderSlice(d);
  // lesions table
  $("lesionWrap").classList.toggle("hidden", !S.lesions.length);
  $("lesionRows").innerHTML = S.lesions.slice(0, 30).map((l) => `<tr>
      <td class="py-2 pr-3 font-semibold">${l.id}</td><td class="py-2 pr-3">${esc(size(l))}</td>
      <td class="py-2 pr-3 tabular-nums">${S.n_slices > 1 ? `${l.slices[0] + 1}–${l.slices[1] + 1}` : "–"}</td>
      <td class="py-2 pr-3 tabular-nums">${l.bbox.join(", ")}</td><td class="py-2 pr-3 tabular-nums">${l.mean_prob.toFixed(2)}</td>
      <td class="py-2">${S.n_slices > 1 ? `<button type="button" class="row-btn" data-slice="${l.slice}">view</button>` : ""}</td></tr>`).join("");
  document.querySelectorAll(".row-btn").forEach((b) => b.addEventListener("click", () => goSlice(+b.dataset.slice)));
  // report
  const rep = d.report;
  $("reportCard").classList.remove("hidden");
  $("repFindings").innerHTML = rep.findings.map((f) => `<li>${esc(f)}</li>`).join("");
  $("repImpression").textContent = rep.impression;
  $("repSource").textContent = rep.source === "llm" ? "LLM-written" : "template";
  $("repDisclaimer").textContent = rep.disclaimer;
  // chat
  $("chatContext").textContent = `Context: ${d.filename}`;
  addBot(!S.liver_detected ? `Analysed **${d.filename}**: no liver was found.`
    : S.tumour_detected ? `Analysed **${d.filename}**: **${S.n_lesions} lesion(s)** segmented, tumour burden **${S.tumour_burden_pct.toFixed(1)}%** of the liver. Ask me about them.`
    : `Analysed **${d.filename}**: liver segmented, **no tumour** found by the model.`, false);
  renderSuggestions();
}

let sliceReq = 0;
async function goSlice(k) {
  const d = state.result; if (!d) return;
  k = Math.max(0, Math.min(d.summary.n_slices - 1, k));
  $("sliceRange").value = k; $("sliceLabel").textContent = `slice ${k + 1} / ${d.summary.n_slices}`;
  const my = ++sliceReq;
  try {
    const r = await fetch(`/api/slice/${d.id}/${k}`); const p = await r.json();
    if (!r.ok) throw new Error(p.error); if (my === sliceReq) renderSlice(p);
  } catch (e) { showError(e.message); }
}
let sliderTimer;
$("sliceRange").addEventListener("input", (e) => { clearTimeout(sliderTimer); sliderTimer = setTimeout(() => goSlice(+e.target.value), 60); });
$("prevSlice").addEventListener("click", () => goSlice(+$("sliceRange").value - 1));
$("nextSlice").addEventListener("click", () => goSlice(+$("sliceRange").value + 1));
document.addEventListener("keydown", (e) => {
  if (!state.result || document.activeElement?.tagName === "INPUT" && document.activeElement.type !== "range") return;
  if (e.key === "ArrowLeft") goSlice(+$("sliceRange").value - 1);
  if (e.key === "ArrowRight") goSlice(+$("sliceRange").value + 1);
});

// ------------------------------------------------------------ downloads
function baseName() { return (state.result?.filename || "scan").replace(/(\.nii)?\.[^.]+$/, ""); }
$("mdBtn").addEventListener("click", () => {
  if (!state.result) return;
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([state.result.report.markdown], { type: "text/markdown" })), download: `liverlenz_${baseName()}.md` });
  a.click(); URL.revokeObjectURL(a.href);
});
$("pdfBtn").addEventListener("click", () => {
  const r = state.result; if (!r) return;
  if (!window.jspdf) { showError("PDF library failed to load (offline?). Use the Markdown download instead."); return; }
  const { jsPDF } = window.jspdf, rep = r.report, S = r.summary, img = state.slice.images;
  const doc = new jsPDF({ unit: "mm", format: "a4" }); const W = 210, M = 16; let y;
  doc.setFillColor(15, 118, 110); doc.rect(0, 0, W, 11, "F");
  doc.setTextColor(255); doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.text("LiverLenz AI  ·  Liver lesion report", M, 7.3);
  doc.setTextColor(30); doc.setFontSize(16); y = 22; doc.text("CT: automated liver and tumour segmentation", M, y);
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(100);
  y += 6; doc.text(`Input: ${r.filename} (${S.kind}, ${S.n_slices} slice(s))     Generated: ${rep.created}`, M, y);
  y += 5; doc.text(`Model: two-stage U-Net (LiTS)     Impression: ${rep.source === "llm" ? "LLM-written" : "template (LLM offline)"}     Slice shown: ${state.slice.slice + 1}`, M, y);
  const iw = (W - 2 * M - 8) / 3; y += 5;
  [["Original", img.original], ["Liver (teal) / tumour (red)", img.overlay], ["Lesion boxes", img.boxes]].forEach(([t, src], i) => {
    const x = M + i * (iw + 4); doc.addImage(src, "PNG", x, y, iw, iw); doc.setFontSize(8); doc.text(t, x, y + iw + 4);
  });
  y += iw + 12;
  const section = (title, body, list) => {
    doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.setTextColor(15, 118, 110); doc.text(title.toUpperCase(), M, y); y += 5;
    doc.setFont("helvetica", "normal"); doc.setFontSize(9.5); doc.setTextColor(30);
    (list ? body : [body]).forEach((t) => {
      const lines = doc.splitTextToSize((list ? "•  " : "") + t.replace(/²/g, "^2").replace(/[–—]/g, "-").replace(/×/g, "x"), W - 2 * M - 3);
      if (y + lines.length * 4.4 > 280) { doc.addPage(); y = 18; }
      doc.text(lines, M + 1, y); y += lines.length * 4.4 + 1;
    });
    y += 3;
  };
  section("Findings", rep.findings, true);
  section("Impression", rep.impression, false);
  const dl = doc.splitTextToSize(rep.disclaimer, W - 2 * M - 6);
  if (y + dl.length * 4.2 + 10 > 285) { doc.addPage(); y = 18; }
  doc.setFillColor(254, 243, 199); doc.rect(M, y - 1, W - 2 * M, dl.length * 4.2 + 9, "F");
  doc.setFont("helvetica", "bold"); doc.setFontSize(9); doc.setTextColor(146, 64, 14); doc.text("DISCLAIMER", M + 3, y + 4);
  doc.setFont("helvetica", "normal"); doc.text(dl, M + 3, y + 9);
  doc.save(`liverlenz_${baseName()}.pdf`);
});

// ------------------------------------------------------------ chat
const SUGGEST_SCAN = ["How many lesions are there?", "How big is the largest lesion?", "Where are they located?", "Is it cancer?"];
const SUGGEST_NONE = ["How does the model work?", "How accurate is it?"];
function renderSuggestions() {
  $("suggestions").innerHTML = (state.result ? SUGGEST_SCAN : SUGGEST_NONE).map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
  $("suggestions").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => send(c.textContent)));
}
function bubble(role, html, source) {
  const wrap = document.createElement("div");
  wrap.className = role === "user" ? "flex justify-end" : "flex flex-col items-start";
  wrap.innerHTML = `<div class="bubble ${role === "user" ? "bubble-user" : "bubble-bot"}">${html}</div>` +
    (source === "template" ? `<span class="mt-0.5 pl-1 text-[10px] text-slate-400">template answer · LLM offline</span>` : "");
  $("chatLog").appendChild(wrap); $("chatLog").scrollTop = $("chatLog").scrollHeight;
  return wrap;
}
function addBot(text, record = true, source) { bubble("bot", md(text), source); if (record) state.history.push({ role: "assistant", content: text }); }
async function send(text) {
  text = text.trim(); if (!text) return;
  bubble("user", esc(text)); $("chatInput").value = "";
  const typing = bubble("bot", '<span class="typing"><span></span><span></span><span></span></span>');
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, summary: state.result?.summary || null, history: state.history }) });
    const d = await r.json(); typing.remove();
    if (!r.ok) throw new Error(d.error || "Chat failed");
    state.history.push({ role: "user", content: text });
    addBot(d.reply, true, d.source);
  } catch (e) { typing.remove(); addBot(`Sorry, I couldn't answer that (${e.message}).`, false); }
}
$("chatForm").addEventListener("submit", (e) => { e.preventDefault(); send($("chatInput").value); });
$("clearChat").addEventListener("click", () => { $("chatLog").innerHTML = ""; state.history = []; greet(); });
function greet() { addBot("Hi! I'm the LiverLenz assistant. Load a CT scan and I can explain the lesions the model found, their size, location and the report.", false); renderSuggestions(); }
greet();
