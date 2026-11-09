// LiverLenz AI front-end (vanilla JS, talks to the Flask JSON API).
const $ = (id) => document.getElementById(id);
const state = { file: null, example: null, result: null, history: [] };

// ------------------------------------------------------------ helpers
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// Minimal, safe markdown: escape first, then bold/italic/code/lists/paragraphs.
function md(text) {
  const lines = esc(text).split(/\n/);
  let html = "", inList = false;
  for (const raw of lines) {
    let l = raw.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>").replace(/`([^`]+)`/g, '<code class="code">$1</code>');
    const m = l.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/);
    if (m) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${m[1]}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    html += l.trim() ? `<p>${l}</p>` : "";
  }
  return html + (inList ? "</ul>" : "");
}
function showError(msg) { const b = $("errorBox"); b.textContent = msg; b.classList.toggle("hidden", !msg); }
function setBusy(busy) {
  $("analyzeBtn").disabled = busy || !(state.file || state.example);
  $("analyzeSpin").classList.toggle("hidden", !busy);
  $("analyzeLabel").textContent = busy ? "Analysing…" : "Analyse slice";
}

// ------------------------------------------------------------ theme
$("themeBtn").addEventListener("click", () => {
  const dark = document.documentElement.classList.toggle("dark");
  try { localStorage.setItem("ll-theme", dark ? "dark" : "light"); } catch (e) {}
});

// ------------------------------------------------------------ status badge
fetch("/api/status").then((r) => r.json()).then((s) => {
  const b = $("llmBadge");
  if (s.llm.online) {
    b.className = "inline-flex items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-800 dark:border-brand-700/50 dark:bg-brand-900/30 dark:text-brand-200";
    b.innerHTML = `<span class="h-2 w-2 rounded-full bg-brand-500"></span><span>LLM: ${esc(s.llm.model)}</span>`;
  } else {
    b.className = "inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2.5 py-1 text-xs font-medium text-slate-500 dark:border-slate-700 dark:text-slate-400";
    b.innerHTML = `<span class="h-2 w-2 rounded-full bg-slate-400"></span><span>LLM offline</span>`;
    b.title = "No LLM_API_KEY set: report and chat use templates built from the measurements.";
  }
}).catch(() => {});

// ------------------------------------------------------------ examples
fetch("/api/examples").then((r) => r.json()).then((list) => {
  $("examples").innerHTML = list.map((e) => `
    <button type="button" class="example" data-file="${esc(e.file)}" title="${esc(e.description || e.title)}">
      <img src="/examples/${encodeURIComponent(e.file)}" alt="${esc(e.title)}" loading="lazy" />
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

// ------------------------------------------------------------ upload / drag-drop
function selectFile(f) {
  if (!f) return;
  if (!f.type.startsWith("image/")) { showError("Please drop an image file (PNG, JPG…)."); return; }
  state.file = f; state.example = null;
  document.querySelectorAll(".example").forEach((b) => b.classList.remove("selected"));
  $("fileName").textContent = `${f.name} · ${(f.size / 1024).toFixed(0)} KB`;
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
  showError(""); setBusy(true);
  $("emptyState").classList.add("hidden");
  $("results").classList.remove("hidden");
  ["imgOriginal", "imgOverlay", "imgBbox"].forEach((id) => { $(id).removeAttribute("src"); $(id).classList.add("skeleton"); });
  const fd = new FormData();
  if (state.file) fd.append("file", state.file); else fd.append("example", state.example);
  fd.append("pixel_spacing", $("spacing").value);
  try {
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await r.json().catch(() => ({ error: `Server error (${r.status})` }));
    if (!r.ok || data.error) throw new Error(data.error || `Server error (${r.status})`);
    state.result = data;
    render(data);
  } catch (err) {
    showError(err.message || "Something went wrong.");
    if (!state.result) { $("results").classList.add("hidden"); $("emptyState").classList.remove("hidden"); }
  } finally {
    ["imgOriginal", "imgOverlay", "imgBbox"].forEach((id) => $(id).classList.remove("skeleton"));
    setBusy(false);
  }
}

function render(d) {
  const s = d.stats;
  $("imgOriginal").src = d.images.original;
  $("imgOverlay").src = $("maskToggle").checked ? d.images.mask : d.images.overlay;
  $("imgBbox").src = d.images.bbox;
  $("resultMeta").textContent = `${d.filename} · ${s.image_width}×${s.image_height}px · ${d.inference_ms} ms`;

  const items = s.liver_detected ? [
    ["Liver area", `${s.liver_area_pct.toFixed(1)}%`],
    ["Area", s.liver_area_mm2 ? `${(s.liver_area_mm2 / 100).toFixed(1)} cm²` : `${s.liver_area_px.toLocaleString()} px`],
    ["Bounding box", s.bbox_mm ? `${s.bbox_mm[0]}×${s.bbox_mm[1]} mm` : `${s.bbox[2]}×${s.bbox[3]} px`],
    ["Box origin (x, y)", `${s.bbox[0]}, ${s.bbox[1]}`],
    ["Centroid (x, y)", `${s.centroid[0]}, ${s.centroid[1]}`],
    ["Location", s.region.replace(" part of the image", "")],
    ["Regions", s.components],
    ["Mean confidence", s.mean_confidence.toFixed(2)],
  ] : [["Liver area", "0%"], ["Result", "No liver found"], ["Slice", `${s.image_width}×${s.image_height}px`], ["Inference", `${d.inference_ms} ms`]];
  $("statsGrid").innerHTML = items.map(([k, v]) => `<div class="stat"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  $("notes").innerHTML = (s.notes || []).map((n) => `<li>⚠ ${esc(n)}</li>`).join("");

  const rep = d.report;
  $("reportCard").classList.remove("hidden");
  $("repFindings").innerHTML = rep.findings.map((f) => `<li>${esc(f)}</li>`).join("");
  $("repImpression").textContent = rep.impression;
  $("repSource").textContent = rep.source === "llm" ? "LLM-written" : "template";
  $("repDisclaimer").textContent = rep.disclaimer;

  $("chatContext").textContent = `Context: ${d.filename}`;
  addBot(s.liver_detected
    ? `Analysed **${d.filename}**: liver covers **${s.liver_area_pct.toFixed(1)}%** of the slice, in the ${s.region}. Ask me anything about the result.`
    : `Analysed **${d.filename}**: no liver was segmented on this slice.`, false);
  renderSuggestions();
}
$("maskToggle").addEventListener("change", () => {
  if (state.result) $("imgOverlay").src = $("maskToggle").checked ? state.result.images.mask : state.result.images.overlay;
  $("overlayCaption").textContent = $("maskToggle").checked ? "Binary mask" : "Liver mask overlay";
});

// ------------------------------------------------------------ downloads
function baseName() { return (state.result?.filename || "slice").replace(/\.[^.]+$/, ""); }
$("mdBtn").addEventListener("click", () => {
  if (!state.result) return;
  const blob = new Blob([state.result.report.markdown], { type: "text/markdown" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `liverlenz_${baseName()}.md` });
  a.click(); URL.revokeObjectURL(a.href);
});
$("pdfBtn").addEventListener("click", () => {
  if (!state.result) return;
  if (!window.jspdf) { showError("PDF library failed to load (offline?). Use the Markdown download instead."); return; }
  const { jsPDF } = window.jspdf; const r = state.result, rep = r.report;
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const W = 210, M = 16; let y = 18;
  doc.setFillColor(15, 118, 110); doc.rect(0, 0, W, 11, "F");
  doc.setTextColor(255); doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.text("LiverLenz AI  ·  Segmentation report", M, 7.3);
  doc.setTextColor(30); doc.setFontSize(16); y = 22; doc.text("Liver CT slice: automated segmentation", M, y);
  doc.setFont("helvetica", "normal"); doc.setFontSize(9); doc.setTextColor(100);
  y += 6; doc.text(`Image: ${r.filename}  (${r.stats.image_width} x ${r.stats.image_height} px)     Generated: ${rep.created}`, M, y);
  y += 5; doc.text(`Model: SegNet-style CNN (LiTS)     Impression: ${rep.source === "llm" ? "LLM-written" : "template (LLM offline)"}`, M, y);
  const iw = (W - 2 * M - 8) / 3; y += 5;
  [["Original", r.images.original], ["Liver mask overlay", r.images.overlay], ["Bounding box", r.images.bbox]].forEach(([t, src], i) => {
    const x = M + i * (iw + 4);
    doc.addImage(src, "PNG", x, y, iw, iw); doc.setFontSize(8); doc.text(t, x, y + iw + 4);
  });
  y += iw + 12;
  const section = (title, body, list) => {
    doc.setFont("helvetica", "bold"); doc.setFontSize(10); doc.setTextColor(15, 118, 110); doc.text(title.toUpperCase(), M, y); y += 5;
    doc.setFont("helvetica", "normal"); doc.setFontSize(10); doc.setTextColor(30);
    (list ? body : [body]).forEach((t) => {
      const lines = doc.splitTextToSize((list ? "•  " : "") + t.replace(/²/g, "^2"), W - 2 * M - (list ? 3 : 0));
      doc.text(lines, M + (list ? 1 : 0), y); y += lines.length * 4.6 + 1;
    });
    y += 3;
  };
  section("Findings", rep.findings, true);
  section("Impression", rep.impression, false);
  doc.setFillColor(254, 243, 199); const dl = doc.splitTextToSize(rep.disclaimer, W - 2 * M - 6);
  doc.rect(M, y - 1, W - 2 * M, dl.length * 4.2 + 9, "F");
  doc.setFont("helvetica", "bold"); doc.setFontSize(9); doc.setTextColor(146, 64, 14); doc.text("DISCLAIMER", M + 3, y + 4);
  doc.setFont("helvetica", "normal"); doc.text(dl, M + 3, y + 9);
  doc.save(`liverlenz_${baseName()}.pdf`);
});

// ------------------------------------------------------------ chat
const SUGGEST_SCAN = ["How large is the liver here?", "Where is the bounding box?", "How confident is the model?", "Can it see tumours?"];
const SUGGEST_NONE = ["How does the model work?", "How accurate is it?"];
function renderSuggestions() {
  const list = state.result ? SUGGEST_SCAN : SUGGEST_NONE;
  $("suggestions").innerHTML = list.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
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
      body: JSON.stringify({ message: text, stats: state.result?.stats || null, history: state.history }) });
    const d = await r.json();
    typing.remove();
    if (!r.ok) throw new Error(d.error || "Chat failed");
    state.history.push({ role: "user", content: text });
    addBot(d.reply, true, d.source);
  } catch (e) { typing.remove(); addBot(`Sorry, I couldn't answer that (${e.message}).`, false); }
}
$("chatForm").addEventListener("submit", (e) => { e.preventDefault(); send($("chatInput").value); });
$("clearChat").addEventListener("click", () => { $("chatLog").innerHTML = ""; state.history = []; greet(); });
function greet() { addBot("Hi! I'm the LiverLenz assistant. Load a CT slice and I can explain the segmentation, measurements and report.", false); renderSuggestions(); }
greet();
