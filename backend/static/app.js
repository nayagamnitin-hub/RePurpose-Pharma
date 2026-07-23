// ---- RePurpose frontend logic ----
const CATEGORIES = [
  { emoji: "⚖️", label: "Lose body weight", q: "how to lose body weight" },
  { emoji: "📏", label: "Grow taller in puberty", q: "how to grow taller in puberty" },
  { emoji: "💈", label: "Regrow hair", q: "how to regrow hair / hair loss" },
  { emoji: "💪", label: "Build muscle", q: "how to build muscle" },
  { emoji: "🧠", label: "Alzheimer's disease", q: "Alzheimer disease" },
  { emoji: "🩸", label: "Type 2 diabetes", q: "type 2 diabetes mellitus" },
  { emoji: "🫀", label: "Parkinson's disease", q: "Parkinson disease" },
  { emoji: "🧬", label: "ALS (Lou Gehrig's)", q: "amyotrophic lateral sclerosis" },
  { emoji: "⏳", label: "Longevity", q: "longevity / anti-aging" },
];

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const pct = (x) => Math.round((x || 0) * 100);
const esc = (s) => (s == null ? "" : String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])));

let STUDY_TOPIC = "";   // clean topic for PubMed searches
let GOAL_CONTEXT = "";  // human goal label for AI context

// soft warm glow tones for the explore-card hover graphic
const GLOWS = ["#f1ddc6", "#e7dcc4", "#f0d6bf", "#e3ddd0", "#f1e0ca", "#ead7c0", "#eedfc9", "#e6d5be"];

// ---- landing setup ----
function buildLanding() {
  const grid = $("#popular-grid");
  CATEGORIES.forEach((c, i) => {
    const card = el("div", "explore-card");
    card.style.setProperty("--card-glow", GLOWS[i % GLOWS.length]);
    card.innerHTML = `<span class="explore-emoji">${c.emoji}</span>
      <h3>${esc(c.label)}</h3>
      <p>See established options and what's in clinical trials.</p>
      <span class="go">Explore →</span>`;
    card.onclick = () => runSearch(c.q);
    grid.appendChild(card);
  });
}

// ---- Custom AI chatbot (the n8n "RePurpose Pharma Chatbot", via the /api/chat proxy) ----
let CURRENT_QUERY = "";  // set during a search, so the chatbot is screen-aware
const CHAT_GREETING =
  "Hi! I'm your RePurpose pharmacology assistant. Ask me anything about drugs and research, "
  + "ask me to build you a plan (like lose weight and gain muscle, or regrow hair), or paste a study "
  + "and I'll break down the takeaways and how it could be repurposed.";

function aiSessionId() {
  // persist across page reloads so the n8n chatbot keeps its memory
  let s = localStorage.getItem("aiSession");
  if (!s) { s = "s-" + Math.random().toString(36).slice(2); localStorage.setItem("aiSession", s); }
  return s;
}

function chatAddMessage(box, text, who) {
  const row = el("div", `ai-row ${who}`);
  if (who === "bot") row.appendChild(el("div", "ai-avatar", "✚"));
  const msg = el("div", `ai-msg ${who}`);
  if (who === "bot") msg.innerHTML = mdLite(text); else msg.textContent = text;
  row.appendChild(msg);
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
  return msg;
}

function chatTyping(box) {
  const row = el("div", "ai-row bot");
  row.appendChild(el("div", "ai-avatar", "✚"));
  const msg = el("div", "ai-msg bot typing");
  msg.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  row.appendChild(msg);
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
  return msg;
}

async function chatSend(text, box) {
  chatAddMessage(box, text, "user");
  const typing = chatTyping(box);
  try {
    const res = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, sessionId: aiSessionId(), context: CURRENT_QUERY }),
    });
    const data = await res.json();
    typing.classList.remove("typing");
    typing.innerHTML = mdLite(data.reply || "(no response)");
    box.scrollTop = box.scrollHeight;
  } catch (e) {
    typing.classList.remove("typing");
    typing.textContent = "Couldn't reach the assistant: " + e.message;
  }
}

// full page (opened by the big hero CTA)
function openAIPage() {
  $("#landing").classList.add("hidden");
  $("#results").classList.add("hidden");
  $("#ai-page").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
  const box = $("#ai-messages");
  if (!box.dataset.greeted) { chatAddMessage(box, CHAT_GREETING, "bot"); box.dataset.greeted = "1"; }
}
function closeAIPage() {
  $("#ai-page").classList.add("hidden");
  $("#landing").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// docked popup (opened by the top-right button; usable while browsing results)
function openChatPopup() {
  $("#chat-modal").classList.remove("hidden");
  const box = $("#chat-modal-messages");
  if (!box.dataset.greeted) { chatAddMessage(box, CHAT_GREETING, "bot"); box.dataset.greeted = "1"; }
  $("#chat-modal-input").focus();
}
function closeChatPopup() { $("#chat-modal").classList.add("hidden"); }

// ---- search flow ----
async function runSearch(query) {
  $("#search-input").value = query;
  $("#landing").classList.add("hidden");
  $("#results").classList.remove("hidden");
  $("#report").classList.add("hidden");
  $("#loading").classList.remove("hidden");
  $("#loading-text").textContent = `Analyzing “${query}” across Open Targets, ClinicalTrials.gov, UniProt & PubMed…`;
  window.scrollTo({ top: 0, behavior: "smooth" });

  try {
    const res = await fetch(`/api/repurpose?q=${encodeURIComponent(query)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    const report = await res.json();
    renderReport(report, query);
  } catch (e) {
    $("#loading").classList.add("hidden");
    $("#report").classList.remove("hidden");
    $("#report").innerHTML = `<button class="btn btn-ghost back-btn" onclick="goHome()">← New search</button>
      <div class="summary-card"><h3>Couldn't complete that search</h3>
      <p class="muted-sm">${esc(e.message)}. This is often a brief upstream API hiccup. Try Analyze again.</p></div>`;
  }
}

function goHome() {
  $("#results").classList.add("hidden");
  $("#landing").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---- rendering ----
function renderReport(report, query) {
  $("#loading").classList.add("hidden");
  const rpt = $("#report");
  rpt.classList.remove("hidden");
  rpt.innerHTML = "";

  const interp = report.interpretation || {};
  STUDY_TOPIC = report.disease_name || interp.goal_label || "";
  GOAL_CONTEXT = report.disease_name || interp.goal_label || query;
  CURRENT_QUERY = GOAL_CONTEXT;  // makes the docked chatbot aware of what's on screen

  // back button
  const back = el("button", "btn btn-ghost back-btn", "← New search");
  back.onclick = goHome;
  rpt.appendChild(back);

  // head
  const head = el("header", "result-head");
  const title = report.disease_name || interp.goal_label || query;
  const eyebrow = interp.mode === "targets" ? "Goal · interpreted by AI" : "Condition";
  head.innerHTML = `<p class="eyebrow">${esc(eyebrow)}</p><h2>${esc(title)}</h2>`;
  if (interp.rationale) head.appendChild(el("p", "result-rationale", esc(interp.rationale)));
  head.appendChild(el("p", "targets-label", "🎯 Proteins targeted"));
  head.appendChild(el("p", "targets-hint",
    "The biological targets (proteins) involved in this condition. Established treatments act on these, " +
    "and repurposing candidates are matched against them."));
  const chips = el("div", "chips");
  (report.targets || []).slice(0, 10).forEach(t => chips.appendChild(el("span", "chip", esc(t.symbol))));
  head.appendChild(chips);
  rpt.appendChild(head);

  // curability note (e.g. "ALS has no cure; treatments slow progression")
  if (report.overview_note) {
    rpt.appendChild(el("div", "overview-note", `ℹ️ ${esc(report.overview_note)}`));
  }

  // AI summary (no heading — the summary has its own structure)
  if (report.summary) {
    const card = el("section", "summary-card");
    card.innerHTML = mdLite(report.summary);
    rpt.appendChild(card);
  }

  // Established
  rpt.appendChild(sectionTitle("Established treatments",
    "Approved or established options that act in the right direction for this goal."));
  const existing = report.existing_solutions || [];
  if (existing.length) {
    const featured = el("div", "featured-grid");
    existing.slice(0, 3).forEach(c => featured.appendChild(drugCard(c, true, true)));
    rpt.appendChild(featured);
    if (existing.length > 3) renderDrugGrid(rpt, existing.slice(3), 6, true);  // 3 featured + 6, rest behind "Show all"
  } else {
    rpt.appendChild(el("p", "empty", "No established treatments matched."));
  }

  // Repurposing
  rpt.appendChild(sectionTitle("Repurposing & in-development candidates",
    "Drugs from other uses, and real investigational agents in active clinical trials, promising but not yet established here."));
  const repur = report.repurposing_candidates || [];
  if (repur.length) {
    renderDrugGrid(rpt, repur, 9, false);  // show 9, rest behind "Show all"
  } else {
    rpt.appendChild(el("p", "empty", "No repurposing candidates found."));
  }

  // Targets
  rpt.appendChild(sectionTitle("Targets explored", ""));
  const tchips = el("div", "chips");
  (report.targets || []).forEach(t => {
    const c = el("span", "chip", esc(t.symbol));
    if (t.protein_function) c.title = t.protein_function;
    tchips.appendChild(c);
  });
  rpt.appendChild(tchips);

  window.scrollTo({ top: 0, behavior: "smooth" });
}

function sectionTitle(title, sub) {
  const s = el("section", "section");
  s.appendChild(el("h2", "section-title", esc(title)));
  if (sub) s.appendChild(el("p", "section-sub", esc(sub)));
  return s;
}

// render up to `cap` drug cards, hiding the rest behind a "Show all N" button
function renderDrugGrid(parent, cands, cap, isExisting) {
  const grid = el("div", "card-grid drug-grid");
  cands.slice(0, cap).forEach(c => grid.appendChild(drugCard(c, false, isExisting)));
  parent.appendChild(grid);
  if (cands.length > cap) {
    const btn = el("button", "btn btn-ghost show-more", `Show all ${cands.length}`);
    btn.onclick = () => { cands.slice(cap).forEach(c => grid.appendChild(drugCard(c, false, isExisting))); btn.remove(); };
    parent.appendChild(btn);
  }
}

// map raw stage codes to readable English
function prettyStage(s) {
  if (!s) return s;
  const map = {
    APPROVAL: "Approved", PHASE_4: "Phase 4", PHASE4: "Phase 4", PHASE_3: "Phase 3", PHASE3: "Phase 3",
    PHASE_2: "Phase 2", PHASE2: "Phase 2", PHASE_1: "Phase 1", PHASE1: "Phase 1", PHASE_1_2: "Phase 1/2",
    EARLY_PHASE_1: "Early Phase 1", EARLY_PHASE1: "Early Phase 1", PRECLINICAL: "Preclinical",
  };
  return map[String(s).toUpperCase()] || s;
}

// ---- one drug card ----
function drugCard(c, featured, isExisting) {
  const card = el("div", "drug-card" + (featured ? " featured" : "") + (c.prospective ? " prospective" : ""));
  const headline = (c.labels && c.labels[0]) || prettyStage(c.clinical_stage) || "";

  // header: name + clickable confidence badge
  const top = el("div", "drug-top");
  top.appendChild(el("h3", "drug-name", esc(c.name)));
  const confLabel = `${pct(c.confidence)}% confidence`;
  const rank = el("span", "rank-badge clickable", esc(confLabel));
  rank.title = "Click for an explanation";
  rank.onclick = (e) => { e.stopPropagation(); openExplain(c, confLabel, isExisting); };
  top.appendChild(rank);
  card.appendChild(top);

  // badges (each clickable -> AI explanation)
  const badges = el("div", "badges");
  (c.labels || []).forEach(l => {
    let cls = "badge clickable";
    if (/safe|low side|approved|established/i.test(l)) cls += " good";
    else if (/side effects|black-box|withdrawn|harsh/i.test(l)) cls += " warn";
    else if (/clinical trials/i.test(l)) cls += " trial";
    else if (/prospective|ai-proposed/i.test(l)) cls += " prospect";
    const b = el("span", cls, esc(l));
    b.title = "Click for an explanation";
    b.onclick = (e) => { e.stopPropagation(); openExplain(c, l, isExisting); };
    badges.appendChild(b);
  });
  if (badges.children.length) card.appendChild(badges);

  // metrics
  const m = el("div", "metrics");
  m.innerHTML = metric("Effect", pct(c.effectiveness_score), "") +
                metric("Safety", pct(c.safety_score), "safe") +
                metric("Confidence", pct(c.confidence), "conf");
  card.appendChild(m);

  // meta
  if (c.mechanism_of_action) card.appendChild(el("p", "drug-meta", `<strong>Mechanism:</strong> ${esc(c.mechanism_of_action)}`));
  if (c.via_targets && c.via_targets.length) card.appendChild(el("p", "drug-meta", `<strong>Acts via:</strong> ${esc(c.via_targets.join(", "))}`));
  if (c.clinical_stage) card.appendChild(el("p", "drug-meta", `<strong>Stage:</strong> ${esc(prettyStage(c.clinical_stage))}`));
  if (c.known_for && c.known_for.length) card.appendChild(el("p", "drug-meta", `<strong>Known for:</strong> ${esc(c.known_for.slice(0, 4).join(", "))}`));

  // source notes
  if (c.source === "clinical_trials") {
    card.appendChild(el("div", "info-note trial",
      "🧫 In clinical trials: a real investigational drug being studied for this area, not yet established."));
  } else if (c.prospective) {
    card.appendChild(el("div", "info-note prospect",
      "🔭 Prospective / AI-proposed: a mechanistic hypothesis. Scores are AI estimates."));
  }
  if (c.rationale) card.appendChild(el("p", "drug-meta", `<strong>Why it could work:</strong> ${esc(c.rationale)}`));
  if (c.trial_ids && c.trial_ids.length) {
    const links = c.trial_ids.map(n => `<a href="https://clinicaltrials.gov/study/${esc(n)}" target="_blank" rel="noopener">${esc(n)}</a>`).join(" · ");
    card.appendChild(el("p", "trial-links", `<strong>Trials:</strong> ${links}`));
  }
  if (c.top_adverse_events && c.top_adverse_events.length) {
    card.appendChild(el("p", "drug-meta", `<strong>Notable adverse events:</strong> ${esc(c.top_adverse_events.slice(0, 3).join(", "))}`));
  }

  // actions
  const actions = el("div", "drug-actions");
  const studiesBtn = el("button", "mini-btn", "📚 View studies");
  const askBtn = el("button", "mini-btn", "🤖 Ask AI");
  actions.append(studiesBtn, askBtn);
  card.appendChild(actions);

  const studiesBox = el("div", "studies");
  const askBox = el("div", "ask-box hidden");
  card.append(studiesBox, askBox);

  studiesBtn.onclick = () => loadStudies(c.name, studiesBox, studiesBtn);
  askBtn.onclick = () => {
    askBox.classList.toggle("hidden");
    if (!askBox.dataset.built) { buildAsk(c, askBox); askBox.dataset.built = "1"; }
  };

  // clicking the card (but not a button / link / input inside it) opens the detail modal
  card.onclick = (e) => { if (e.target.closest("a, button, input")) return; openModal(c); };
  return card;
}

// ---- drug detail modal (images + plain-English "what it is") ----
function openModal(c) {
  const body = $("#modal-body");
  body.innerHTML = `
    <h2 class="modal-name">${esc(c.name)}</h2>
    <div class="modal-badges">${(c.labels || []).map(l => `<span class="badge">${esc(l)}</span>`).join("")}</div>
    <div id="modal-images" class="modal-images"><p class="muted-sm">Looking for images…</p></div>
    <div id="modal-summary" class="modal-summary"><p class="muted-sm">Loading a quick description…</p></div>
    <div class="modal-facts">
      ${c.mechanism_of_action ? `<p class="drug-meta"><strong>Mechanism:</strong> ${esc(c.mechanism_of_action)}</p>` : ""}
      ${c.clinical_stage ? `<p class="drug-meta"><strong>Stage:</strong> ${esc(c.clinical_stage)}</p>` : ""}
      ${c.via_targets && c.via_targets.length ? `<p class="drug-meta"><strong>Acts via:</strong> ${esc(c.via_targets.join(", "))}</p>` : ""}
      ${c.known_for && c.known_for.length ? `<p class="drug-meta"><strong>Known for:</strong> ${esc(c.known_for.slice(0, 5).join(", "))}</p>` : ""}
      ${c.variants && c.variants.length ? `<p class="drug-meta"><strong>Includes these forms:</strong> ${esc(c.variants.join(", "))}</p>` : ""}
    </div>
    <div class="modal-actions"></div>
    <div id="modal-studies" class="studies"></div>
    <div id="modal-ask" class="ask-box"></div>`;

  const sBtn = el("button", "mini-btn", "📚 View studies");
  body.querySelector(".modal-actions").appendChild(sBtn);
  sBtn.onclick = () => loadStudies(c.name, $("#modal-studies"), sBtn);
  buildAsk(c, $("#modal-ask"));

  $("#modal-overlay").classList.remove("hidden");
  document.body.style.overflow = "hidden";

  fetch("/api/drug", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: c.name, mechanism: c.mechanism_of_action || "", targets: c.via_targets || [],
      known_for: c.known_for || [], clinical_stage: c.clinical_stage || "", source: c.source || "",
      prospective: !!c.prospective, goal: GOAL_CONTEXT, adverse_events: c.top_adverse_events || [],
      variants: c.variants || [],
    }),
  }).then(r => r.json()).then(info => {
    const imgs = $("#modal-images");
    const list = info.images || [];
    if (!list.length) imgs.innerHTML = `<p class="muted-sm">No images found for this drug.</p>`;
    // images are clickable -> open full size in a new tab; broken ones remove themselves
    else imgs.innerHTML = list.map(u =>
      `<a href="${esc(u)}" target="_blank" rel="noopener"><img src="${esc(u)}" alt="${esc(c.name)}" onerror="this.parentElement.remove()" loading="lazy"></a>`).join("");
    const sum = info.summary || info.description || "No plain-language description available.";
    const brandLine = info.brand ? `<p class="modal-brand">💊 Best known by the brand <strong>${esc(info.brand)}</strong></p>` : "";
    $("#modal-summary").innerHTML = brandLine + `<p>${esc(sum)}</p>` +
      (info.wiki_url ? `<p class="muted-sm"><a href="${esc(info.wiki_url)}" target="_blank" rel="noopener">More on Wikipedia →</a></p>` : "");
  }).catch(() => { $("#modal-summary").innerHTML = `<p class="muted-sm">Couldn't load the description.</p>`; });
}

function closeModal() {
  $("#modal-overlay").classList.add("hidden");
  document.body.style.overflow = "";
}

// ---- badge / score explanation popup ----
function openExplain(c, label, isExisting) {
  const body = $("#explain-body");
  body.innerHTML = `<p class="eyebrow">${esc(c.name)}</p><h3 class="explain-label">${esc(label)}</h3>
    <div id="explain-text"><p class="muted-sm">Loading explanation…</p></div>`;
  $("#explain-overlay").classList.remove("hidden");
  document.body.style.overflow = "hidden";
  fetch("/api/explain", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      drug: c.name, label, goal: GOAL_CONTEXT || CURRENT_QUERY || "",
      mechanism: c.mechanism_of_action || "", stage: prettyStage(c.clinical_stage) || "",
      known_for: c.known_for || [], warnings: c.warnings || [],
      effectiveness: pct(c.effectiveness_score), safety: pct(c.safety_score), confidence: pct(c.confidence),
      established: !!isExisting,
    }),
  }).then(r => r.json()).then(d => {
    $("#explain-text").innerHTML = mdLite(d.explanation || "No explanation available.");
    addLearnMore(c, isExisting, label);  // real studies + evidence, tailored to this badge
  }).catch(() => { $("#explain-text").innerHTML = `<p class="muted-sm">Couldn't load the explanation.</p>`; });
}

function addLearnMore(c, isExisting, label) {
  const wrap = $("#explain-body");
  const btn = el("button", "mini-btn learn-more", "📚 Learn more (see the evidence)");
  const out = el("div", "learn-out");
  wrap.append(btn, out);
  btn.onclick = () => {
    btn.disabled = true; btn.textContent = "Gathering evidence…";
    fetch("/api/evidence", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        drug: c.name, goal: GOAL_CONTEXT || CURRENT_QUERY || "", established: !!isExisting,
        mechanism: c.mechanism_of_action || "", targets: c.via_targets || [], label: label || "",
      }),
    }).then(r => r.json()).then(d => {
      btn.remove();
      let html = mdLite(d.summary || "No evidence summary available.");
      const studies = d.studies || [];
      if (studies.length) {
        html += `<p class="muted-sm" style="margin-top:10px"><strong>Sources:</strong></p><ul class="learn-studies">`;
        studies.forEach(s => {
          html += `<li><a href="https://pubmed.ncbi.nlm.nih.gov/${esc(s.pmid)}/" target="_blank" rel="noopener">PMID ${esc(s.pmid)}</a>`
            + (s.snippet ? `: ${esc(s.snippet.slice(0, 120))}…` : "") + `</li>`;
        });
        html += `</ul>`;
      }
      out.innerHTML = html;
    }).catch(() => { btn.disabled = false; btn.textContent = "📚 Learn more (see the evidence)"; out.innerHTML = `<p class="muted-sm">Couldn't load the evidence.</p>`; });
  };
}

function closeExplain() {
  $("#explain-overlay").classList.add("hidden");
  document.body.style.overflow = "";
}

function metric(label, value, cls) {
  return `<div class="metric">
    <div class="bar ${cls}"><span style="width:${value}%"></span></div>
    <div class="val">${value}%</div><div class="label">${label}</div></div>`;
}

async function loadStudies(drug, box, btn) {
  btn.disabled = true; btn.textContent = "Loading…";
  try {
    const res = await fetch(`/api/studies?drug=${encodeURIComponent(drug)}&topic=${encodeURIComponent(STUDY_TOPIC)}`);
    const data = await res.json();
    const studies = data.studies || [];
    if (!studies.length) { box.innerHTML = `<p class="muted-sm">No PubMed studies found for this drug.</p>`; }
    else {
      const ul = el("ul");
      studies.forEach(s => {
        const li = el("li");
        li.innerHTML = `<a href="https://pubmed.ncbi.nlm.nih.gov/${esc(s.pmid)}/" target="_blank" rel="noopener">PMID ${esc(s.pmid)}</a>`
          + (s.snippet ? `: ${esc(s.snippet.slice(0, 160))}…` : "");
        ul.appendChild(li);
      });
      box.innerHTML = ""; box.appendChild(ul);
    }
  } catch (e) {
    box.innerHTML = `<p class="muted-sm">Couldn't load studies (${esc(e.message)}).</p>`;
  }
  btn.disabled = false; btn.textContent = "📚 View studies";
}

function buildAsk(c, box) {
  const row = el("div", "ask-row");
  const input = el("input");
  input.type = "text";
  input.placeholder = `Ask about ${c.name}…`;
  const send = el("button", "mini-btn", "Ask");
  row.append(input, send);
  const answer = el("div", "ask-answer hidden");
  box.append(row, answer);

  const ask = async () => {
    const q = input.value.trim();
    if (!q) return;
    answer.classList.remove("hidden");
    answer.textContent = "Reading the literature…";
    try {
      const res = await fetch("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          drug: c.name, question: q, topic: STUDY_TOPIC,
          context: `${c.name}: ${c.mechanism_of_action || ""}; goal: ${GOAL_CONTEXT}`,
        }),
      });
      const data = await res.json();
      answer.innerHTML = mdLite(data.answer || "(no answer)");  // render bold/lists, not raw **
    } catch (e) {
      answer.textContent = "Couldn't reach the AI: " + e.message;
    }
  };
  send.onclick = ask;
  input.addEventListener("keydown", e => { if (e.key === "Enter") ask(); });
}

// very small markdown -> html (bold, headings, bullets) for the summary
function mdLite(text) {
  const lines = esc(text).split("\n");
  let html = "", inList = false;
  for (let raw of lines) {
    let line = raw.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${line.replace(/^\s*[-*]\s+/, "")}</li>`;
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      const t = line.trim();
      if (!t) continue;
      if (/^<strong>.*<\/strong>:?$/.test(t)) html += `<h4>${t}</h4>`;
      else html += `<p>${line}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

// ---- init ----
buildLanding();
$("#search-form").addEventListener("submit", e => {
  e.preventDefault();
  const q = $("#search-input").value.trim();
  if (q.length >= 2) runSearch(q);
});
$("#modal-close").addEventListener("click", closeModal);
$("#modal-overlay").addEventListener("click", e => { if (e.target.id === "modal-overlay") closeModal(); });
$("#explain-close").addEventListener("click", closeExplain);
$("#explain-overlay").addEventListener("click", e => { if (e.target.id === "explain-overlay") closeExplain(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") { closeModal(); closeExplain(); } });

// AI chatbot: top-right = docked popup (usable mid-search); big hero CTA = full page
$("#ai-nav-btn").addEventListener("click", openChatPopup);
$("#ai-cta").addEventListener("click", openAIPage);
$("#ai-back").addEventListener("click", closeAIPage);
$("#chat-modal-close").addEventListener("click", closeChatPopup);
$("#ai-chat-form").addEventListener("submit", e => {
  e.preventDefault();
  const input = $("#ai-chat-input");
  const t = input.value.trim();
  if (t) { chatSend(t, $("#ai-messages")); input.value = ""; }
});
$("#chat-modal-form").addEventListener("submit", e => {
  e.preventDefault();
  const input = $("#chat-modal-input");
  const t = input.value.trim();
  if (t) { chatSend(t, $("#chat-modal-messages")); input.value = ""; }
});
window.goHome = goHome;
