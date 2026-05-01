const API_BASE = "http://127.0.0.1:8000";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const checkBtn     = document.getElementById("check");
const clearBtn     = document.getElementById("clear");
const resultDiv    = document.getElementById("result");
const previewBox   = document.getElementById("preview-box");
const previewEl    = document.getElementById("selected-preview");
const noTextBox    = document.getElementById("no-text-box");
const manualText   = document.getElementById("manual-text");
const healthDot    = document.getElementById("health-dot");

// Ensemble
const ensLabel     = document.getElementById("ensemble-label");
const ensConf      = document.getElementById("ensemble-conf");
const agreeBadge   = document.getElementById("agree-badge");
const ensBanner    = document.getElementById("verdict-banner");
const ensBarFake   = document.getElementById("ens-bar-fake");
const ensBarReal   = document.getElementById("ens-bar-real");
const ensProbFake  = document.getElementById("ens-prob-fake");
const ensProbReal  = document.getElementById("ens-prob-real");

// DistilBERT
const dbLabel      = document.getElementById("db-label");
const dbBarFake    = document.getElementById("db-bar-fake");
const dbBarReal    = document.getElementById("db-bar-real");
const dbProbFake   = document.getElementById("db-prob-fake");
const dbProbReal   = document.getElementById("db-prob-real");

// RoBERTa
const rbLabel      = document.getElementById("rb-label");
const rbBarFake    = document.getElementById("rb-bar-fake");
const rbBarReal    = document.getElementById("rb-bar-real");
const rbProbFake   = document.getElementById("rb-prob-fake");
const rbProbReal   = document.getElementById("rb-prob-real");

// XAI
const flagsEl      = document.getElementById("flags");
const flagsContent = document.getElementById("flags-content");

// ── Helpers: show/hide using style.display (no Tailwind dependency) ───────────
const show = (el, type = "block") => { el.style.display = type; };
const hide = (el) => { el.style.display = "none"; };

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2500) });
    if (res.ok) {
      healthDot.className = "online";
      healthDot.title = "Backend online";
    } else throw new Error();
  } catch {
    healthDot.className = "offline";
    healthDot.title = "Backend offline — is uvicorn running?";
  }
}
checkHealth();

// ── Load selected text from background storage ────────────────────────────────
chrome.storage.local.get(["pendingText", "pendingTimestamp"], (data) => {
  const fresh = data.pendingTimestamp && (Date.now() - data.pendingTimestamp < 30000);
  if (data.pendingText && fresh) {
    applySelectedText(data.pendingText);
    chrome.action.setBadgeText({ text: "" });
  }
});

function applySelectedText(text) {
  manualText.value = text;
  previewEl.textContent = text;
  show(previewBox);
  hide(noTextBox);
}

// ── Textarea sync ─────────────────────────────────────────────────────────────
manualText.addEventListener("input", () => {
  const val = manualText.value.trim();
  if (val) {
    previewEl.textContent = val;
    show(previewBox);
    hide(noTextBox);
  } else {
    hide(previewBox);
    show(noTextBox);
  }
});

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
  manualText.value = "";
  hide(previewBox);
  show(noTextBox);
  hide(resultDiv);
  chrome.storage.local.remove(["pendingText", "pendingTimestamp"]);
});

// ── Analyze ───────────────────────────────────────────────────────────────────
checkBtn.addEventListener("click", async () => {
  const text = manualText.value.trim();
  if (!text) { show(noTextBox); return; }

  setLoading(true);
  hide(resultDiv);

  try {
    const res = await fetch(`${API_BASE}/predict/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(60000)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    ensLabel.textContent = "⚠ Error";
    ensLabel.className   = "verdict-text";
    ensConf.textContent  = err.message || "Request failed";
    ensConf.className    = "conf-pill";
    agreeBadge.textContent = "";
    hide(flagsEl);
    show(resultDiv);
  } finally {
    setLoading(false);
  }
});

// ── Render /predict/compare response ─────────────────────────────────────────
function renderResult(data) {
  const ens = data.ensemble;
  const db  = data.distilbert;
  const rb  = data.roberta;

  const isFake = ens.prediction === "Fake";

  // ── Ensemble verdict banner
  ensBanner.className = `verdict-banner fade-in ${isFake ? "fake" : "real"}`;
  ensLabel.textContent = isFake ? "🚨 Likely Fake" : "✅ Likely Real";
  ensLabel.className   = `verdict-text ${isFake ? "fake" : "real"}`;
  ensConf.textContent  = `${ens.confidence}%`;
  ensConf.className    = `conf-pill ${isFake ? "fake" : "real"}`;

  // Agreement badge
  agreeBadge.textContent = ens.models_agree ? "Both models agree" : "⚠ Models disagree";
  agreeBadge.className   = `badge ${ens.models_agree ? "agree" : "disagree"}`;

  // Ensemble bars
  setBar(ensBarFake, ensProbFake, ens.prob_fake);
  setBar(ensBarReal, ensProbReal, ens.prob_real);

  // ── DistilBERT
  dbLabel.textContent = db.prediction === "Fake" ? "🚨 Fake" : "✅ Real";
  dbLabel.className   = `model-verdict ${db.prediction === "Fake" ? "fake" : "real"}`;
  setBar(dbBarFake, dbProbFake, db.prob_fake);
  setBar(dbBarReal, dbProbReal, db.prob_real);

  // ── RoBERTa
  rbLabel.textContent = rb.prediction === "Fake" ? "🚨 Fake" : "✅ Real";
  rbLabel.className   = `model-verdict ${rb.prediction === "Fake" ? "fake" : "real"}`;
  setBar(rbBarFake, rbProbFake, rb.prob_fake);
  setBar(rbBarReal, rbProbReal, rb.prob_real);

  // ── XAI red flags
  const allFlags = [...new Set([...(db.red_flags || []), ...(rb.red_flags || [])])];
  if (allFlags.length > 0) {
    flagsContent.innerHTML = allFlags
      .map(f => `<span class="token-chip">${f}</span>`)
      .join("");
    show(flagsEl);
  } else {
    hide(flagsEl);
  }

  show(resultDiv);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setBar(barEl, labelEl, value) {
  barEl.style.width   = `${value}%`;
  labelEl.textContent = `${value}%`;
}

function setLoading(on) {
  checkBtn.disabled  = on;
  checkBtn.innerHTML = on
    ? `<span class="spinner"></span>Analyzing…`
    : "Analyze";
}