/**
 * Sign Language Translator – Frontend JavaScript
 *
 * Polls /api/state every 100 ms and updates the UI:
 *   • Model & hand-detected badges
 *   • Confidence / stability meters
 *   • Text output (with per-character pop animation)
 *   • Control buttons (space, backspace, clear, speak, copy)
 */

"use strict";

// ── DOM refs ─────────────────────────────────────────────────────────────
const modelDot    = document.getElementById("modelDot");
const modelLabel  = document.getElementById("modelLabel");
const handDot     = document.getElementById("handDot");
const handLabel   = document.getElementById("handLabel");

const cameraFrame = document.getElementById("cameraFrame");
const signBubble  = document.getElementById("signBubbleText");

const confFill    = document.getElementById("confFill");
const confValue   = document.getElementById("confValue");
const stabFill    = document.getElementById("stabFill");
const stabHint    = document.getElementById("stabHint");

const textOutput  = document.getElementById("textOutput");
const placeholder = document.getElementById("placeholder");

const charCount   = document.getElementById("charCount");
const wordCount   = document.getElementById("wordCount");

const spaceBtn    = document.getElementById("spaceBtn");
const backBtn     = document.getElementById("backBtn");
const clearBtn    = document.getElementById("clearBtn");
const speakBtn    = document.getElementById("speakBtn");
const copyBtn     = document.getElementById("copyBtn");
const toast       = document.getElementById("toast");

const setupNotice = document.getElementById("setupNotice");

// ── State ────────────────────────────────────────────────────────────────
let prevText      = "";
let prevSign      = "NOTHING";
let toastTimer    = null;
let speechSynth   = window.speechSynthesis || null;

// ── Polling ───────────────────────────────────────────────────────────────
async function fetchState() {
  try {
    const resp = await fetch("/api/state");
    if (!resp.ok) return;
    const data = await resp.json();
    updateUI(data);
  } catch {
    // Server temporarily unreachable – silently retry
  }
}

function updateUI(data) {
  // ── Model badge ──────────────────────────────────────────────────────
  if (data.model_trained) {
    modelDot.className   = "badge-dot ready";
    modelLabel.textContent = "Model ready";
    if (setupNotice) setupNotice.style.display = "none";
  } else {
    modelDot.className   = "badge-dot warn";
    modelLabel.textContent = "Model not trained";
    if (setupNotice) setupNotice.style.display = "";
  }

  // ── Hand-detected badge ──────────────────────────────────────────────
  if (data.hand_detected) {
    handDot.className   = "badge-dot active";
    handLabel.textContent = "Hand detected";
    cameraFrame.classList.add("hand-active");
  } else {
    handDot.className   = "badge-dot dim";
    handLabel.textContent = "No hand";
    cameraFrame.classList.remove("hand-active");
  }

  // ── Sign bubble ───────────────────────────────────────────────────────
  const sign = data.current_sign;
  if (sign !== "NOTHING") {
    signBubble.textContent = sign === "SPACE" ? "⎵" : sign === "DELETE" ? "⌫" : sign;
    if (sign !== prevSign) {
      signBubble.parentElement.style.animation = "none";
      void signBubble.parentElement.offsetWidth;           // reflow
      signBubble.parentElement.style.animation = "";
    }
  } else {
    signBubble.textContent = "–";
  }
  prevSign = sign;

  // ── Confidence meter ──────────────────────────────────────────────────
  const confPct = Math.round(data.confidence * 100);
  confFill.style.width   = confPct + "%";
  confValue.textContent  = confPct + "%";

  // Color shift: red → orange → green
  if (confPct >= 75) {
    confFill.style.background = "linear-gradient(90deg, #10b981, #34d399)";
  } else if (confPct >= 40) {
    confFill.style.background = "linear-gradient(90deg, #f59e0b, #fbbf24)";
  } else {
    confFill.style.background = "linear-gradient(90deg, #ef4444, #f87171)";
  }

  // ── Stability meter ───────────────────────────────────────────────────
  const stabPct = Math.round(data.stability * 100);
  stabFill.style.width = stabPct + "%";
  stabHint.textContent  = stabPct >= 100 ? "registered!" : stabPct > 0 ? "hold…" : "–";

  // ── Text output ───────────────────────────────────────────────────────
  const text = data.typed_text;
  if (text !== prevText) {
    renderText(text, prevText);
    prevText = text;
  }
}

/**
 * Render the typed text, animating only newly added characters.
 */
function renderText(text, prev) {
  if (!text) {
    textOutput.innerHTML = "";
    placeholder.style.display = "";
    textOutput.prepend(placeholder);
    charCount.textContent = "0 characters";
    wordCount.textContent = "0 words";
    return;
  }

  // Hide placeholder
  placeholder.style.display = "none";

  if (text.length < prev.length) {
    // Deletion – just set plain text
    textOutput.textContent = text;
  } else {
    // Addition – animate new characters
    const added = text.slice(prev.length);
    if (prev.length === 0) {
      textOutput.textContent = "";
    }
    added.split("").forEach((ch) => {
      const span = document.createElement("span");
      span.textContent = ch;
      span.className   = "new-char";
      textOutput.appendChild(span);
    });
  }

  // Counts
  charCount.textContent = text.length + " character" + (text.length !== 1 ? "s" : "");
  const words = text.trim().split(/\s+/).filter((w) => w.length > 0);
  wordCount.textContent = words.length + " word" + (words.length !== 1 ? "s" : "");

  // Scroll to bottom
  textOutput.scrollTop = textOutput.scrollHeight;
}

// ── Button handlers ───────────────────────────────────────────────────────
spaceBtn.addEventListener("click", () =>
  fetch("/api/space", { method: "POST" }).then(fetchState)
);

backBtn.addEventListener("click", () =>
  fetch("/api/backspace", { method: "POST" }).then(fetchState)
);

clearBtn.addEventListener("click", () =>
  fetch("/api/clear", { method: "POST" }).then(fetchState)
);

speakBtn.addEventListener("click", () => {
  const text = prevText.trim();
  if (!text) return;
  if (!speechSynth) {
    alert("Text-to-speech is not supported in your browser.");
    return;
  }
  speechSynth.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate  = 0.95;
  utt.pitch = 1.0;
  speechSynth.speak(utt);
});

copyBtn.addEventListener("click", () => {
  const text = prevText;
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => showToast("Copied!")).catch(() => {
    // Fallback for browsers that block clipboard
    const el = document.createElement("textarea");
    el.value = text;
    el.style.cssText = "position:fixed;opacity:0;";
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
    showToast("Copied!");
  });
});

// ── Toast ──────────────────────────────────────────────────────────────────
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2200);
}

// ── Keyboard shortcuts ─────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === " ")           { e.preventDefault(); spaceBtn.click(); }
  if (e.key === "Backspace")   { e.preventDefault(); backBtn.click(); }
  if (e.key === "Escape")      { clearBtn.click(); }
  if (e.key === "Enter")       { speakBtn.click(); }
});

// ── Start polling ──────────────────────────────────────────────────────────
fetchState();                         // immediate first fetch
setInterval(fetchState, 100);         // then every 100 ms
