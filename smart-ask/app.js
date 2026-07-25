import { evidence, sessionSteps } from "./evidence.js";
import { calculateEstimate, presets } from "./calculator.js";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const money = (value) => "$" + value.toFixed(2);

const terminal = $("#terminal-lines");
const smartLines = [
  ["USER", "Fix the failing slugify tests.", ""],
  ["ROUTER", "sonnet · 0.96 confidence", "router"],
  ["SONNET", "Reads test_slugify.py", "sonnet"],
  ["ROUTER", "sonnet · 0.98 confidence", "router"],
  ["SONNET", "Applies one-line delimiter fix", "sonnet"],
  ["RESULT", "3 tests passed", "success"],
];

function renderSession(mode = "smart") {
  const lines = mode === "smart"
    ? smartLines
    : smartLines.map(([label, value, kind]) => label === "ROUTER"
      ? ["ROUTER", "opus · forced baseline", "opus"]
      : label === "SONNET"
        ? ["OPUS", value, "opus"]
        : [label, value, kind]);
  terminal.innerHTML = lines.map(([label, value, kind], index) =>
    '<div class="terminal-line ' + kind + '" style="animation-delay:' + (index * 55) + 'ms"><span class="label">' + label + '</span><span class="value">' + value + "</span></div>"
  ).join("");
  const isSmart = mode === "smart";
  $("#footer-model").textContent = isSmart ? "claude-sonnet-4-6" : "claude-opus-4-1";
  $("#turn-cost").textContent = isSmart ? "$0.0848 this turn" : "$0.1404 this turn";
  $("#total-cost").textContent = isSmart ? "$0.1273 total" : "$0.2108 total";
  $("#cost-bar").style.width = isSmart ? (100 - evidence.lookup.value) + "%" : "100%";
  $("#cost-delta").textContent = isSmart ? "−39.6%" : "BASELINE";
}

$$("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    $$("[data-mode]").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    renderSession(button.dataset.mode);
  });
});
renderSession();

const calcInputs = {
  baseline: $("#baseline"),
  sonnetShare: $("#sonnet-share"),
  handoffs: $("#handoffs"),
  overhead: $("#overhead"),
};

function renderCalculator() {
  const values = Object.fromEntries(Object.entries(calcInputs).map(([key, input]) => [key, input.value]));
  const result = calculateEstimate(values);
  $("#share-output").textContent = values.sonnetShare + "%";
  $("#routed-cost").textContent = money(result.routedCost);
  $("#saving-cost").textContent = money(result.saving);
  $("#saving-percent").textContent = result.savingPercent.toFixed(1) + "%";
  $("#formula").innerHTML = money(result.opusWork) + " Opus work<br>+ " + money(result.sonnetWork) + " Sonnet work<br>+ " + money(result.classifierCost) + " classifier<br>+ " + money(result.handoffCost) + " handoffs";
}

Object.values(calcInputs).forEach((input) => input.addEventListener("input", () => {
  $$("[data-preset]").forEach((button) => button.classList.remove("active"));
  renderCalculator();
}));
$$("[data-preset]").forEach((button) => button.addEventListener("click", () => {
  const preset = presets[button.dataset.preset];
  Object.entries(preset).forEach(([key, value]) => { calcInputs[key].value = value; });
  $$("[data-preset]").forEach((item) => item.classList.toggle("active", item === button));
  renderCalculator();
}));
renderCalculator();

const range = $("#compress-range");
function renderCompression() {
  const width = Number(range.value);
  $(".compact-context").style.width = width + "%";
  const tokenCount = Math.round(evidence.handoff.compactTokens + ((width - 15) / 85) * (evidence.handoff.fullTokens - evidence.handoff.compactTokens));
  $("#compress-output").textContent = tokenCount.toLocaleString() + " tokens";
  $(".compact-context b").textContent = tokenCount.toLocaleString() + " input tokens";
}
range.addEventListener("input", renderCompression);
renderCompression();

const installCommand = [
  "git clone git@github.com:arshiaafzal/smart-ask.git",
  "cd smart-ask",
  "cp scripts/claude-smart-ask.local.env.example scripts/claude-smart-ask.local.env",
  "# Add ANTHROPIC_API_KEY, then:",
  "./scripts/claude-smart-ask",
].join("\n");

$$(".copy-button").forEach((button) => button.addEventListener("click", async () => {
  const text = button.id === "copy-install" ? installCommand : button.dataset.copy;
  try {
    await navigator.clipboard.writeText(text);
    const original = button.textContent;
    button.textContent = "Copied ✓";
    setTimeout(() => { button.textContent = original; }, 1800);
  } catch {
    button.textContent = "Select & copy";
  }
}));

const menuButton = $(".menu-toggle");
menuButton.addEventListener("click", () => {
  const open = menuButton.getAttribute("aria-expanded") !== "true";
  menuButton.setAttribute("aria-expanded", String(open));
  $("#nav-links").classList.toggle("open", open);
});
$$(".nav-links a").forEach((link) => link.addEventListener("click", () => {
  menuButton.setAttribute("aria-expanded", "false");
  $("#nav-links").classList.remove("open");
}));

if ("IntersectionObserver" in window && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });
  $$(".reveal").forEach((element) => observer.observe(element));
} else {
  $$(".reveal").forEach((element) => element.classList.add("visible"));
}

document.documentElement.dataset.sessionCost = sessionSteps
  .reduce((total, step) => total + step.smart, 0)
  .toFixed(4);
