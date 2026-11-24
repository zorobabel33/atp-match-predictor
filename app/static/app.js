"use strict";

const $ = (selector) => document.querySelector(selector);
const form = $("#match-form");
const submit = $("#simulate");
const sides = ["a", "b"];
const selected = { a: null, b: null };
const activeOptions = { a: -1, b: -1 };
let players = [];
let contract = null;
let modelInfo = null;
let currentRequest = null;
const number = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 });
const percent = new Intl.NumberFormat("fr-FR", {
  style: "percent",
  maximumFractionDigits: 1,
});
const labels = {
  surface: { Hard: "Dur", Clay: "Terre battue", Grass: "Gazon" },
  tourney_level: {
    G: "Grand Chelem",
    M: "Masters 1000",
    A: "ATP 250 / 500",
    F: "ATP Finals",
  },
  hand: { R: "Droitier", L: "Gaucher" },
};

async function getJSON(url, options = {}) {
  const response = await fetch(url, {
    signal: AbortSignal.timeout(15000),
    ...options,
  });
  if (!response.ok) {
    const error = new Error(`HTTP ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function normalize(text) {
  return text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase();
}

function displayName(side) {
  return selected[side]?.name || `Joueur ${side.toUpperCase()}`;
}

function updateButton() {
  submit.disabled =
    !contract || !selected.a || !selected.b || !modelInfo || !!currentRequest;
}

function showError(message) {
  $("#form-error").textContent = message;
  $("#form-error").hidden = false;
}

function invalidate() {
  if (currentRequest) {
    currentRequest.abort();
    currentRequest = null;
  }
  submit.classList.remove("loading");
  submit.querySelector("span").textContent = "Simuler le match";
  $("#result").setAttribute("aria-busy", "false");
  $("#result").setAttribute("aria-labelledby", "result-title");
  $("#result").classList.remove("has-result");
  $("#result-content").hidden = true;
  $("#result-empty").hidden = false;
  $("#form-error").hidden = true;
  updateButton();
}

function showIdentity(side) {
  const person = selected[side];
  const words = displayName(side).split(" ");
  $("#first-" + side).textContent = words.slice(0, -1).join(" ");
  $("#last-" + side).textContent = words.at(-1);
  $("#country-" + side).textContent = person?.country || "—";
}

function setPlayer(side, person) {
  selected[side] = { ...person };
  for (const feature of ["rank", "rank_points", "age", "ht", "hand"]) {
    form.elements[`player_${side}_${feature}`].value =
      feature === "hand"
        ? labels.hand[person[feature]]
        : number.format(person[feature]);
  }
  showIdentity(side);
  closePicker(side);
  invalidate();
}

function closePicker(side, returnFocus = false) {
  $("#picker-" + side).hidden = true;
  $("#choice-" + side).setAttribute("aria-expanded", "false");
  $("#search-" + side).setAttribute("aria-expanded", "false");
  $("#search-" + side).removeAttribute("aria-activedescendant");
  if (returnFocus) $("#choice-" + side).focus();
}

function filteredPlayers(side) {
  const query = normalize($("#search-" + side).value.trim());
  return players.filter((person) =>
    normalize(`${person.name} ${person.country}`).includes(query),
  );
}

function renderOptions(side) {
  const list = $("#options-" + side);
  list.replaceChildren();
  activeOptions[side] = -1;
  $("#search-" + side).removeAttribute("aria-activedescendant");
  const filtered = filteredPlayers(side);
  for (const person of filtered) {
    const item = document.createElement("li");
    item.role = "presentation";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "player-option";
    button.id = `option-${side}-${person.id}`;
    button.role = "option";
    button.tabIndex = -1;
    button.setAttribute(
      "aria-selected",
      String(selected[side]?.id === person.id),
    );
    for (const [text, className] of [
      [`#${person.rank}`, "option-rank"],
      [person.name, "option-name"],
      [person.country, "option-country"],
    ]) {
      const span = document.createElement("span");
      span.textContent = text;
      span.className = className;
      button.append(span);
    }
    button.addEventListener("click", () => {
      setPlayer(side, person);
      $("#choice-" + side).focus();
    });
    item.append(button);
    list.append(item);
  }
  if (!filtered.length) {
    const empty = document.createElement("li");
    empty.className = "no-results";
    empty.role = "presentation";
    empty.textContent =
      "Aucun joueur trouvé. Essayez un autre nom.";
    list.append(empty);
  }
}

function openPicker(side) {
  if (!players.length) return;
  sides
    .filter((other) => other !== side)
    .forEach((other) => closePicker(other));
  const opening = $("#picker-" + side).hidden;
  if (!opening) return closePicker(side);
  $("#picker-" + side).hidden = false;
  $("#choice-" + side).setAttribute("aria-expanded", "true");
  const search = $("#search-" + side);
  search.value = "";
  search.setAttribute("aria-expanded", "true");
  renderOptions(side);
  search.focus();
}

for (const side of sides) {
  $("#choice-" + side).addEventListener("click", () => openPicker(side));
  $(`[data-open="${side}"]`).addEventListener("click", () => openPicker(side));
  $("#search-" + side).addEventListener("input", () => renderOptions(side));
  $("#picker-" + side).addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      closePicker(side, true);
      return;
    }
    const buttons = [...$("#options-" + side).querySelectorAll("button")];
    if (["ArrowDown", "ArrowUp"].includes(event.key) && buttons.length) {
      event.preventDefault();
      const change = event.key === "ArrowDown" ? 1 : -1;
      activeOptions[side] =
        (activeOptions[side] + change + buttons.length) % buttons.length;
      buttons.forEach((button, index) =>
        button.classList.toggle(
          "keyboard-active",
          index === activeOptions[side],
        ),
      );
      const active = buttons[activeOptions[side]];
      $("#search-" + side).setAttribute("aria-activedescendant", active.id);
      active.scrollIntoView({ block: "nearest" });
    }
    if (event.key === "Enter") {
      event.preventDefault();
      buttons[Math.max(activeOptions[side], 0)]?.click();
    }
  });
}

document.addEventListener("click", (event) => {
  for (const side of sides) {
    if (!$(`.side-${side} .player-picker`).contains(event.target))
      closePicker(side);
  }
});
document.addEventListener("focusin", (event) => {
  for (const side of sides) {
    if (!$(`.side-${side} .player-picker`).contains(event.target))
      closePicker(side);
  }
});

form.addEventListener("input", (event) => {
  if (!event.target.matches("[type=search]")) invalidate();
});

$("#swap-players").addEventListener("click", () => {
  if (!selected.a || !selected.b) return;
  const { a, b } = selected;
  setPlayer("a", b);
  setPlayer("b", a);
});

function makeRadio(group, name, value, text, checked, surfaceClass = null) {
  const label = document.createElement("label");
  const input = document.createElement("input");
  input.type = "radio";
  input.name = name;
  input.value = value;
  input.checked = checked;
  input.required = true;
  const span = document.createElement("span");
  if (surfaceClass) {
    const dot = document.createElement("i");
    dot.className = `surface-dot ${surfaceClass}`;
    span.append(dot);
  }
  span.append(document.createTextNode(text));
  label.append(input, span);
  group.append(label);
}

function configureForm(data) {
  contract = data;
  for (const value of data.allowed_values.surface) {
    makeRadio(
      $("#surface-options"),
      "surface",
      value,
      labels.surface[value] || value,
      value === data.allowed_values.surface[0],
      value.toLowerCase(),
    );
  }
  for (const value of data.allowed_values.best_of) {
    makeRadio(
      $("#format-options"),
      "best_of",
      value,
      `${value} sets`,
      value === data.allowed_values.best_of[0],
    );
  }
  for (const value of data.allowed_values.tourney_level) {
    $("#tourney-level").add(
      new Option(labels.tourney_level[value] || value, value),
    );
  }
  $("#tourney-level").value = data.allowed_values.tourney_level.includes("M")
    ? "M"
    : data.allowed_values.tourney_level[0];
}

function renderResult(result, names) {
  const score = result.probability_player_a_wins;
  if (
    !Number.isFinite(score) ||
    score < 0 ||
    score > 1 ||
    !["player_a", "player_b"].includes(result.predicted_winner)
  )
    throw new Error("Invalid model response");
  const side = result.predicted_winner === "player_a" ? "a" : "b";
  $("#winner-caption").textContent =
    score === 0.5 ? "Un duel à égalité" : "Favori du match";
  $("#winner-name").textContent =
    score === 0.5 ? "Tout reste à jouer." : names[side];
  $("#winner-probability").textContent = percent.format(
    side === "a" ? score : 1 - score,
  );
  $("#result-player-a").textContent = `${names.a} · ${percent.format(score)}`;
  $("#result-player-b").textContent =
    `${names.b} · ${percent.format(1 - score)}`;
  $("#probability-a").style.width = `${score * 100}%`;
  $("#result-version").textContent =
    `${modelInfo.data_source === "synthetic" ? "Démonstration · " : ""}Modèle v${result.model_version}`;
  $("#result-empty").hidden = true;
  $("#result-content").hidden = false;
  $("#result").classList.add("has-result");
  $("#result").setAttribute("aria-labelledby", "winner-name");
  $("#result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (
    !contract || !selected.a || !selected.b || currentRequest || !modelInfo
    || !form.reportValidity()
  )
    return;
  if (selected.a?.id && selected.a.id === selected.b?.id) {
    showError("Pour un vrai duel, choisissez deux joueurs différents.");
    return;
  }
  $("#form-error").hidden = true;
  const values = new FormData(form);
  const payload = {};
  for (const name of contract.features) {
    const playerFeature = name.match(/^player_([ab])_(.+)$/);
    if (playerFeature) {
      payload[name] = selected[playerFeature[1]][playerFeature[2]];
      continue;
    }
    payload[name] =
      contract.integer.includes(name) || Object.hasOwn(contract.ranges, name)
        ? Number(values.get(name))
        : values.get(name);
  }
  const names = { a: displayName("a"), b: displayName("b") };
  const controller = new AbortController();
  currentRequest = controller;
  const timeout = setTimeout(() => controller.abort("timeout"), 20000);
  submit.classList.add("loading");
  submit.querySelector("span").textContent = "Le modèle entre en jeu…";
  $("#result").setAttribute("aria-busy", "true");
  updateButton();
  try {
    const result = await getJSON("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (currentRequest === controller) renderResult(result, names);
  } catch (error) {
    if (currentRequest !== controller) return;
    if (error.status === 422)
      showError(
        "Les données de ce duel n’ont pas pu être validées. Choisissez d’autres joueurs ou conditions de match.",
      );
    else if (error.status === 503)
      showError(
        "Le modèle n’est pas encore disponible. Réessayez dans quelques instants.",
      );
    else
      showError(
        "Le match n’a pas pu être calculé. Vérifiez la connexion, puis réessayez.",
      );
  } finally {
    clearTimeout(timeout);
    if (currentRequest === controller) {
      currentRequest = null;
      submit.classList.remove("loading");
      submit.querySelector("span").textContent = "Simuler le match";
      $("#result").setAttribute("aria-busy", "false");
      updateButton();
    }
  }
});

async function connectModel() {
  try {
    await getJSON("/ready");
    modelInfo = await getJSON("/model-info");
    const demo = modelInfo.data_source === "synthetic";
    $("#model-status").textContent = demo
      ? "Modèle de démonstration"
      : `Modèle v${modelInfo.model_version} · Prêt à jouer`;
    $("#model-status").className = `model-status ${demo ? "demo" : "ready"}`;
    $("#method-model").textContent =
      `${modelInfo.model_name} · Version ${modelInfo.model_version}`;
    if (demo) {
      $("#result-empty p").textContent =
        "Aperçu avec un modèle entraîné sur des données synthétiques. Les pronostics sont illustratifs.";
      $("#method-model").textContent +=
        " · Démonstration sur données synthétiques";
    }
    updateButton();
  } catch {
    modelInfo = null;
    $("#model-status").textContent = "Modèle indisponible";
    $("#model-status").className = "model-status";
    $("#method-model").textContent =
      "Le modèle approuvé n’est pas encore chargé.";
    $("#result-empty p").textContent =
      "Composez votre affiche. La simulation sera disponible dès que le modèle sera connecté.";
    updateButton();
    setTimeout(connectModel, 5000);
  }
}

const dialog = $("#about-dialog");
for (const id of ["#about-open", "#source-open"])
  $(id).addEventListener("click", () => dialog.showModal());
$("#about-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) {
    const r = dialog.getBoundingClientRect();
    if (
      event.clientX < r.left ||
      event.clientX > r.right ||
      event.clientY < r.top ||
      event.clientY > r.bottom
    )
      dialog.close();
  }
});

async function initialize() {
  try {
    const [schema, snapshot] = await Promise.all([
      getJSON("/contract"),
      getJSON("/static/players.json"),
    ]);
    configureForm(schema);
    players = snapshot.players;
    const date = new Date(
      `${snapshot.ranking_date}T12:00:00Z`,
    ).toLocaleDateString("fr-FR", {
      day: "numeric",
      month: "short",
      year: "numeric",
      timeZone: "UTC",
    });
    $("#snapshot-label").textContent =
      `Snapshot du ${date} · ${number.format(players.length)} joueurs`;
    $("#method-source").textContent =
      `Classement du ${date}. Archive épinglée au commit ${snapshot.source_commit.slice(0, 7)}. ${players.length} profils complets disponibles.`;
    setPlayer(
      "a",
      players.find((p) => p.name.includes("Alcaraz")) || players[0],
    );
    setPlayer(
      "b",
      players.find((p) => p.name.includes("Djokovic")) ||
        players.find((p) => p.id !== selected.a.id),
    );
    await connectModel();
  } catch {
    $("#snapshot-label").textContent = "Catalogue indisponible";
    $("#model-status").textContent = "Chargement incomplet";
    showError(
      "Le catalogue des joueurs n’a pas pu être chargé. Rechargez la page pour réessayer.",
    );
  }
}

initialize();
