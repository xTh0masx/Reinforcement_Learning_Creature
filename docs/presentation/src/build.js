// Endemic Creatures - Projektpräsentation (Deutsch). Build: cd docs/presentation/src && npm install pptxgenjs && node build.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("./apply_theme.js");

const REPO = "C:/Users/thoma/PycharmProjects/Reinforcement_Learning_Creature";
const FIG = (p) => path.join(REPO, "docs/figures", p);
const OUT = path.join(REPO, "docs/presentation/Endemic_Creatures_RL.pptx");

const THEME = {
  name: "Endemic Creatures",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C2321", lt1: "FFFFFF", dk2: "1F3A35", lt2: "EDF2EF",
    accent1: "D98E04",   // Bernstein - Akzent
    accent2: "2E7D6B",   // Moosgrün - positiv / Myrranox
    accent3: "B23A48",   // Karminrot - negativ
    accent4: "5B6C8F",   // Schieferblau - DC
    accent5: "8FA89B", accent6: "C9B27C",
    hlink: "2E7D6B", folHlink: "5B6C8F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Endemic Creatures - Laufen lernen mit Reinforcement Learning";
pres.author = "Thomas";
pres.subject = "Projektpräsentation";
const C = pres.SchemeColor;
const W = 13.333, H = 7.5, M = 0.6;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITEL", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 4.3, w: 12.1, h: 1.3, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 5.7, w: 12.1, h: 1.1, fontSize: 18, color: C.accent6, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "ABSCHNITT", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.6, w: 12.1, h: 1.2, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.9, w: 12.1, h: 1.0, fontSize: 18, color: C.accent6 }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.35, fontSize: 10, color: "C9B27C" },
});
pres.defineSlideMaster({
  title: "INHALT", background: { color: C.background1 },
  margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: 12.1, h: 0.85, fontFace: "Cambria", fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "Endemic Creatures · Reinforcement Learning", options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: "8FA89B" } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8FA89B" },
});

// ------------------------------------------------------------------ helpers
let n = 0;
const content = (title, section) => {
  const s = pres.addSlide({ masterName: "INHALT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, ...o });
const card = (s, x, y, w, h, fill = C.background2, name = "card") =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
const dot = (s, x, y, label, fill = C.accent1, d = 0.5) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  s.addText(label, { isTextBox: true, x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 14, bold: true, color: C.background1, margin: 0 });
};
const bullets = (items, size = 14) => items.map((t, i) => {
  const run = typeof t === "string" ? { text: t, options: {} } : t;
  return { text: run.text, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size, ...run.options } };
});
const table = (s, rows, o, fs = 15, rowH = 0.48) => s.addTable(rows.map((r, i) => r.map((c) => (typeof c === "object" ? c : { text: String(c),
  options: i === 0 ? { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: fs, valign: "middle" }
                   : { fontSize: fs, color: HEX.dk1, fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 }, valign: "middle" } }))),
  { x: M, w: W - 2 * M, rowH, border: { type: "solid", pt: 0.5, color: "D5DDD8" }, fontFace: "Calibri", margin: [0.05, 0.1, 0.05, 0.1], ...o });
const notes = (s, t) => s.addNotes(t);

// ================================================================== 1 Titel
pres.addSection({ title: "Einleitung" });
let s = pres.addSlide({ masterName: "TITEL", sectionTitle: "Einleitung" });
s.addImage({ path: FIG("showcase_v6_still.jpg"), x: 0, y: 0, w: W, h: W * 544 / 1928, objectName: "hero" });
s.addText("Endemic Creatures: Laufen lernen mit Reinforcement Learning", { placeholder: "title" });
s.addText("Zwei Raubtiere lernen in einer Physik-Simulation zu gehen, zu jagen und sich wie ihre Vorbilder zu bewegen · Projektstand 28.10.2026", { placeholder: "body" });
notes(s, "Projektüberblick: Myrranox (Ameise, sechs Beine) und DC (Panther mit Zusatzarmen) lernen ihre Bewegungen selbst mit Reinforcement Learning in MuJoCo; Darstellung in Blender.");

// ================================================================== 2 Überblick in Zahlen
s = content("Das Projekt in Zahlen", "Einleitung");
const stats = [["2", "Kreaturen\nMyrranox & DC"], ["12", "Versionen des\nLaufgehirns"], ["≈ 50", "trainierte\nGehirne"], ["≈ 185 Mio.", "simulierte\nTrainingsschritte"], ["218", "Physik-Teile im\nDC-Körper (v12)"], ["≈ 38 h", "Rechenzeit\nLauf-Training"]];
stats.forEach(([big, label], i) => {
  const x = M + (i % 3) * 4.1, y = 1.5 + Math.floor(i / 3) * 2.55;
  card(s, x, y, 3.8, 2.25, C.background2, `stat${i}`);
  T(s, big, { x: x + 0.3, y: y + 0.25, w: 3.2, h: 1.0, fontFace: "Cambria", fontSize: 44, bold: true, color: i % 2 ? C.accent2 : C.accent1 });
  T(s, label, { x: x + 0.3, y: y + 1.25, w: 3.2, h: 0.8, fontSize: 15, color: C.text2 });
});
notes(s, "Simulierte Schritte: Summe aller Lauf- und Raubtier-Trainings inkl. Pilotläufe (Größenordnung). Rechenzeit Lauf-Training: Wandzeit, meist zwei Läufe parallel.");

// ================================================================== 3 Agenda
s = content("Inhalt", "Einleitung");
const agenda = [["1", "Aufbau", "Kreaturen, Architektur, Habitat"], ["2", "Dimensionen & Parameter", "Körper, Physik, Training"], ["3", "Regeln", "Belohnung: Plus- und Minuspunkte"],
  ["4", "Prozess", "Versionen v1 bis v12, Raubtier-Gehirne"], ["5", "DC v12", "Detail-Körper, Wolf, T-Rex-Arme, Hand"], ["6", "Analyse", "Lernkurven, Fußfolgen, Messwerte"], ["7", "Tools & Laufzeiten", "Software, Hardware, Rechenzeit"], ["8", "Erfahrungen", "Positives, Negatives, Lehren"]];
agenda.forEach(([nr, t, d], i) => {
  const col = i < 4 ? 0 : 1, row = i < 4 ? i : i - 4, x = M + col * 6.2, y = 1.45 + row * 1.25;
  dot(s, x, y + 0.1, nr, col ? C.accent2 : C.accent1, 0.6);
  T(s, t, { x: x + 0.85, y: y + 0.05, w: 5, h: 0.4, fontSize: 20, bold: true, color: C.text2 });
  T(s, d, { x: x + 0.85, y: y + 0.5, w: 5, h: 0.4, fontSize: 14, color: "4A5A55" });
});

// ================================================================== Abschnitt Aufbau
pres.addSection({ title: "Aufbau" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Aufbau" });
s.addText("1 · Aufbau", { placeholder: "title" });
s.addText("Kreaturen, Steuerungsarchitektur und Lebensraum", { placeholder: "body" });

// 5 Kreaturen
s = content("Zwei Raubtiere, zwei Vorbilder", "Aufbau");
[["Myrranox", "Ameise / Insekt", C.accent2, ["6 Beine × 3 Gelenke (Hüfte, Heben, Knie)", "8 kg, 1,5 m lang", "Gang: Dreifuß (3 Beine am Boden)", "Gehen 0,6 m/s · max. 1,2 m/s · dreht auf der Stelle", "schwerer Hinterleib (Gaster), Kopf mit Mandibeln"]],
 ["DC", "Panther mit Zusatzarmen", C.accent4, ["4 Beine × 4 Gelenke + 2 Arme × 4 Gelenke", "60 kg, 1,3 m hoch, ~2,7 m mit Schwanz", "Gang: lateraler Viertakt → Trab", "Gehen 0,8 m/s · Trab 2,5 m/s · Sprint 4 m/s", "Wirbelsäule, Hals, Kopf aktiv; Schwanz federnd"]]].forEach(([name, sub, col, items], i) => {
  const x = M + i * 6.2;
  card(s, x, 1.5, 5.9, 5.1, C.background2, `creature${i}`);
  s.addImage({ path: path.join(__dirname, i ? "dc_still.jpg" : "myr_still.jpg"), x: x + 0.3, y: 4.55, w: 5.3, h: 1.85, sizing: { type: "cover", w: 5.3, h: 1.85 }, objectName: `still${i}` });
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.75, w: 0.6, h: 0.6, fill: { color: col }, line: { color: col } });
  T(s, name, { x: x + 1.1, y: 1.7, w: 4.5, h: 0.45, fontSize: 24, bold: true, color: C.text2, fontFace: "Cambria" });
  T(s, sub, { x: x + 1.1, y: 2.15, w: 4.5, h: 0.35, fontSize: 14, italic: true, color: col });
  s.addText(bullets(items, 15), { isTextBox: true, x: x + 0.3, y: 2.7, w: 5.3, h: 1.8, valign: "top", color: C.text1, margin: 0 });
});

// 6 Architektur
s = content("Hierarchische Steuerung: vom Entschluss zur Bewegung", "Aufbau");
const steps = [["Navigationsgehirn", "10 Hz", "entscheidet: jagen, rasten, wohin\nAusgabe: Tempo + Drehrate", C.accent1],
  ["Laufgehirn", "50 Hz", "setzt Befehl in Gelenkziele um\n18 (Myr) / 29 (DC) Motoren", C.accent2],
  ["MuJoCo-Physik", "200 Hz", "Kräfte, Kontakte, Gleichgewicht\nNetz-Körper aus dem Modell", C.accent4],
  ["Blender", "30 fps", "Knochen folgen den Gelenkwinkeln\nKameras, HUD, Rendering", C.accent3]];
steps.forEach(([t, hz, d, col], i) => {
  const x = M + i * 3.1;
  card(s, x, 1.7, 2.8, 3.0, C.background2, `arch${i}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 1.9, w: 0.55, h: 0.55, fill: { color: col }, line: { color: col } });
  T(s, String(i + 1), { x: x + 0.2, y: 1.9, w: 0.55, h: 0.55, align: "center", valign: "middle", bold: true, color: C.background1 });
  T(s, t, { x: x + 0.2, y: 2.6, w: 2.45, h: 0.45, fontSize: 17, bold: true, color: C.text2 });
  T(s, hz, { x: x + 0.2, y: 3.05, w: 2.45, h: 0.35, fontSize: 15, bold: true, color: col });
  T(s, d, { x: x + 0.2, y: 3.5, w: 2.45, h: 1.1, fontSize: 13, color: C.text1 });
  if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.83, y: 3.05, w: 0.25, h: 0.35, fill: { color: "8FA89B" }, line: { color: "8FA89B" } });
});
T(s, "Beide Gehirne sind neuronale Netze (2 × 256 Neuronen), trainiert mit PPO. Das Laufgehirn lernt nur aus Belohnung und Strafe, ohne vorgegebene Animation (Entscheidung vom 25.09.).", { x: M, y: 5.1, w: 12.1, h: 0.8, fontSize: 15, color: C.text2 });

// 7 Habitat
s = content("Lebensraum und Spielregeln", "Aufbau");
card(s, M, 1.5, 5.8, 5.1, C.background2, "habitat");
T(s, "Habitat 100 × 100 m", { x: M + 0.3, y: 1.7, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
s.addText(bullets(["2 Reviere (Radius 28 m), überlappen an einer Wasserstelle", "6 Rastplätze, 14 Felsen, Mauern (Showcase ohne Felsen)", "Beute: rote Punkte, wandern 0,15 m/s, fliehen 0,35 m/s", "Je Revier 5 Beute, 3 neutral"], 14), { isTextBox: true, x: M + 0.3, y: 2.25, w: 5.2, h: 2.2, valign: "top", margin: 0 });
T(s, "Bedürfnisse", { x: M + 0.3, y: 4.5, w: 5.2, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
s.addText(bullets(["Hunger steigt mit Energieverbrauch, Beute −0,5", "Müdigkeit: ab 100 % nur noch 30 % Tempo"], 14), { isTextBox: true, x: M + 0.3, y: 5.0, w: 5.2, h: 1.4, valign: "top", margin: 0 });
card(s, M + 6.2, 1.5, 5.9, 5.1, C.background2, "rules");
T(s, "Revier-Regeln (Eigentümer-Entscheidungen)", { x: M + 6.5, y: 1.7, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
s.addText(bullets(["Beute im eigenen Revier gefressen: kommt nicht zurück", "Beute im fremden Revier gefressen: erscheint im eigenen Revier neu (Raubzüge bringen Futter heim)", "Neutrale Beute: kommt nach 60–120 s zurück", "Beide dürfen ins fremde Revier, kleiner Malus (−0,01 pro 0,1 s)", "Rivale ist im Training anwesend"], 14), { isTextBox: true, x: M + 6.5, y: 2.25, w: 5.4, h: 4.2, valign: "top", margin: 0 });

// ================================================================== Abschnitt Dimensionen
pres.addSection({ title: "Dimensionen & Parameter" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Dimensionen & Parameter" });
s.addText("2 · Dimensionen & Parameter", { placeholder: "title" });
s.addText("Körper, Physik und Trainings-Einstellungen", { placeholder: "body" });

// 9 Dimensionen-Tabelle
s = content("Körper in Zahlen (Stand: Myrranox v10, DC v12)", "Dimensionen & Parameter");
table(s, [["", "Myrranox", "DC"],
  ["Masse / Größe", "8 kg · 1,5 m lang", "60 kg · 1,3 m hoch"],
  ["Gliedmaßen", "6 Beine", "4 Beine + 2 Arme"],
  ["Gelenkmotoren (Aktionen)", "18", "29 (Beine 16, Arme 8, Wirbelsäule 2, Hals/Kopf 3)"],
  ["Federgelenke (passiv)", "Hinterleib 2, Kopf 2", "Schwanz 4"],
  ["Beobachtung (Eingänge)", "79 Werte", "112 Werte"],
  ["Physik-Formen", "21 Netz-Hüllen", "218 Teile aus dem Detail-Modell (CoACD)"],
  ["Gewichtsverteilung", "Hinterleib 23 % (Dichte ×0,25), Brust, Kopf, Beine", "Brust 29 %, Heck 23 %, Hinterbeine 18 %, Vorderbeine 12 %, Arme 6 %, Schwanz 5,5 %, Hals+Kopf 6,5 %"],
  ["Befehlsbereich je Gelenk", "±50°", "Beine ±34°, Arme ±17° (T-Rex-Haltung), Wirbelsäule ±14°"],
  ["Tempo (Gehen / max.)", "0,6 / 1,2 m/s", "0,8 / 2,5 m/s (Sprint 4,0)"]], { y: 1.45, colW: [3.4, 4.1, 4.6] }, 15, 0.5);
notes(s, "Myrranox: Masse nach Hüllenvolumen × Dichte (Hinterleib ×0,25). DC v12: feste Anteile nach Großkatzen-Vorbild, am 14.10. nach hinten verschoben (vorn/hinten 50/50). Zahlen gerundet.");

// 10 Physik-Körper Myrranox
s = content("Physik-Körper aus dem echten Modell: Myrranox", "Dimensionen & Parameter");
s.addImage({ path: path.join(__dirname, "myrranox_physics_body_mesh_deck.jpg"), x: M, y: 1.4, w: 6.9, h: 6.9 * 1240 / 1650, objectName: "myr_hulls" });
card(s, 7.8, 1.4, 4.9, 5.2, C.background2, "myr_hull_text");
s.addText(bullets(["Pro Knochen eine konvexe Hülle direkt aus dem Netz geschnitten (blau)", "Krallen sind die Füße: Kontakt an der echten Sohle", "Rechte Beine = Spiegelbild der linken (Modell war 3 cm asymmetrisch)", "Hinterleib mit ¼ Dichte: 54 % → 23 % Gewicht, Schwerpunkt nach vorn", "Federn skalieren mit der echten Masse"], 14), { isTextBox: true, x: 8.05, y: 1.65, w: 4.45, h: 4.8, valign: "top", margin: 0 });

// 11 Physik-Körper DC
s = content("Physik-Körper aus dem echten Modell: DC", "Dimensionen & Parameter");
s.addImage({ path: path.join(__dirname, "dc_physics_body_mesh_deck.jpg"), x: M, y: 1.4, w: 6.9, h: 6.9 * 1240 / 1650, objectName: "dc_hulls" });
card(s, 7.8, 1.4, 4.9, 5.2, C.background2, "dc_hull_text");
s.addText(bullets(["24 Hüllen statt 30 Kapseln, Platzierung exakt (0 mm Fehler)", "Stacheln und Hörner abgeschnitten: kein Volumen ohne Masse", "Rumpf am Wirbelsäulengelenk geteilt, Schulterbuckel zur Brust", "Arme als echte Gliedmaßen: Schulter, Ellbogen, Handgelenk", "Beine seitlich nur ±8°: Pfoten unter dem Körper"], 14), { isTextBox: true, x: 8.05, y: 1.65, w: 4.45, h: 4.8, valign: "top", margin: 0 });

// 12 Standhaltung
s = content("Natürliche Standhaltung als Nullpunkt", "Dimensionen & Parameter");
s.addImage({ path: FIG("standing_pose_v9.png"), x: M, y: 1.4, w: 7.4, h: 7.4 * 840 / 1280, objectName: "pose" });
card(s, 8.3, 1.4, 4.4, 4.85, C.background2, "pose_text");
T(s, "Warum?", { x: 8.55, y: 1.6, w: 4, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
T(s, "Die Gehirne mussten Gelenke an die Grenze drücken, nur um richtig zu stehen; für das Laufen blieb kein Spielraum.", { x: 8.55, y: 2.05, w: 3.95, h: 1.3, fontSize: 14 });
s.addText(bullets(["Myrranox: vorn/Mitte 6 cm, hinten 12 cm höher → eben, alle 6 Füße am Boden, Hinterleib frei", "DC: Arme angelegt, Vorderbeine 3 cm länger → Neigung 4,1°"], 14), { isTextBox: true, x: 8.55, y: 3.45, w: 3.95, h: 2.6, valign: "top", margin: 0 });

// 13 Trainings-Parameter
s = content("Trainings-Parameter (PPO)", "Dimensionen & Parameter");
table(s, [["Parameter", "Wert", "Bedeutung"],
  ["Algorithmus", "PPO (Stable-Baselines3) + Symmetrie-Verlust", "Links/rechts gespiegelte Situationen → gespiegelte Antwort"],
  ["Netz", "2 × 256 Neuronen, ELU, getrennt für Aktion/Wert", "pi und vf je 256-256"],
  ["Parallele Welten", "8 Umgebungen", "SubprocVecEnv"],
  ["Rollout / Batch / Epochen", "512 Schritte × 8 / 1024 / 5", "4096 Schritte pro Update"],
  ["Lernrate · γ · λ · Clip", "3·10⁻⁴ · 0,99 · 0,95 · 0,2", "target_kl 0,03, max_grad_norm 1,0"],
  ["Normierung", "VecNormalize (Beobachtung + Belohnung)", "bei Belohnungswechsel zurücksetzen"],
  ["Takt", "Physik 200 Hz, Steuerung 50 Hz", "4 Physikschritte pro Entscheidung"],
  ["Aktionsfilter", "Ziel += 0,5 · (Aktion − Ziel)", "≈ 5,5 Hz Grenzfrequenz: keine Sprünge"],
  ["Episode", "20 s = 1000 Schritte", "Befehle wechseln alle 2–5 s, als Rampe"],
  ["Checkpoints", "alle 250.000 Schritte", "alle behalten (Videos, Rückfall)"]], { y: 1.45, colW: [3.0, 4.6, 4.5] }, 14, 0.46);

// ================================================================== Abschnitt Regeln
pres.addSection({ title: "Regeln" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Regeln" });
s.addText("3 · Regeln", { placeholder: "title" });
s.addText("Wofür es Punkte gibt und wofür Abzug", { placeholder: "body" });

// 15 Formel
s = content("Die Belohnungsformel (Version 9)", "Regeln");
const parts = [["Tracking", "0 … 1", "Tempo und Drehung wie befohlen; Toleranz wächst mit dem Befehl", C.accent1],
  ["0,2 + 0,8 × Tierähnlichkeit", "0,2 … 1", "Gangtakt, Beinrhythmus, Tiermerkmale", C.accent2],
  ["Qualität", "0 … 1", "exp(−Σ Kosten): Ruck, Energie, Rutschen, Kippen …", C.accent4]];
parts.forEach(([t, r, d, col], i) => {
  const x = M + i * 4.15;
  card(s, x, 1.6, 3.75, 2.6, C.background2, `formula${i}`);
  T(s, t, { x: x + 0.25, y: 1.8, w: 3.3, h: 0.8, fontSize: 19, bold: true, color: col, fontFace: "Cambria" });
  T(s, r, { x: x + 0.25, y: 2.6, w: 3.3, h: 0.35, fontSize: 15, bold: true, color: C.text2 });
  T(s, d, { x: x + 0.25, y: 3.0, w: 3.3, h: 1.1, fontSize: 13 });
  if (i < 2) T(s, "×", { x: x + 3.75, y: 2.5, w: 0.4, h: 0.6, fontSize: 32, bold: true, color: C.text2, align: "center" });
});
card(s, M, 4.5, 12.1, 2.1, C.background2, "why");
T(s, "Warum multiplizieren statt abziehen?", { x: M + 0.3, y: 4.7, w: 11.5, h: 0.4, fontSize: 18, bold: true, color: C.accent3 });
T(s, "Früher wurden Strafen abgezogen: Bei voller Stärke verlor Stillstehen weniger als Laufen, und die Gehirne blieben stehen (Myrranox v3, DC v8). Jetzt verkleinern Strafen nur die Punkte, die Laufen einbringt. Wer bei Lauf-Befehl stillsteht, bekommt fast nichts. Ein Sturz (Neigung > 25°) beendet die Episode mit −10.", { x: M + 0.3, y: 5.15, w: 11.5, h: 1.3, fontSize: 14 });

// 16 Pluspunkte
s = content("Pluspunkte: wie ein echtes Tier laufen", "Regeln");
[["Myrranox (Insekt)", C.accent2, ["Dreifuß-Takt: zwei Dreiergruppen im Wechsel", "Hüftschwung 30–40°, Knie im Schwung 40–60° gebeugt", "Vorderbeine greifen vor, Hinterbeine schieben", "Drehen auf der Stelle: Seiten gegenläufig", "Füße heben und setzen im Takt der Gangtakt-Uhr"]],
 ["DC (Panther)", C.accent4, ["Lateraler Viertakt: hinten L → vorn L → hinten R → vorn R", "Langsame, lange Schritte (~1 Schritt/s beim Gehen)", "Hinterpfote landet in der Spur der Vorderpfote", "Wolf-Vorbild: tiefer Kopf, Pfoten flach und niedrig, Handgelenk klappt im Schwung", "Arme wie ein T-Rex vor der Brust, nie am Boden"]]].forEach(([t, col, items], i) => {
  const x = M + i * 6.2;
  card(s, x, 1.5, 5.9, 4.4, C.background2, `plus${i}`);
  dot(s, x + 0.3, 1.7, "+", col, 0.55);
  T(s, t, { x: x + 1.05, y: 1.75, w: 4.6, h: 0.45, fontSize: 20, bold: true, color: C.text2 });
  s.addText(bullets(items, 15), { isTextBox: true, x: x + 0.3, y: 2.5, w: 5.3, h: 3.3, valign: "top", margin: 0 });
});
T(s, "Für beide: Tempo und Drehung treffen (Tracking) · alle Gewichte als Tierähnlichkeit gemittelt (Gangtakt, Rhythmus, Pfotenspur, Arme)", { x: M, y: 6.15, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: C.text2 });

// 17 Minuspunkte
s = content("Abzüge: Qualitätskosten (Gewicht c je Kostenart)", "Regeln");
table(s, [["Kosten", "c", "Gemessen als", "Kosten", "c", "Gemessen als"],
  ["Energie", "0,08", "mechanische Leistung / (m·g·v)", "Rumpf rollen/nicken", "0,3", "Winkelgeschwindigkeit²"],
  ["Ruck", "1,0", "(Gelenkbeschleunigung / 500)²", "Wippen", "3,0", "vertikale Rumpfgeschwindigkeit²"],
  ["Befehlssprünge", "1,0", "Änderung der gefilterten Ziele²", "Kippen", "6,0", "statische Neigung"],
  ["Rutschen", "1,0", "Fußgeschwindigkeit am Boden²", "Beschleunigungs-Pumpen", "0,2", "Tempoänderung im Schritt"],
  ["Tippen", "1,0", "Aufsetzen nach < halbem Schwung", "Harte Landung", "0,05", "vertikale Fußgeschw.²"],
  ["Unbenutztes Bein", "0,5", "> 2 Zyklen ohne Bodenkontakt", "Zappeln im Stand", "0,5", "Gelenkgeschw.² bei Stand"],
  ["Hinterleib am Boden", "1,0", "Myrranox", "Arm am Boden", "1,0", "DC, beim Gehen"],
  ["Pfotenspur zu breit", "1,0", "DC: breiter als im Stand", "Kopf wackelt/kippt", "0,02/2,0", "DC"],
  ["Rumpf am Boden", "2,0", "Körperkontakt", "Gelenk an Grenze", "0,5", "Anteil nahe Grenze"]], { y: 1.45, colW: [2.3, 0.75, 3.05, 2.45, 0.9, 2.65] }, 14, 0.46);
T(s, "Qualität = exp(−Σ c · Kosten). Viele Gewichte wurden nach Messungen verschärft (z. B. Kippen und Wippen ×3, Rutschen und Tippen ×3).", { x: M, y: 6.35, w: 12.1, h: 0.45, fontSize: 13, italic: true, color: C.text2 });

// 18 Curricula / harte Regeln
s = content("Harte Regeln, Lehrpläne und Hilfen", "Regeln");
const rules = [["Sturz", "Neigung > 25° oder Rumpf zu tief → Episode endet, −10 (verhindert z. B. den DC-Handstand)"],
  ["Strafen-Rampe", "Strafen starten bei 20 % und steigen bis 40 % der Trainingszeit auf 100 %"],
  ["Toleranz-Rampe", "Tempo-Toleranz wird zwischen 20 % und 80 % des Laufs strenger (1,0 → 0,5–0,7)"],
  ["Befehls-Stufen", "--max-command: erst Gehen, dann Trab, Lauf, Sprint"],
  ["Laufhilfe", "Bonus für jede Vorwärtsbewegung, ausgeblendet über 1,5 Mio. Schritte; nie für Stillstand"],
  ["Gelenk-Wächter", "meldet Gelenke, deren Befehl dauerhaft jenseits der Grenze liegt (kein Lernsignal mehr)"],
  ["Schrittgenerator (Myr v10)", "Dreifuß-Rhythmus fest vorgegeben, Gehirn korrigiert bis ±26°"]];
rules.forEach(([t, d], i) => {
  const y = 1.45 + i * 0.75;
  dot(s, M, y + 0.05, String(i + 1), i % 2 ? C.accent2 : C.accent1, 0.48);
  T(s, t, { x: M + 0.7, y: y + 0.1, w: 3.2, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  T(s, d, { x: M + 3.9, y: y + 0.12, w: 8.2, h: 0.55, fontSize: 14 });
});

// ================================================================== Abschnitt Prozess
pres.addSection({ title: "Prozess" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Prozess" });
s.addText("4 · Prozess", { placeholder: "title" });
s.addText("Vom ersten Schritt bis zum Tiergang: zehn Versionen in drei Wochen", { placeholder: "body" });

// 20 Zeitachse
s = content("Zeitachse des Projekts", "Prozess");
const tl = [["23.09", "Projektplan, Habitat, Blender-Grundlage"], ["25.09", "Laufgehirn v1 + v2, erste Raubtier-Gehirne"], ["26.09", "Raubtiere jagen, Showcase, Körper v3"],
  ["28.09", "v4 präzise & flüssig, Showcase-Videos"], ["29.09", "v5 natürlicher Stil, Knie befreit"], ["30.09", "v6–v8: sechsgliedriger DC, Netz-Körper, Sprint"], ["13.10", "v9–v11: neue Belohnung, Dreifuß-Generator, Myrranox final"], ["14.10", "DC v12: Detail-Körper, Wolf, T-Rex-Arme, Hand; DC final"], ["21.10", "Showcase v6 mit den finalen Läufern"]];
s.addShape(pres.shapes.LINE, { x: M + 0.3, y: 3.45, w: 11.6, h: 0, line: { color: "8FA89B", width: 2 } });
tl.forEach(([d, t], i) => {
  const x = M + 0.05 + i * 1.34;
  s.addShape(pres.shapes.OVAL, { x: x + 0.55, y: 3.25, w: 0.4, h: 0.4, fill: { color: i === 8 ? HEX.accent1 : HEX.accent2 }, line: { color: "FFFFFF", width: 2 } });
  T(s, d, { x: x - 0.08, y: 2.65, w: 1.5, h: 0.45, fontSize: 15, bold: true, color: C.text2, align: "center" });
  T(s, t, { x: x - 0.04, y: 3.9, w: 1.38, h: 1.9, fontSize: 11, align: "center" });
});
T(s, "Kalenderzeit 23.09.–28.10. · 10 Arbeitstage · Trainings und Renderings liefen oft über Nacht oder parallel zur Entwicklung", { x: M, y: 6.2, w: 12.1, h: 0.45, fontSize: 14, italic: true, color: C.text2 });

// 21 Versionen 1
s = content("Laufgehirn-Versionen v1–v5", "Prozess");
table(s, [["Version", "Datum", "Neu", "Ergebnis"],
  ["v1", "25.09", "Tempo-Belohnung, breite Toleranz, Strafen-Rampe", "beide laufen; Pilotläufe vorher: Stillstand-Falle"],
  ["v2", "25.09", "Gangtakt-Uhr, Dreifuß / Schritt→Trab, Spiegel-Symmetrie", "natürlicher Gang, ~5× (Myr) / 2,4× (DC) weniger Energie"],
  ["v3", "26.09", "Wirbelsäule, Hals, Kopf aktiv; ruhiger Kopf", "DC: Kopfwackeln −41 %; Myrranox: Stillstand (gescheitert)"],
  ["v4", "28.09", "Aktionsfilter, Toleranz-Rampe, Befehlsrampen", "Myr dreht auf der Stelle 7 → 58°/s; −25 % Ruck"],
  ["v5", "29.09", "Rhythmus-Referenz, Knie-Entlastung, Panther-Schritt", "Myr Knie 4,5° → 20° (war blockiert seit v2)"],
  ["v5c", "30.09", "Kopf ×3, strenges Tempo", "DC-Standard: 0,62 m Schritt, Kopf im Bild stabilisiert"]], { y: 1.45, colW: [1.0, 1.0, 5.0, 5.1] }, 15, 0.62);

// 22 Versionen 2
s = content("Laufgehirn-Versionen v6–v12", "Prozess");
table(s, [["Version", "Datum", "Neu", "Ergebnis"],
  ["v6", "30.09", "DC mit 6 Gliedmaßen, Beine ±8°, strengere Takt-Regeln", "Gehirn-Transplantat stand still; von Null abgebrochen"],
  ["v7", "30.09", "Schrittgenerator für den DC", "Rhythmus allein: Sturz alle 1,5 s (Vierbeiner instabil)"],
  ["v8", "30.09", "DC-Netz-Körper, Sprint 4 m/s, Brems-Rampen", "Handstand; Kollaps bei 3,2 Mio. (Strafen voll)"],
  ["v9", "13.10", "Multiplikative Belohnung, Netz-Körper beide, Standhaltung", "DC: Katzen-Fußfolge, 0,70 m/s; Myr: Schubkarre"],
  ["v10", "13.10", "Myr: Dreifuß-Generator + Korrekturen, leichter Hinterleib", "Myrranox final: 0,52 / 1,08 m/s, rückwärts, Knie 60°"],
  ["v11", "13.10", "DC: Großkatzen-Gewichte, runde Pfotenpolster", "kein Trippeln mehr, aber langsam (0,23 m/s)"],
  ["v12", "14.10", "DC: 218-Teile-Körper, Wolf-Regeln, T-Rex-Arme, neue Hand", "DC final (v12z): flüssig, Sprint 4 m/s, bremst sicher"]], { y: 1.45, colW: [1.0, 1.0, 5.0, 5.1] }, 15, 0.56);
T(s, "Stand 28.10.: Beide Läufer sind final (Eigentümer-Entscheidungen): Myrranox v10 (13.10.) und DC v12z (14.10.).", { x: M, y: 6.15, w: 12.1, h: 0.45, fontSize: 14, italic: true, color: C.text2 });

// 23 Raubtier-Gehirne
s = content("Raubtier-Gehirne: vom Verhungern zum Jagen", "Prozess");
table(s, [["Versuch", "Änderung", "Ergebnis", "Lehre"],
  ["v1", "Belohnung laut Plan", "blieb daheim, verhungerte (0 Beute)", "Jagen und Verhungern kaum unterscheidbar"],
  ["v2", "satt + jeder Fang + Jagd-Hinweis", "stand still (0 Beute)", "halber Aktionsbereich hieß „Stopp“"],
  ["Pilot", "voller Tempobereich, Flucht-Lehrplan", "0 → 2,5 Beute / 10 min", "Erkunden ermöglichen"],
  ["v3", "Runde 1, 3 Mio. Schritte", "4,0–4,4 Beute / 10 min", "seltene Lagen werden selten gelernt"],
  ["v4", "Vorrat, Eindringling, Hungertage", "DC freigegeben: 4,6 Beute, rastet", "Myrranox ermüdet 2× schneller"]], { y: 1.45, colW: [1.1, 3.5, 3.6, 3.9] }, 15, 0.62);
T(s, "Freigegeben: DC dc_brain_v4_r1 · Myrranox myr_brain_v3_r1. Raubzüge und Revierverteidigung wurden von keinem Gehirn gelernt. Spiel-Idee „3 Min jagen, dann 30 s zum Wasserloch“ (14.10.) gebaut und verworfen.", { x: M, y: 5.5, w: 12.1, h: 0.6, fontSize: 14, italic: true, color: C.text2 });

// ================================================================== Abschnitt DC v12
pres.addSection({ title: "DC v12" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "DC v12" });
s.addText("5 · DC v12", { placeholder: "title" });
s.addText("Körper wie das echte Modell, Wolf als Vorbild, Arme wie ein T-Rex, neue Hand (13./14.10.)", { placeholder: "body" });

// Detail-Körper
s = content("Physik-Körper so nah wie möglich am Modell", "DC v12");
s.addImage({ path: FIG("dc_physics_body_v11_vs_v12.jpg"), x: M, y: 1.3, w: 4.75, h: 4.75 * 1660 / 1400, objectName: "dc_v12_body" });
card(s, 5.65, 1.3, 7.05, 5.35, C.background2, "dc_v12_body_text");
s.addText(bullets(["Detail-Skulptur (953.000 Punkte) pro Knochen in konvexe Teile zerlegt (CoACD, wie MuJoCo-Modellsammlung)", "218 Teile statt 24 Blöcke: Stacheln, Hörner, offenes Maul, Schulterbuckel, Bauchlinie", "Stacheln und Hörner mit ⅓ Dichte (sonst Schwerpunkt 9 cm höher)", "Simulation fast gleich schnell: 1.235 statt 1.311 Schritte/s"], 14), { isTextBox: true, x: 5.9, y: 1.5, w: 6.6, h: 2.3, valign: "top", margin: 0 });
table(s, [["Teil", "Abstand zum Modell vorher", "nachher"], ["Rumpf / Hals", "18,1 / 13,4 cm", "3,3 / 4,3 cm"], ["Heck", "12,9 cm", "3,0 cm"], ["Oberschenkel", "7,7–10,9 cm", "3,1–4,2 cm"], ["Kopf / Hand", "6,9 / 3,9 cm", "3,7 / 1,4 cm"]],
  { x: 5.9, y: 4.05, w: 6.55, colW: [2.2, 2.4, 1.95] }, 13, 0.42);
notes(s, "Abstand = wie weit die Oberfläche der Physik-Teile vom echten Modell abweicht (95 % der Punkte). Links: alte Einzelhüllen (v11), rechts: 218 Teile (v12), über dem Modell und allein.");

// Wolf
s = content("Vorbild: Gangzyklus eines Wolfs", "DC v12");
s.addImage({ path: FIG("wolf_reference_frames.png"), x: M, y: 1.3, w: 12.1, h: 12.1 * 868 / 1840 * 0.62, sizing: { type: "cover", w: 12.1, h: 12.1 * 868 / 1840 * 0.62 }, objectName: "wolf" });
s.addText(bullets(["Jedes Bein einmal pro Zyklus: vorn und hinten gleicher Takt (DC hatte hinten getrippelt)", "Immer 2–3 Pfoten am Boden, Pfoten nur ~8 cm angehoben", "Vorderes Handgelenk klappt im Schwung unter", "Rücken eben, Kopf tief und nach vorn (Hals-Ruhehaltung 16° tiefer)"], 15), { isTextBox: true, x: M, y: 5.0, w: 12.1, h: 1.7, valign: "top", margin: 0 });
notes(s, "Vom Eigentümer bereitgestellte Bildschirmaufnahme eines Sketchfab-Wolf-Gangzyklus. Diese Regeln gelten nur beim Gehen bis 1,2 m/s; im Sprint hatten sie die Qualität auf 0 gedrückt.");

// T-Rex-Arme + Gewicht
s = content("Arme wie ein T-Rex, Gewicht nach hinten", "DC v12");
s.addImage({ path: FIG("dc_trex_arms_walk.png"), x: M, y: 1.3, w: 6.6, h: 6.6 * 840 / 1280, objectName: "trex" });
card(s, 7.5, 1.3, 5.2, 5.35, C.background2, "trex_text");
T(s, "Arme", { x: 7.75, y: 1.5, w: 4.8, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
s.addText(bullets(["Oberarm nach vorn, Hände vor der Brust, über dem Schwung der Vorderbeine", "Unterarm/Hand stoßen physisch an die Vorderbeine", "Nur ±17° Spielraum zum Ausbalancieren (Gehirn hielt die Handgelenke sonst 70–90° geknickt)"], 13), { isTextBox: true, x: 7.75, y: 1.95, w: 4.75, h: 2.2, valign: "top", margin: 0 });
T(s, "Gewicht (DC kippte vorn über)", { x: 7.75, y: 4.2, w: 4.8, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
s.addText(bullets(["Hinterbeine 7,5 → 9 % je Bein, Schwanz 3 → 5,5 %", "Kopf 4 → 2,5 %, Hals 5 → 4 %", "vorn/hinten 55/45 → 50/50"], 13), { isTextBox: true, x: 7.75, y: 4.65, w: 4.75, h: 1.9, valign: "top", margin: 0 });

// Hand
s = content("Die Hand neu modelliert", "DC v12");
s.addImage({ path: FIG("dc_hand_real_model_closeup.png"), x: M, y: 1.3, w: 3.6, h: 3.6, objectName: "hand_real" });
s.addImage({ path: FIG("dc_hand_v12_rig_folded_open.png"), x: M + 3.8, y: 1.3, w: 7.2, h: 3.6, objectName: "hand_rig" });
s.addText(bullets([{ text: "Problem: Im Modell hängt die Hand wie ein Haken unter dem gebogenen Unterarm. Das Physik-Handgelenk saß 10 cm zu hoch in der Biegung – die Hand konnte sich nie öffnen.", options: {} },
  { text: "Lösung: Handgelenk an der echten Stelle, Hand vom Gelenk zu den Fingerspitzen, öffnet bis 250° (Handfläche zum Boden); Kontaktpolster auf der Handfläche.", options: {} },
  { text: "Blender-Rig gleich umgebaut (Knochen + Gewichte), Physik-Teile der Hand neu: 3,9 → 1,4 cm Abstand.", options: {} }], 14), { isTextBox: true, x: M, y: 5.1, w: 12.1, h: 1.7, valign: "top", margin: 0 });
notes(s, "Links das echte Modell (rote Punkte: altes Physik-Gelenk), rechts das Rig mit gefaltet/geöffneter Hand.");

// Arme im Sprint
s = content("Arme im Sprint: versucht und verworfen", "DC v12");
s.addImage({ path: FIG("dc_sprint_pose_palms.png"), x: M, y: 1.3, w: 5.4, h: 5.4 * 420 / 640, objectName: "sprint_pose" });
card(s, 6.3, 1.3, 6.4, 5.35, C.background2, "sprint_text");
s.addText(bullets(["Idee: ab 2 m/s laufen die Arme mit, Brust 9 cm tiefer, Handflächen flach", "Weiches Handgelenk + Handflächen-Reflex, Handflächen- und Duck-Regel, eigenes Arm-Gehirn", "Ergebnis: DC bäumte sich auf (Nase 11° hoch), Hände nur auf den Fingerspitzen (−61°)", "Im Sprint mit vollen Strafen verlernte das Gehirn sogar das Gehen", { text: "Entscheidung: keine Arme im Sprint – T-Rex-Arme bei jedem Tempo, Sprint auf vier Beinen", options: { bold: true } }], 14), { isTextBox: true, x: 6.55, y: 1.55, w: 5.9, h: 5.0, valign: "top", margin: 0 });

// Stufen
s = content("Trainings-Stufen DC v12 (13./14.10.)", "DC v12");
table(s, [["Lauf", "Änderung", "Ergebnis"],
  ["v12", "218-Teile-Körper, Wolf-Regeln", "0,42 / 0,71 m/s, Takt sehr gleichmäßig, Hals am Anschlag"],
  ["v12b", "Kopf-Ruhehaltung tiefer", "Beine nach innen gezogen, 0,11 m/s (2 Regelfehler)"],
  ["v12c", "Regelfehler behoben", "bestes Gehen: 0,45 / 0,73 m/s, Takt ±1–5 %"],
  ["v12d–f", "T-Rex-Arme, Gewicht nach hinten", "0,37 / 0,61 m/s, Takt ±1 %"],
  ["v12s / h", "Sprint-Stufe, neue Hand", "Gehen 0,63 / 0,81 m/s; Hände auf Fingerspitzen"],
  ["v12p / q", "Handflächen-, Duck-Regel, Reflex", "Aufbäumen; v12q verlernt Gehen"],
  ["dc_arms_v1", "eigenes Arm-Gehirn", "gestoppt: Arme raus aus dem Sprint"],
  ["v12u–w", "Sprint auf vier Beinen; Planke = Sturz", "Planke weg, Sprint 2,2 m/s, aber kein langsames Gehen"],
  ["v12x–z", "Anlaufhilfe, Glätten (Wippen, Takt)", "final: flüssig, 0,59–3,6 m/s, bremst ohne Sturz"]], { y: 1.35, colW: [1.6, 4.4, 6.1] }, 13, 0.47);
T(s, "Tempo bei Befehl 0,6 / 1,0 m/s, Sprint bei 2,5 / 4,0 m/s. Laptop schlief mehrmals ein / Sitzung endete: ≈ 9 h Training verloren.", { x: M, y: 6.45, w: 12.1, h: 0.4, fontSize: 12, italic: true, color: C.text2 });

// DC final
s = content("Der finale DC (v12z): Planke weg, Gang geglättet", "DC v12");
card(s, M, 1.35, 5.9, 2.55, C.background2, "plank");
T(s, "Problem 1: „Planke“", { x: M + 0.25, y: 1.5, w: 5.4, h: 0.4, fontSize: 17, bold: true, color: C.accent3 });
s.addText(bullets(["Beim Stehen und langsamen Gehen: Beine weit gespreizt, Rumpf 16–18 cm tief", "Sehr stabil, also billig – keine Regel verlangte eine natürliche Haltung", "Lösung: Rumpf > 12 cm zu tief = Sturz; Haltungs-Kosten ohne Deckel"], 13), { isTextBox: true, x: M + 0.25, y: 1.95, w: 5.45, h: 1.9, valign: "top", margin: 0 });
card(s, M + 6.2, 1.35, 5.9, 2.55, C.background2, "clunky");
T(s, "Problem 2: „klobig statt flüssig“", { x: M + 6.45, y: 1.5, w: 5.4, h: 0.4, fontSize: 17, bold: true, color: C.accent3 });
s.addText(bullets(["Kurze hastige Schritte (0,28 m), Takt ±37 %, 6 cm Wippen", "Ursache: Anlaufhilfe belohnte jedes Vorwärts-Ruckeln", "Lösung: Glätt-Stufe – 70 % Gehen, Wippen und Tempo-Schwanken teurer, Takt doppelt"], 13), { isTextBox: true, x: M + 6.45, y: 1.95, w: 5.45, h: 1.9, valign: "top", margin: 0 });
table(s, [["Befehl", "Tempo", "Schritte/s · Takt-Schwankung", "Schrittlänge · Wippen"],
  ["0,6 m/s", "0,59 ± 0,14 m/s", "1,12 · 9 %", "0,52 m · 1,8 cm"],
  ["1,0 m/s", "0,86 ± 0,17 m/s", "1,07 · 1 %", "0,81 m · 2,0 cm"],
  ["2,5 / 4,0 m/s", "1,84 / 3,56 m/s (Spitze 4,1–4,2)", "1,89 · 2 %", "0,97 m · 1,7 cm"]], { y: 4.15, colW: [2.0, 3.8, 3.3, 3.0] }, 14, 0.47);
T(s, "Bremsen aus 4 m/s: 3 von 3 ohne Sturz · Kurven +31 / −28 °/s · keine Stürze · schwach: Drehen auf der Stelle (33 statt 57 °/s)", { x: M, y: 6.15, w: 12.1, h: 0.5, fontSize: 13, italic: true, color: C.text2 });

// ================================================================== Abschnitt Analyse
pres.addSection({ title: "Analyse" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Analyse" });
s.addText("6 · Analyse", { placeholder: "title" });
s.addText("Lernkurven, Fußfolgen und Messwerte gegen die Ziele", { placeholder: "body" });

// 25 Lernkurven
s = content("Lernkurven der Lauf-Trainings", "Analyse");
s.addImage({ path: FIG("locomotion/learning_curves_myrranox.png"), x: M, y: 1.4, w: 5.95, h: 5.95 * 750 / 1350, objectName: "lc_myr" });
s.addImage({ path: FIG("locomotion/learning_curves_dc.png"), x: M + 6.15, y: 1.4, w: 5.95, h: 5.95 * 750 / 1350, objectName: "lc_dc" });
T(s, "Belohnung pro Episode über die Trainingsschritte, alle Läufe inkl. gescheiterter Pilotläufe. Werte zwischen Versionen nicht vergleichbar, die Belohnung änderte sich jedes Mal.", { x: M, y: 4.95, w: 12.1, h: 0.8, fontSize: 14, color: C.text2 });

// 26 Fußfolgen
s = content("Fußfolge-Diagramme aller Gehirne", "Analyse");
s.addImage({ path: FIG("gait/overview_myrranox.png"), x: M, y: 1.35, w: 7.0, h: 7.0 * 1325 / 2180, objectName: "ff_myr" });
s.addImage({ path: FIG("gait/overview_dc.png"), x: M + 7.25, y: 1.35, w: 4.85, h: 4.85 * 1230 / 1530, objectName: "ff_dc" });
T(s, "Dunkel = Fuß am Boden, je 8 s. Links Myrranox (Gehen, Vollgas, Drehen auf der Stelle), rechts DC (Gehen, Trab).", { x: M, y: 5.85, w: 12.1, h: 0.6, fontSize: 13, color: C.text2 });

// 27 Messwerte Myrranox
s = content("Messung gegen die Ziele: Myrranox", "Analyse");
table(s, [["Messwert", "v5 (vorher)", "v10 (final)", "Ziel"],
  ["Gehen / Vollgas (Befehl 0,6 / 1,2)", "0,59 / 1,13 m/s", "0,52 / 1,08 m/s", "0,6 / 1,2 m/s"],
  ["Drehen gehend / auf der Stelle", "+65 −58 / 65 °/s", "+57 −49 / 56 °/s", "±54 / 72 °/s"],
  ["Rückwärts (Befehl −0,3)", "−0,01 (nie gelernt)", "−0,26 m/s", "−0,3 m/s"],
  ["Rumpfneigung / Wippen", "–", "2,0° / 1,4 cm", "< 5°"],
  ["Knie / Hüftschwung", "20° / 24°", "60° / 28°", "40–60° / 30–40°"],
  ["Dreifuß-Gruppen im Wechsel", "–", "87 %", "hoch"],
  ["Hinterleib am Boden", "–", "0 %", "~0 %"],
  ["Stürze", "0", "0", "0"]], { y: 1.45, colW: [3.6, 2.9, 2.8, 2.8] }, 15, 0.46);
T(s, "Offener Punkt (als final akzeptiert): Vorderbeine nur 21 % am Boden und trippeln 5–6 Schritte/s.", { x: M, y: 6.2, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: C.accent3 });

// 28 Messwerte DC
s = content("Messung gegen die Ziele: DC", "Analyse");
table(s, [["Messwert", "v9 (13.10.)", "v11", "v12z (final)", "Ziel"],
  ["Tempo bei Befehl 0,6 m/s", "0,65 m/s", "0,23 m/s", "0,59 m/s", "0,6 m/s"],
  ["Tempo bei Befehl 1,0 / Sprint 4,0", "–", "0,80 (bei 1,2)", "0,86 / 3,56 m/s", "1,0 / 4,0 m/s"],
  ["Schritte/s je Bein (Gehen)", "hinten 2–4 (Trippeln)", "1,0 alle", "1,07–1,12, Takt ±1–9 %", "≈ 1, gleich vorn/hinten"],
  ["Rumpfneigung", "3,7°", "≤ 2°", "≤ 3°", "< 5°"],
  ["Körper-Form", "24 Hüllen", "24 Hüllen", "218 Teile", "wie das Modell"],
  ["Arme", "angelegt", "angelegt", "T-Rex vor der Brust", "T-Rex"],
  ["Stürze", "0", "0", "0", "0"]], { y: 1.45, colW: [3.3, 2.2, 1.9, 2.6, 2.1] }, 14, 0.5);
T(s, "Erreicht: flüssiger Wolfs-ähnlicher Gang, Sprint und sicheres Bremsen. Schwach: Drehen auf der Stelle; Hinterpfote nicht in der Vorderpfoten-Spur.", { x: M, y: 6.2, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: C.accent3 });

// 29 Analysewerkzeuge
s = content("Analysewerkzeuge", "Analyse");
const tools = [["precision_metrics.py", "Tempo- und Drehfehler pro Befehlsabschnitt"], ["style_metrics.py", "Bewegungsumfang, Schrittlänge, Fußhöhe, Pfotenspur"],
  ["alive_metrics.py", "Kopf- und Rumpfwackeln, Ruck, Energie"], ["gait_analysis.py / gait_overview.py", "Fußfolge-Diagramme aller Gehirne"],
  ["Gelenk-Wächter + unsaturate_brain.py", "blockierte Gelenke finden und befreien"], ["Überlagerungs-Renderings", "Physik-Hüllen über dem Blender-Modell"],
  ["watch_locomotion.py", "MuJoCo-Live-Fenster je Training"], ["Trainings-Videos", "jeder Checkpoint, ungeübt bis fertig, neben der Lernkurve"]];
tools.forEach(([t, d], i) => {
  const col = i % 2, row = Math.floor(i / 2), x = M + col * 6.2, y = 1.5 + row * 1.25;
  card(s, x, y, 5.9, 1.05, C.background2, `tool${i}`);
  T(s, t, { x: x + 0.25, y: y + 0.12, w: 5.4, h: 0.4, fontSize: 15, bold: true, color: col ? C.accent2 : C.accent1 });
  T(s, d, { x: x + 0.25, y: y + 0.52, w: 5.4, h: 0.4, fontSize: 13 });
});

// ================================================================== Abschnitt Tools & Laufzeiten
pres.addSection({ title: "Tools & Laufzeiten" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Tools & Laufzeiten" });
s.addText("7 · Tools & Laufzeiten", { placeholder: "title" });
s.addText("Software, Hardware und wohin die Rechenzeit ging", { placeholder: "body" });

// 31 Software
s = content("Werkzeuge und Umgebung", "Tools & Laufzeiten");
const stack = [["Simulation", "MuJoCo 3 · Gymnasium-Umgebungen", C.accent4], ["Lernen", "Stable-Baselines3 (PPO) · PyTorch (CPU)", C.accent2], ["3D & Video", "Blender 5.2 (EEVEE) · imageio-ffmpeg", C.accent3],
  ["Analyse", "TensorBoard · NumPy · SciPy · Matplotlib", C.accent1], ["Entwicklung", "Python 3.12 · Git + Git LFS · Claude Code", C.accent4], ["Hardware", "8 CPU-Kerne · keine CUDA-GPU · Windows 11", C.accent2]];
stack.forEach(([t, d, col], i) => {
  const c2 = i % 3, r = Math.floor(i / 3), x = M + c2 * 4.1, y = 1.5 + r * 2.5;
  card(s, x, y, 3.8, 2.2, C.background2, `stack${i}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col } });
  T(s, t, { x: x + 1.0, y: y + 0.35, w: 2.6, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  T(s, d, { x: x + 0.3, y: y + 1.0, w: 3.3, h: 1.0, fontSize: 14 });
});

// 32 Laufzeiten-Diagramm
s = content("Rechenzeit nach Aufgabe (Wandzeit, ca.)", "Tools & Laufzeiten");
s.addChart(pres.charts.BAR, [{ name: "Stunden", labels: ["Lauf v1–v2", "Raubtier-Gehirne", "Lauf v4–v5c", "Lauf v6–v8", "Lauf v9–v11 (13.10.)", "DC v12 (13./14.10.)", "Showcase-Renderings"], values: [6.8, 5.5, 5.0, 3.0, 9.5, 9.0, 20.1] }], {
  x: M, y: 1.4, w: 7.4, h: 5.2, barDir: "bar", chartColors: [HEX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
  dataLabelFormatCode: "0.0 \"h\"", catAxisLabelColor: HEX.dk1, valAxisLabelColor: "6B7A75", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  valGridLine: { color: "E3E8E5", size: 0.5 }, catGridLine: { style: "none" }, showLegend: false, showTitle: false, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt" });
card(s, 8.3, 1.4, 4.4, 5.2, C.background2, "runtime_text");
s.addText(bullets(["Lauf-Training gesamt ≈ 38 h, meist 2 Läufe parallel", "Raubtier-Training ≈ 5,5 h (24,5 Mio. Schritte)", "4 Showcase-Videos à 5 min: je 2 × 9.001 Bilder, 4,3–5,7 h; Clip v6 (30 s): 16 min", "Arbeitssitzungen ≈ 88 h an 10 Tagen; ≈ 9 h Training durch Schlafmodus / Sitzungsende verloren", "Projektplan: 9–13 Wochen geschätzt"], 14), { isTextBox: true, x: 8.55, y: 1.65, w: 3.95, h: 4.8, valign: "top", margin: 0 });
notes(s, "Werte gerundet aus TIME_LOG und Trainingsprotokollen; v9–v10 inkl. der abgebrochenen Stufe-1-Versuche vom 13.10.");

// 33 Geschwindigkeiten
s = content("Wie schnell rechnet der PC?", "Tools & Laufzeiten");
const speed = [["650–1.700", "Trainingsschritte/s je Lauf (Kapsel-Körper)"], ["−25 %", "langsamer mit Netz-Körpern"], ["≈ 1,4 s", "pro Bild und Kamera (Blender EEVEE 960 × 540)"], ["≈ 1 h", "für 3 Mio. Schritte am Netzteil – im Akkubetrieb ≈ 2,5× länger"]];
speed.forEach(([big, label], i) => {
  const x = M + (i % 2) * 6.2, y = 1.5 + Math.floor(i / 2) * 2.4;
  card(s, x, y, 5.9, 2.1, C.background2, `speed${i}`);
  T(s, big, { x: x + 0.3, y: y + 0.25, w: 5.3, h: 0.9, fontFace: "Cambria", fontSize: 40, bold: true, color: i % 2 ? C.accent3 : C.accent1 });
  T(s, label, { x: x + 0.3, y: y + 1.2, w: 5.3, h: 0.7, fontSize: 15, color: C.text2 });
});
T(s, "Offene MuJoCo-Live-Fenster und TensorBoard bremsen das Training spürbar. Wird die Sitzung geschlossen, stoppen Trainings und Renderings; Checkpoints und Einzelbilder machen das Fortsetzen billig.", { x: M, y: 6.3, w: 12.1, h: 0.6, fontSize: 13, italic: true, color: C.text2 });

// Showcase v6
s = content("Showcase v6: beide Kreaturen mit den finalen Läufern", "Tools & Laufzeiten");
s.addImage({ path: FIG("showcase_v6_still.jpg"), x: M, y: 1.35, w: 12.1, h: 12.1 * 544 / 1928, objectName: "showcase_v6" });
s.addText(bullets(["Myrranox v10 (links) und DC v12z (rechts) im Habitat ohne Felsen, Raubtier-Gehirne entscheiden, Laufgehirne bewegen die Beine", "5 Minuten aufgenommen: beide überleben, je 4 Beute, kein Sturz (2 Min Rechenzeit)", "30-s-Clip gerendert (videos/showcase_v6_30s.mp4); die vollen 5 Minuten sind offen", "Myrranox noch ohne Texturen (rosa)"], 14), { isTextBox: true, x: M, y: 5.0, w: 12.1, h: 1.8, valign: "top", margin: 0 });

// Rendering
s = content("Warum das Rendern so lange dauert", "Tools & Laufzeiten");
table(s, [["Einstellung", "s / Bild", "5-Min-Video (2 Kameras)"],
  ["16 Samples, weiche Schatten, 30 fps (bis v5)", "0,83", "≈ 4,2 h (gemessen 4,3–5,7 h)"],
  ["8 Samples", "0,71", "≈ 3,6 h"],
  ["8 Samples, harte Schatten", "0,56", "≈ 2,8 h"],
  ["+ 24 fps (Showcase v6)", "≈ 0,6", "≈ 2,2 h"],
  ["4 Samples", "0,55", "sichtbares Rauschen – nicht genutzt"],
  ["eigene Grafikkarte (Desktop / Cloud, Schätzung)", "0,05–0,1", "≈ 15–30 min"]], { y: 1.4, colW: [6.0, 1.8, 4.3] }, 15, 0.5);
s.addText(bullets(["Grund: Der Laptop rendert mit der eingebauten Grafik (Intel Iris Xe), nicht mit einer Grafikkarte", "18.000 Bilder für 5 Minuten aus zwei Kameras; jedes Bild: Rig, Netz-Verformung, HUD, PNG", "Größter Hebel: ein Rechner mit eigener Grafikkarte (10–20× schneller)"], 14), { isTextBox: true, x: M, y: 5.2, w: 12.1, h: 1.5, valign: "top", margin: 0 });

// ================================================================== Abschnitt Erfahrungen
pres.addSection({ title: "Erfahrungen" });
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Erfahrungen" });
s.addText("8 · Erfahrungen", { placeholder: "title" });
s.addText("Was gut lief, was schiefging und was wir gelernt haben", { placeholder: "body" });

// 35 Positiv
s = content("Positive Erlebnisse", "Erfahrungen");
const pos = [["Beide laufen per RL", "keine vorgegebene Animation, alles gelernt"], ["v2: natürlicher Gang", "Dreifuß und Trab, bis 5× weniger Energie"], ["Raubtiere jagen", "DC 4,6 Beute / 10 min, rastet bei Müdigkeit"],
  ["Myrranox dreht auf der Stelle", "von 7 auf 58–65°/s (v4/v5)"], ["DC v12z final", "flüssiger Gang, Sprint 4 m/s, bremst sicher"], ["Myrranox v10 final", "Dreifuß, rückwärts, dreht auf der Stelle"],
  ["DC-Körper wie das Modell", "218 Teile, Hand und Rig neu modelliert"], ["Alles dokumentiert", "29 Trainingsvideos, Messungen, Protokolle"]];
pos.forEach(([t, d], i) => {
  const col = i % 2, row = Math.floor(i / 2), x = M + col * 6.2, y = 1.45 + row * 1.3;
  card(s, x, y, 5.9, 1.1, C.background2, `pos${i}`);
  dot(s, x + 0.25, y + 0.28, "✓", C.accent2, 0.52);
  T(s, t, { x: x + 0.95, y: y + 0.13, w: 4.8, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  T(s, d, { x: x + 0.95, y: y + 0.55, w: 4.8, h: 0.4, fontSize: 13 });
});

// 36 Negativ
s = content("Negative Erlebnisse und ihre Lösung", "Erfahrungen");
table(s, [["Problem", "Was passierte", "Lösung"],
  ["Stillstand-Falle", "Stehen war billiger als Laufen (Pilot, Myr v3, DC v8)", "multiplikative Belohnung"],
  ["Blockierte Gelenke", "Knie-Befehl jenseits der Grenze, kein Lernsignal (v2–v4)", "Gelenk-Wächter, Befehle zurückschieben"],
  ["Handstand", "DC stützte sich auf Vorderbeine und Arme", "Kipp-Grenze 25°, Arme am Boden kosten"],
  ["Schubkarre", "Myr: Vorderbeine hoch, Mittelbeine paddeln", "jedes Bein muss mit, Körper korrigiert"],
  ["Rauschen als Ruck", "Erkundungs-Rauschen drückte Qualität auf 0,06", "nach dem Filter messen"],
  ["Alte Statistik", "Belohnungsnormierung vom alten Maßstab (σ ≈ 95)", "Statistik zurücksetzen"],
  ["Falsche Stürze", "Höhenreferenz der Pose statt echter Standhöhe", "Standhöhe messen"],
  ["Volumen ≠ Masse", "Hinterleib 54 %: Myr kippte auch im perfekten Takt", "¼ Dichte, 23 %"]], { y: 1.4, colW: [2.4, 5.5, 4.2] }, 15, 0.54);

// Negativ DC v12
s = content("Negative Erlebnisse DC v12 (13./14.10.)", "Erfahrungen");
table(s, [["Problem", "Was passierte", "Lösung"],
  ["Regel am Anschlag", "Gelenk-Grenzzone 5° bei ±8°-Gelenk: immer „an der Grenze“", "Zone skaliert mit dem Bereich"],
  ["Nur eine Seite bestraft", "nur zu breite Pfoten kosteten: Beine nach innen gezogen", "zu eng kostet auch"],
  ["Geh-Regeln im Sprint", "Wolf-Regeln (Pfoten tief, 2 am Boden) drückten Sprint-Qualität auf 0", "nur bis 1,2 m/s"],
  ["Neue Fähigkeit zu teuer", "volle Sprint-Strafen: Gehirn verlernte das Gehen", "Gehen weiter üben, Strafen-Rampe"],
  ["Geometrie", "Arme erreichten den Boden nicht, Handgelenk an falscher Stelle", "Hand neu modelliert"],
  ["Haltung geändert", "neue Arm-Ruhehaltung: Gehirn erkannte sich nicht, stand still", "Beobachtungs-Bezug verschoben"],
  ["Laptop schläft", "Sitzungsende / Schlafmodus stoppten Training (≈ 9 h)", "Wach-Halter, Netzteil"],
  ["Planke", "stabile Liege-Haltung brachte im Stehen die meisten Punkte", "Planke = Sturz, Haltungs-Kosten"],
  ["Klobiger Gang", "Anlaufhilfe belohnte hastige kurze Schritte", "Glätt-Stufe, Hilfe nur langsam & sanft"]], { y: 1.35, colW: [2.6, 6.0, 3.5] }, 13, 0.5);

// 37 Lehren
s = content("Die wichtigsten Lehren", "Erfahrungen");
const lessons = ["Immer prüfen, was Stillstehen unter jedem Befehl verdient", "Belohnungsstrafen multiplizieren, nicht abziehen", "Rohe Gehirn-Befehle beobachten, nicht nur Gelenkwinkel",
  "Erst der Körper, dann das Lernen: Masse, Symmetrie, Standhaltung", "Kleine Schritte auf einem laufenden Gehirn schlagen große Sprünge", "Jeden Checkpoint behalten und gegen messbare Ziele testen",
  "Gelernte Gewohnheiten überleben Regeländerungen", "Live zuschauen findet Fehler, die Zahlen verschweigen",
  "Regeln nur für den Gang, den sie beschreiben", "Erst die Geometrie prüfen: kann das Glied den Boden erreichen?"];
lessons.forEach((t, i) => {
  const col = i % 2, row = Math.floor(i / 2), x = M + col * 6.2, y = 1.4 + row * 1.03;
  dot(s, x, y + 0.08, String(i + 1), col ? C.accent2 : C.accent1, 0.55);
  T(s, t, { x: x + 0.8, y: y + 0.05, w: 5.2, h: 0.75, fontSize: 16, color: C.text2, valign: "middle" });
});

// 38 Ausblick
s = pres.addSlide({ masterName: "ABSCHNITT", sectionTitle: "Erfahrungen" });
s.addText("Ausblick", { placeholder: "title" });
s.addText("Die nächsten Schritte", { placeholder: "body" });
["Showcase v6: volle 5 Minuten rendern", "Myrranox-Texturen verknüpfen", "Tempo-Unterschied DC / Myrranox (Balance)",
 "Raubtier-Gehirne mit den neuen Läufern nachtrainieren", "Showcase mit 2 + 2 Kreaturen", "Präsentation & Paper"].forEach((t, i) => {
  const x = M + (i % 2) * 6.2, y = 4.75 + Math.floor(i / 2) * 0.72;
  dot(s, x, y, String(i + 1), i % 2 ? C.accent2 : C.accent1, 0.45);
  T(s, t, { x: x + 0.65, y: y + 0.03, w: 5.4, h: 0.45, fontSize: 16, color: C.background1, valign: "middle" });
});

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
