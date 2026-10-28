// Endemic Creatures - Kurzpräsentation 10 Minuten (Deutsch). Build: cd docs/presentation/src && node build_10min.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("./apply_theme.js");

const REPO = path.join(__dirname, "..", "..", "..");
const FIG = (p) => path.join(REPO, "docs/figures", p);
const LOCAL = (p) => path.join(__dirname, p);
const OUT = path.join(REPO, "docs/presentation/Endemic_Creatures_RL_10min.pptx");

const THEME = {
  name: "Endemic Creatures",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C2321", lt1: "FFFFFF", dk2: "1F3A35", lt2: "EDF2EF",
    accent1: "D98E04", accent2: "2E7D6B", accent3: "B23A48", accent4: "5B6C8F",
    accent5: "8FA89B", accent6: "C9B27C", hlink: "2E7D6B", folHlink: "5B6C8F",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Endemic Creatures - Laufen und Jagen lernen mit Reinforcement Learning";
pres.author = "Thomas";
pres.subject = "Kurzpräsentation (10 Minuten)";
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "TITEL", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 4.3, w: 12.1, h: 1.3, fontFace: "Cambria", fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 5.7, w: 12.1, h: 1.1, fontSize: 18, color: C.accent6, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "INHALT", background: { color: C.background1 },
  margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: 12.1, h: 0.85, fontFace: "Cambria", fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left" }, text: "" } },
    { text: { text: "Endemic Creatures · Reinforcement Learning", options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: "8FA89B" } } },
  ],
  slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: "8FA89B" },
});

// ------------------------------------------------------------------ helpers
const content = (title, section) => {
  const s = pres.addSlide({ masterName: "INHALT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const T = (s, text, o) => s.addText(text, { isTextBox: true, fontFace: "Calibri", fontSize: 15, color: C.text1, valign: "top", margin: 0, ...o });
const card = (s, x, y, w, h, name, fill = C.background2) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
const dot = (s, x, y, label, fill, d = 0.55) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
  s.addText(label, { isTextBox: true, x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 15, bold: true, color: C.background1, margin: 0 });
};
const bullets = (items, size = 15) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6, fontSize: size } }));
const table = (s, rows, o, fs = 15, rowH = 0.5) => s.addTable(rows.map((r, i) => r.map((c) => ({ text: String(c),
  options: i === 0 ? { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: fs, valign: "middle" }
                   : { fontSize: fs, color: HEX.dk1, fill: { color: i % 2 ? "FFFFFF" : HEX.lt2 }, valign: "middle" } }))),
  { x: M, w: W - 2 * M, rowH, border: { type: "solid", pt: 0.5, color: "D5DDD8" }, fontFace: "Calibri", margin: [0.05, 0.12, 0.05, 0.12], ...o });

// ================================================================== 1 Titel
pres.addSection({ title: "Einstieg" });
let s = pres.addSlide({ masterName: "TITEL", sectionTitle: "Einstieg" });
s.addImage({ path: FIG("showcase_v6_still.jpg"), x: 0, y: 0, w: W, h: W * 544 / 1928, objectName: "hero" });
s.addText("Endemic Creatures: Laufen und Jagen lernen mit Reinforcement Learning", { placeholder: "title" });
s.addText("Zwei erfundene Raubtiere lernen in einer Physik-Simulation selbst zu gehen und zu jagen · Thomas · Oktober 2026", { placeholder: "body" });
s.addNotes("ca. 30 s. Kurz vorstellen: Projekt, zwei Kreaturen, alles gelernt statt animiert. Bild: der aktuelle Showcase mit beiden finalen Läufern.");

// ================================================================== 2 Idee
s = content("Die Idee", "Einstieg");
[["Myrranox", "Ameisen-artig, 6 Beine, 8 kg", C.accent2, "myr_still.jpg"], ["DC", "Panther-artig mit Zusatzarmen, 60 kg", C.accent4, "dc_still.jpg"]].forEach(([name, sub, col, img], i) => {
  const x = M + i * 3.15;
  card(s, x, 1.45, 2.95, 3.05, `creature${i}`);
  s.addImage({ path: LOCAL(img), x: x + 0.15, y: 1.6, w: 2.65, h: 1.3, sizing: { type: "cover", w: 2.65, h: 1.3 }, objectName: `img${i}` });
  T(s, name, { x: x + 0.2, y: 3.05, w: 2.6, h: 0.45, fontSize: 20, bold: true, color: col, fontFace: "Cambria" });
  T(s, sub, { x: x + 0.2, y: 3.55, w: 2.6, h: 0.8, fontSize: 14 });
});
card(s, M + 6.4, 1.45, 5.7, 5.15, "idea_text");
T(s, "Ziel", { x: M + 6.7, y: 1.65, w: 5.2, h: 0.45, fontSize: 20, bold: true, color: C.text2, fontFace: "Cambria" });
s.addText(bullets(["Zwei Raubtiere leben in einem Habitat von 100 × 100 m mit Revieren, Beute und Rastplätzen",
  "Sie lernen selbst: wohin sie gehen (jagen, rasten, Revier) und wie sie ihre Beine bewegen",
  "Keine Animation: jede Bewegung entsteht durch Belohnung und Strafe in einer Physik-Simulation",
  "Bewegung wie das Vorbild: Insekt im Dreifuß-Gang, Raubkatze / Wolf im Viertakt",
  "Ergebnis als Video in Blender"], 15), { isTextBox: true, x: M + 6.7, y: 2.2, w: 5.2, h: 4.3, valign: "top", margin: 0 });
card(s, M, 4.75, 6.1, 1.85, "question");
T(s, "Kernfrage", { x: M + 0.25, y: 4.9, w: 5.6, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
T(s, "Kann ein Computer ohne Vorgaben lernen, sich wie ein echtes Tier zu bewegen und zu verhalten?", { x: M + 0.25, y: 5.35, w: 5.6, h: 1.1, fontSize: 16, italic: true, color: C.text2 });
s.addNotes("ca. 1 min. Idee: lebendige Kreaturen, die nichts vorprogrammiert bekommen. Nur Regeln, wofür es Punkte gibt.");

// ================================================================== 3 Vorgehen
pres.addSection({ title: "Vorgehen" });
s = content("Wie bin ich vorgegangen?", "Vorgehen");
const steps = [["1", "Plan & Welt", "Projektplan, Habitat, Regeln, 3D-Modelle in Blender"],
  ["2", "Jagen in 2D", "Schnelle 2D-Simulation: Raubtier-Gehirne lernen jagen und rasten"],
  ["3", "Laufen in 3D", "MuJoCo-Physik: Laufgehirne lernen die Beine zu bewegen"],
  ["4", "Körper wie das Modell", "Physik-Körper direkt aus dem 3D-Netz (bis 218 Teile)"],
  ["5", "Natürlicher Stil", "Regeln nach Vorbild (Insekt, Wolf), messen, nachschärfen"],
  ["6", "Video", "Beide Gehirne zusammen im Habitat, in Blender gerendert"]];
steps.forEach(([nr, t, d], i) => {
  const col = i % 3, row = Math.floor(i / 3), x = M + col * 4.1, y = 1.5 + row * 2.15;
  card(s, x, y, 3.85, 1.9, `step${i}`);
  dot(s, x + 0.25, y + 0.25, nr, i < 3 ? C.accent1 : C.accent2);
  T(s, t, { x: x + 1.0, y: y + 0.3, w: 2.7, h: 0.45, fontSize: 17, bold: true, color: C.text2 });
  T(s, d, { x: x + 0.25, y: y + 0.95, w: 3.4, h: 0.85, fontSize: 14 });
});
T(s, "Arbeitsweise: immer eine Änderung pro Training, danach messen (Tempo, Takt, Stürze) und live zuschauen · 12 Versionen des Laufgehirns in 5 Wochen", { x: M, y: 6.0, w: 12.1, h: 0.6, fontSize: 14, italic: true, color: C.text2 });
s.addNotes("ca. 1 min 15 s. Schritt für Schritt: erst die Welt, dann Entscheidungen in 2D (schnell), dann echte Physik. Iterativ: eine Änderung, messen, zuschauen.");

// ================================================================== 4 Tools
s = content("Welche Mittel und Tools habe ich verwendet?", "Vorgehen");
const tools = [["MuJoCo", "Physik-Simulation: Kräfte, Kontakte, Gleichgewicht", C.accent4],
  ["Gymnasium", "Lernumgebungen: Beobachtung, Aktion, Belohnung", C.accent2],
  ["Stable-Baselines3 + PyTorch", "Lernalgorithmus PPO, neuronale Netze", C.accent1],
  ["Blender 5.2", "3D-Modelle, Rig, Animation, Rendering", C.accent3],
  ["CoACD", "zerlegt das 3D-Netz in Physik-Teile", C.accent4],
  ["Python, Git / GitHub", "Code, Versionen, alle Gehirne und Videos gesichert", C.accent2],
  ["Claude Code", "KI-Assistent: Code, Analysen, Doku", C.accent1],
  ["Laptop", "8 CPU-Kerne, keine Grafikkarte", C.accent3]];
tools.forEach(([t, d, col], i) => {
  const c2 = i % 2, r = Math.floor(i / 2), x = M + c2 * 6.2, y = 1.45 + r * 1.3;
  card(s, x, y, 5.9, 1.1, `tool${i}`);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.3, w: 0.5, h: 0.5, fill: { color: col }, line: { color: col } });
  T(s, t, { x: x + 0.95, y: y + 0.15, w: 4.8, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  T(s, d, { x: x + 0.95, y: y + 0.58, w: 4.8, h: 0.4, fontSize: 14 });
});
s.addNotes("ca. 50 s. Alles Open Source ausser dem Assistenten. Rechnen nur auf dem Laptop: ein Grund, warum Training und Rendern lange dauerten.");

// ================================================================== 5 Algorithmus
pres.addSection({ title: "Lernen" });
s = content("Welchen Lernalgorithmus habe ich verwendet?", "Lernen");
card(s, M, 1.45, 5.6, 5.15, "ppo");
T(s, "PPO – Proximal Policy Optimization", { x: M + 0.25, y: 1.6, w: 5.1, h: 0.5, fontSize: 18, bold: true, color: C.accent1 });
s.addText(bullets(["Reinforcement Learning: Ausprobieren, Belohnung bekommen, besser werden",
  "Actor-Critic: ein Netz wählt die Aktion, ein zweites schätzt den Wert der Lage",
  "„Proximal“: pro Update nur kleine Änderungen – stabiles Lernen",
  "8 Simulationen parallel, Millionen Versuche (≈ 185 Mio. Schritte insgesamt)",
  "Extra: Spiegel-Symmetrie links/rechts, Lehrpläne (erst leicht, dann schwer)"], 14), { isTextBox: true, x: M + 0.25, y: 2.2, w: 5.15, h: 4.3, valign: "top", margin: 0 });
const arch = [["Raubtier-Gehirn", "10 Hz", "wohin? jagen, rasten\n→ Tempo + Drehung", C.accent1],
  ["Laufgehirn", "50 Hz", "wie? Gelenkziele für\n18 (Myr) / 29 (DC) Motoren", C.accent2],
  ["MuJoCo-Physik", "200 Hz", "Kräfte, Kontakte,\nGleichgewicht", C.accent4]];
arch.forEach(([t, hz, d, col], i) => {
  const y = 1.45 + i * 1.75;
  card(s, 6.55, y, 6.15, 1.45, `arch${i}`);
  dot(s, 6.8, y + 0.45, String(i + 1), col);
  T(s, `${t} · ${hz}`, { x: 7.6, y: y + 0.2, w: 4.9, h: 0.45, fontSize: 17, bold: true, color: C.text2 });
  T(s, d, { x: 7.6, y: y + 0.65, w: 4.9, h: 0.75, fontSize: 14 });
  if (i < 2) s.addShape(pres.shapes.DOWN_ARROW, { x: 9.45, y: y + 1.47, w: 0.3, h: 0.26, fill: { color: "8FA89B" }, line: { color: "8FA89B" } });
});
s.addNotes("ca. 1 min 15 s. PPO einfach erklären. Zwei Gehirne übereinander: das obere entscheidet, das untere setzt es in Beinbewegung um. Belohnung beim Laufen: Tempo treffen × wie tierähnlich × wie sauber (Ruck, Kippen, Rutschen).");

// ================================================================== 6 Ergebnis
s = content("Was ist herausgekommen? Überblick und DC", "Lernen");
const stats = [["12", "Versionen des\nLaufgehirns"], ["≈ 50", "trainierte\nGehirne"], ["218", "Physik-Teile im\nDC-Körper"]];
stats.forEach(([big, label], i) => {
  const x = M + i * 2.1;
  card(s, x, 1.45, 1.95, 1.9, `stat${i}`);
  T(s, big, { x: x + 0.15, y: 1.55, w: 1.7, h: 0.8, fontFace: "Cambria", fontSize: 34, bold: true, color: i % 2 ? C.accent2 : C.accent1, align: "center" });
  T(s, label, { x: x + 0.1, y: 2.45, w: 1.75, h: 0.8, fontSize: 13, color: C.text2, align: "center" });
});
s.addImage({ path: FIG("dc_physics_body_v11_vs_v12.jpg"), x: M, y: 3.55, w: 6.1, h: 3.1, sizing: { type: "cover", w: 6.1, h: 3.1 }, objectName: "body" });
table(s, [["", "Myrranox v10", "DC v12z"], ["Gehen", "0,52 m/s, Dreifuß", "0,59 m/s, flüssig"], ["Vollgas", "1,08 m/s", "bis 4,1 m/s Sprint"], ["Besonderes", "rückwärts, dreht am Ort", "bremst ohne Sturz"], ["Jagen (2D)", "4,0 Beute / 10 min", "4,6 Beute / 10 min"]],
  { x: 7.0, w: 5.7, y: 1.45, colW: [1.6, 2.05, 2.05] }, 14, 0.55);
T(s, "Beide Läufer final, beide jagen im Habitat; ein 30-s-Showcase-Video zeigt sie zusammen.", { x: 7.0, y: 4.45, w: 5.7, h: 0.9, fontSize: 14, italic: true, color: C.text2 });
s.addNotes("ca. 40 s. Kurz die Zahlen; Bild: alter DC-Physik-Körper (Blöcke) gegen den neuen aus 218 Teilen.");

// ================================================================== 6b Ergebnis Myrranox
s = content("Was ist herausgekommen? Myrranox", "Lernen");
s.addImage({ path: LOCAL("myrranox_physics_body_mesh_deck.jpg"), x: M, y: 1.45, w: 5.6, h: 5.6 * 1240 / 1650, objectName: "myr_body" });
T(s, "Physik-Körper aus dem echten Modell: 21 Hüllen, Krallen als Füße", { x: M, y: 5.75, w: 5.6, h: 0.4, fontSize: 13, italic: true, color: C.text2 });
[["0,52 m/s", "Gehen\n(Befehl 0,6)"], ["1,08 m/s", "Vollgas\n(Befehl 1,2)"], ["56 °/s", "Drehen auf\nder Stelle"], ["−0,26 m/s", "rückwärts –\nerstmals gelernt"]].forEach(([big, label], i) => {
  const x = 6.55 + (i % 4) * 1.55;
  card(s, x, 1.45, 1.42, 1.75, `myr_stat${i}`);
  T(s, big, { x: x + 0.05, y: 1.6, w: 1.32, h: 0.55, fontFace: "Cambria", fontSize: 19, bold: true, color: i % 2 ? C.accent1 : C.accent2, align: "center" });
  T(s, label, { x: x + 0.05, y: 2.25, w: 1.32, h: 0.85, fontSize: 12, color: C.text2, align: "center" });
});
card(s, 6.55, 3.4, 6.15, 2.65, "myr_path");
T(s, "Der Weg zum Insekten-Gang", { x: 6.8, y: 3.55, w: 5.7, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
s.addText(bullets(["v1: läuft, aber nur mit den Mittelbeinen", "v2: Dreifuß-Takt (3 Beine am Boden), 5× weniger Energie", "v5: Knie endlich frei (waren seit v2 blockiert)",
  "v9: „Schubkarre“ – der Hinterleib war 54 % des Gewichts", "v10: Hinterleib leichter (23 %) + Dreifuß-Schrittgeber → final"], 14), { isTextBox: true, x: 6.8, y: 4.05, w: 5.75, h: 1.95, valign: "top", margin: 0 });
T(s, "Dreifuß-Takt 87 % der Zeit · Knie 60° · Hinterleib nie am Boden · keine Stürze · offen: Vorderbeine nur 21 % am Boden", { x: M, y: 6.35, w: 12.1, h: 0.4, fontSize: 13, italic: true, color: C.accent3 });
s.addNotes("ca. 45 s. Der Myrranox war schwieriger als gedacht: erst als das Gewicht stimmte (leichterer Hinterleib) und ein fester Dreifuß-Rhythmus vorgegeben wurde, lief er wie ein Insekt. Rückwärts und Drehen auf der Stelle hat er erst mit v10 gelernt.");

// ================================================================== 7 Gelernt
pres.addSection({ title: "Rückblick" });
s = content("Was habe ich gelernt?", "Rückblick");
const learned = [["Belohnung ist alles", "Das Gehirn findet jedes Schlupfloch: Stillstehen, Handstand, „Planke“ – wenn es Punkte bringt"],
  ["Erst der Körper, dann das Lernen", "Falsche Masse, Geometrie oder Gelenke kann keine Regel ausgleichen"],
  ["Multiplizieren statt abziehen", "Strafen verkleinern die Punkte – Stehen darf nie billiger sein als Laufen"],
  ["Messen und zuschauen", "Zahlen zeigen Trends, live zuschauen zeigt, was wirklich passiert"],
  ["Kleine Schritte", "Eine Änderung pro Training; jeden Checkpoint behalten"],
  ["Regeln gezielt einsetzen", "Eine Geh-Regel im Sprint hat das Sprinten verhindert"]];
learned.forEach(([t, d], i) => {
  const c2 = i % 2, r = Math.floor(i / 2), x = M + c2 * 6.2, y = 1.45 + r * 1.72;
  card(s, x, y, 5.9, 1.5, `learn${i}`);
  dot(s, x + 0.25, y + 0.45, String(i + 1), c2 ? C.accent2 : C.accent1);
  T(s, t, { x: x + 1.0, y: y + 0.18, w: 4.7, h: 0.45, fontSize: 16, bold: true, color: C.text2 });
  T(s, d, { x: x + 1.0, y: y + 0.65, w: 4.7, h: 0.8, fontSize: 14 });
});
s.addNotes("ca. 1 min 15 s. Wichtigste Erkenntnis: RL optimiert genau das, was man belohnt - nicht das, was man meint. Beispiele nennen.");

// ================================================================== 8 Schlecht gelaufen
s = content("Was ist schlecht gelaufen?", "Rückblick");
table(s, [["Problem", "Was passierte"],
  ["Stillstand-Fallen", "Mehrmals lernten die Gehirne, einfach stehen zu bleiben (Strafen zu stark)"],
  ["Schlupflöcher", "DC: Handstand auf Vorderbeinen, später „Planke“ mit gespreizten Beinen"],
  ["Regelfehler", "Fehler in meinen Regeln haben ganze Trainings verdorben (z. B. Beine nach innen gezogen)"],
  ["Arme & Hand", "Arme zu kurz, Handgelenk an falscher Stelle – Arme im Sprint am Ende verworfen"],
  ["Rechner", "Laptop schlief ein oder Sitzung endete: ≈ 9 h Training verloren; Rendern 4–6 h pro Video"],
  ["Jagen", "Raubzüge und Revierverteidigung nie gelernt; Spiel-Idee gebaut und verworfen"]], { y: 1.45, colW: [2.9, 9.2] }, 17, 0.75);
s.addNotes("ca. 1 min 15 s. Ehrlich: viele Runden waren Fehlversuche. Jede hat aber eine Lehre gebracht.");

// ================================================================== 9 Anders machen
s = content("Was würde ich anders machen?", "Rückblick");
const diff = [["Körper zuerst prüfen", "Masse, Geometrie, Reichweite der Glieder vor dem ersten Training messen"],
  ["Belohnung vorher testen", "Mit einfachen Skript-Figuren prüfen, ob Stehen oder Schlupflöcher sich lohnen"],
  ["Vorbild früher nutzen", "Eine Referenz-Bewegung (z. B. der Wolf) von Anfang an als Ziel"],
  ["Ziele messbar festlegen", "Tempo, Takt, Haltung als feste Tests vor jedem Training"],
  ["Stärkere Hardware", "Rechner mit Grafikkarte oder Cloud: Rendern 10–20× schneller, nichts schläft ein"],
  ["Weniger auf einmal", "Erst ein Tier fertig, dann das zweite; weniger Ideen parallel"]];
diff.forEach(([t, d], i) => {
  const c2 = i % 3, r = Math.floor(i / 3), x = M + c2 * 4.1, y = 1.5 + r * 2.5;
  card(s, x, y, 3.85, 2.2, `diff${i}`);
  dot(s, x + 0.25, y + 0.25, "→", r ? C.accent2 : C.accent1);
  T(s, t, { x: x + 1.0, y: y + 0.3, w: 2.7, h: 0.5, fontSize: 16, bold: true, color: C.text2 });
  T(s, d, { x: x + 0.25, y: y + 0.95, w: 3.4, h: 1.15, fontSize: 14 });
});
s.addNotes("ca. 1 min. Die drei wichtigsten: Körper zuerst, Belohnung vorher testen, bessere Hardware.");

// ================================================================== 10 Danke
s = pres.addSlide({ masterName: "TITEL", sectionTitle: "Rückblick" });
s.addImage({ path: FIG("showcase_v6_still.jpg"), x: 0, y: 0, w: W, h: W * 544 / 1928, objectName: "outro" });
s.addText("Danke! Fragen?", { placeholder: "title" });
s.addText("Video: docs/videos/showcase_v6_30s.mp4 · Code, Gehirne und Doku: GitHub, Branch rl-locomotion-setup", { placeholder: "body" });
s.addNotes("ca. 30 s plus Fragen. Optional hier das 30-s-Video zeigen.");

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
});
