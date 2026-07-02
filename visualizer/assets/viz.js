/* Hex Agent Visualizer — renders the trace produced by record_game.py.
   Zero dependencies: vanilla JS + SVG. TRACE is injected as window.TRACE. */
(function () {
"use strict";
const T = window.TRACE;
const app = document.getElementById("app");
if (!T || !T.moves || !T.moves.length) {
  app.innerHTML = "<p style='padding:24px'>No trace data found.</p>";
  return;
}
const N = T.meta.board_size;
const MOVES = T.moves;
const V7SIDE = T.meta.v7_side;
const P0 = "#e0574a", P1 = "#4a90e0";
const pcolor = p => (p === 0 ? P0 : P1);
const SVGNS = "http://www.w3.org/2000/svg";

// ---------- small helpers ----------
function el(tag, attrs) { const e = document.createElementNS(SVGNS, tag); if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }
function h(tag, attrs, txt) { const e = document.createElement(tag); if (attrs) for (const k in attrs) { if (k === "class") e.className = attrs[k]; else if (k === "html") e.innerHTML = attrs[k]; else e.setAttribute(k, attrs[k]); } if (txt != null) e.textContent = txt; return e; }
const $ = id => document.getElementById(id);
function rc(idx) { return [Math.floor(idx / N), idx % N]; }
function coord(idx) { if (idx == null) return "—"; const [r, c] = rc(idx); return r + "," + c; }
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function hex2rgb(x){x=x.replace('#','');return [parseInt(x.slice(0,2),16),parseInt(x.slice(2,4),16),parseInt(x.slice(4,6),16)];}
function mix(c1,c2,t){const a=hex2rgb(c1),b=hex2rgb(c2);const L=(i)=>Math.round(a[i]+(b[i]-a[i])*t);return `rgb(${L(0)},${L(1)},${L(2)})`;}

// ---------- geometry ----------
const SIZE = 22, SQ3 = Math.sqrt(3);
function cellXY(idx) { const [r, c] = rc(idx); return { x: SIZE * SQ3 * (c + r / 2), y: SIZE * 1.5 * r }; }
function hexPts(cx, cy) { let p = []; for (let i = 0; i < 6; i++) { const a = (-90 + 60 * i) * Math.PI / 180; p.push((cx + SIZE * Math.cos(a)).toFixed(2) + "," + (cy + SIZE * Math.sin(a)).toFixed(2)); } return p.join(" "); }
let minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
for (let i = 0; i < N * N; i++) { const { x, y } = cellXY(i); minx = Math.min(minx, x); maxx = Math.max(maxx, x); miny = Math.min(miny, y); maxy = Math.max(maxy, y); }
const PAD = SIZE * 1.9;
const cx0 = (minx + maxx) / 2, cy0 = (miny + maxy) / 2;

// ---------- state ----------
const state = {
  mi: 0, playing: false, timer: null, tab: "move",
  resP: V7SIDE,
  ov: { heat: false, current: true, dijkstra: false, bridges: false, candidates: false, tactics: true, labels: false, lastmove: true },
  treeNull: true,
};

// ---------- build static board ----------
const board = $("board");
board.setAttribute("viewBox", `${(minx - PAD).toFixed(1)} ${(miny - PAD).toFixed(1)} ${(maxx - minx + 2 * PAD).toFixed(1)} ${(maxy - miny + 2 * PAD).toFixed(1)}`);
const gEdges = el("g"), gCells = el("g"), gHeat = el("g"), gCur = el("g"), gDij = el("g"), gBri = el("g"), gCand = el("g"), gTac = el("g"), gMark = el("g"), gLabels = el("g");
[gEdges, gHeat, gCells, gCur, gDij, gBri, gCand, gTac, gLabels, gMark].forEach(g => board.appendChild(g));

// colored side indicators (P0 left/right red, P1 top/bottom blue)
function sideLine(aIdx, bIdx, color) {
  const a = cellXY(aIdx), b = cellXY(bIdx);
  const out = (p) => { const dx = p.x - cx0, dy = p.y - cy0, L = Math.hypot(dx, dy) || 1; return { x: p.x + dx / L * SIZE * 0.85, y: p.y + dy / L * SIZE * 0.85 }; };
  const A = out(a), B = out(b);
  gEdges.appendChild(el("line", { x1: A.x, y1: A.y, x2: B.x, y2: B.y, stroke: color, "stroke-width": 6, "stroke-linecap": "round", opacity: .65 }));
}
sideLine(0, N - 1, P1);                       // top
sideLine((N - 1) * N, N * N - 1, P1);         // bottom
sideLine(0, (N - 1) * N, P0);                 // left
sideLine(N - 1, N * N - 1, P0);               // right

const cellEls = [];
for (let i = 0; i < N * N; i++) {
  const { x, y } = cellXY(i);
  const poly = el("polygon", { points: hexPts(x, y), class: "hex empty" });
  poly.dataset.i = i;
  poly.addEventListener("mousemove", (ev) => showTip(ev, i));
  poly.addEventListener("mouseleave", hideTip);
  gCells.appendChild(poly);
  cellEls.push(poly);
  const lbl = el("text", { x: x, y: y, class: "hexlabel" }); lbl.textContent = rc(i)[0] + "," + rc(i)[1];
  gLabels.appendChild(lbl);
}

// ---------- tooltip ----------
const tip = $("tooltip");
function showTip(ev, i) {
  const rec = MOVES[state.mi];
  const [r, c] = rc(i);
  const st = rec.board[i];
  let html = `<b>${r},${c}</b> — ${st === -1 ? "empty" : ("P" + st)}`;
  const vv = rec.resistance[String(state.resP)].v[i];
  html += `<br>potential(P${state.resP}) = ${vv.toFixed(3)}`;
  const cand = rec.ranking.find(x => x.cell === i);
  if (cand) { html += `<br>candidate #${rec.ranking.indexOf(cand) + 1}, total ${cand.total}`; }
  tip.innerHTML = html; tip.hidden = false;
  const wrap = $("boardwrap").getBoundingClientRect();
  tip.style.left = (ev.clientX - wrap.left + 12) + "px";
  tip.style.top = (ev.clientY - wrap.top + 12) + "px";
}
function hideTip() { tip.hidden = true; }

// ---------- overlay drawing ----------
function clear(g) { while (g.firstChild) g.removeChild(g.firstChild); }
function line(g, i, j, attrs) { const a = cellXY(i), b = cellXY(j); g.appendChild(el("line", Object.assign({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, "stroke-linecap": "round" }, attrs))); }
function dot(g, i, attrs) { const a = cellXY(i); g.appendChild(el("circle", Object.assign({ cx: a.x, cy: a.y, r: 4 }, attrs))); }

function drawHeat(rec) {
  clear(gHeat); if (!state.ov.heat) return;
  const v = rec.resistance[String(state.resP)].v, col = pcolor(state.resP);
  for (let i = 0; i < N * N; i++) {
    const t = clamp(v[i], 0, 1); if (t < 0.03) continue;
    const { x, y } = cellXY(i);
    gHeat.appendChild(el("polygon", { points: hexPts(x, y), fill: mix("#0c1a33", col, t), opacity: 0.62, stroke: "none" }));
  }
}
function drawCurrent(rec) {
  clear(gCur); if (!state.ov.current) return;
  const res = rec.resistance[String(state.resP)], col = pcolor(state.resP);
  let mx = 1e-9; for (const e of res.edges) mx = Math.max(mx, Math.abs(e[2]));
  for (const [i, j, cur] of res.edges) {
    const t = Math.abs(cur) / mx; if (t < 0.05) continue;
    line(gCur, i, j, { stroke: col, "stroke-width": (0.6 + 5.2 * t).toFixed(2), opacity: (0.18 + 0.62 * t).toFixed(2) });
  }
  for (const [i, j] of res.bridges) line(gCur, i, j, { stroke: col, "stroke-width": 2, opacity: .8, "stroke-dasharray": "3,3" });
}
function drawDijkstra(rec) {
  clear(gDij); if (!state.ov.dijkstra) return;
  for (const p of [0, 1]) {
    const path = rec.dijkstra[String(p)].path; if (!path.length) continue;
    for (const i of path) dot(gDij, i, { r: 5, fill: "none", stroke: pcolor(p), "stroke-width": 2, opacity: .9 });
  }
}
function drawBridges(rec) {
  clear(gBri); if (!state.ov.bridges) return;
  for (const p of [0, 1]) {
    for (const b of rec.tactics.bridges[String(p)]) {
      line(gBri, b.a, b.b, { stroke: pcolor(p), "stroke-width": 2.5, opacity: .85 });
      for (const cc of b.c) dot(gBri, cc, { r: 2.6, fill: pcolor(p), opacity: .8, stroke: "none" });
    }
  }
}
function drawCandidates(rec) {
  clear(gCand); if (!state.ov.candidates) return;
  const K = Math.min(12, rec.ranking.length);
  for (let k = 0; k < K; k++) {
    const cell = rec.ranking[k].cell, { x, y } = cellXY(cell);
    gCand.appendChild(el("polygon", { points: hexPts(x, y), fill: "#5fd08a", opacity: (0.5 * (K - k) / K + 0.08).toFixed(2), stroke: "none" }));
    const tx = el("text", { x: x, y: y + 3, "text-anchor": "middle", "font-size": 9, fill: "#08130c", "font-weight": "700" }); tx.textContent = (k + 1);
    gCand.appendChild(tx);
  }
}
function drawTactics(rec) {
  clear(gTac); if (!state.ov.tactics) return;
  for (const i of rec.tactics.immediate_win) dot(gTac, i, { r: 6, fill: "none", stroke: "#5fd08a", "stroke-width": 3 });
  for (const i of rec.tactics.immediate_block) dot(gTac, i, { r: 6, fill: "none", stroke: "#e0574a", "stroke-width": 3 });
  for (const i of rec.tactics.save_bridge) dot(gTac, i, { r: 8, fill: "none", stroke: "#f6c85f", "stroke-width": 2, "stroke-dasharray": "2,2" });
}
function drawMark(rec) {
  clear(gMark);
  if (state.ov.lastmove && rec.chosen != null) { const { x, y } = cellXY(rec.chosen); gMark.appendChild(el("polygon", { points: hexPts(x, y), class: "chosen" })); }
}
gLabels.style.display = "none";

// ---------- render ----------
function render() {
  const rec = MOVES[state.mi];
  for (let i = 0; i < N * N; i++) { const s = rec.board[i]; cellEls[i].setAttribute("class", "hex " + (s === -1 ? "empty" : "p" + s)); }
  gLabels.style.display = state.ov.labels ? "" : "none";
  drawHeat(rec); drawCurrent(rec); drawDijkstra(rec); drawBridges(rec); drawCandidates(rec); drawTactics(rec); drawMark(rec);
  renderMovePanel(rec); renderRanking(rec); renderTree(rec); markCharts();
  $("range").value = state.mi;
  const [r, c] = rc(rec.chosen);
  $("movelabel").textContent = `move ${rec.move_number || state.mi + 1}/${MOVES.length} · ply ${rec.ply != null ? rec.ply : "?"} · P${rec.player} plays ${r},${c}`;
}

// ---------- panels ----------
function fmtScore(nd) {
  if (nd.term === "win") return "WIN";
  if (nd.term === "loss") return "LOSS";
  if (nd.leaf) return "eval " + nd.s;
  return String(nd.s);
}
function fmtBound(x) { if (x == null) return "·"; if (x <= -1e8) return "-∞"; if (x >= 1e8) return "+∞"; return String(x); }

function renderMovePanel(rec) {
  const p = $("panel-move"); p.innerHTML = "";
  p.appendChild(h("h3", null, "Decision"));
  const badge = h("span", { class: "badge " + rec.stage }, rec.stage.replace(/_/g, " "));
  const bd = h("div"); bd.appendChild(document.createTextNode("stage: ")); bd.appendChild(badge); p.appendChild(bd);
  const kv = h("div", { class: "kv" });
  const add = (k, v) => { kv.appendChild(h("div", null, k)); kv.appendChild(h("div", null, v)); };
  add("move #", (rec.move_number || "?") + " (ply " + (rec.ply != null ? rec.ply : "?") + ")");
  add("player", "P" + rec.player + (rec.player === V7SIDE ? " (v7)" : ""));
  add("chosen", coord(rec.chosen));
  add("eval score", rec.eval_score + "  = 1000·ln(R_opp/R_me)");
  add("search depth", rec.tree_depth);
  add("nodes", rec.nodes);
  add("TT entries", rec.tt_size);
  add("time", rec.time + " s");
  const R0 = rec.resistance["0"].R, R1 = rec.resistance["1"].R;
  add("R (P0 L↔R)", R0 == null ? "∞" : R0);
  add("R (P1 T↔B)", R1 == null ? "∞" : R1);
  p.appendChild(kv);
  // ID progression
  if (rec.id && rec.id.length) {
    p.appendChild(h("h3", null, "Iterative deepening"));
    const t = h("table");
    t.innerHTML = "<tr><th>depth</th><th>best</th><th>score</th><th>nodes</th><th>time</th></tr>";
    for (const r of rec.id) {
      const tr = h("tr"); tr.innerHTML = `<td>${r.depth}</td><td>${coord(r.move)}</td><td>${r.score}</td><td>${r.nodes}</td><td>${r.time}s</td>`;
      t.appendChild(tr);
    }
    p.appendChild(t);
  }
}

const COMPCOLORS = { race: "#4a90e0", disrupt: "#e0574a", static: "#8b96b5", save: "#f6c85f", on_my_path: "#4a90e0", on_opp_path: "#e0574a", near_my_path: "#3a6ea8", near_opp_path: "#a8483f", edge: "#7a5", danger: "#c0392b" };
function renderRanking(rec) {
  const p = $("panel-ranking"); p.innerHTML = "";
  p.appendChild(h("h3", null, "Root candidate ranking (why this move)"));
  const K = Math.min(rec.ranking.length, 18);
  for (let k = 0; k < K; k++) {
    const it = rec.ranking[k];
    const row = h("div", { class: "rank-row" + (it.cell === rec.chosen ? " chosen" : "") });
    const head = h("div", { class: "rank-head" });
    head.appendChild(h("span", { class: "coord" }, "#" + (k + 1) + "  " + coord(it.cell) + (it.cell === rec.chosen ? "  ✓" : "")));
    head.appendChild(h("span", { class: "total" }, it.total));
    row.appendChild(head);
    // comp bar (magnitudes)
    const comps = Object.entries(it.comp).filter(([, v]) => Math.abs(v) > 0.001);
    const tot = comps.reduce((s, [, v]) => s + Math.abs(v), 0) || 1;
    const bar = h("div", { class: "compbar" });
    for (const [name, v] of comps) { const seg = h("span"); seg.style.width = (100 * Math.abs(v) / tot) + "%"; seg.style.background = COMPCOLORS[name] || "#666"; seg.title = name + ": " + v; bar.appendChild(seg); }
    row.appendChild(bar);
    const cl = h("div", { class: "complist" });
    for (const [name, v] of comps) cl.appendChild(h("span", null, name + " " + (v > 0 ? "+" : "") + v));
    row.appendChild(cl);
    row.addEventListener("click", () => row.classList.toggle("open"));
    p.appendChild(row);
  }
}

// ---------- tree ----------
function renderTree(rec) {
  const p = $("panel-tree"); p.innerHTML = "";
  const ctr = h("div", { id: "tree-controls" });
  ctr.appendChild(mkBtn("Expand all", () => setAll(true)));
  ctr.appendChild(mkBtn("Collapse all", () => setAll(false)));
  const nb = mkBtn("Null probes: " + (state.treeNull ? "on" : "off"), () => { state.treeNull = !state.treeNull; renderTree(rec); });
  ctr.appendChild(nb);
  p.appendChild(ctr);
  if (!rec.tree) { p.appendChild(h("p", { class: "muted" }, "No search tree for this move (" + rec.stage.replace(/_/g, " ") + ").")); return; }
  p.appendChild(h("div", null, "root player = P" + rec.player + " (maximizing) · depth " + rec.tree_depth + " · " + countNodes(rec.tree) + " nodes"));
  const container = h("div", { class: "tree" });
  container.appendChild(nodeEl(rec.tree, rec.player, true));
  p.appendChild(container);
  function setAll(open) { container.querySelectorAll(".kids").forEach(k => k.classList.toggle("hidden", !open)); container.querySelectorAll(".toggle").forEach(t => { if (t.textContent !== "·") t.textContent = open ? "▾" : "▸"; }); }
}
function mkBtn(txt, fn) { const b = h("button", null, txt); b.addEventListener("click", fn); return b; }
function countNodes(nd) { let n = 1; for (const k of nd.kids) n += countNodes(k); return n; }
function nodeEl(nd, rootPlayer, forceOpen) {
  const wrap = h("div", { class: "tnode" });
  const isMax = nd.t === rootPlayer;
  const row = h("div", { class: "trow" + (nd.pv ? " pv" : "") + (nd.w === "null" ? " tw-null" : "") });
  const kids = nd.kids.filter(k => state.treeNull || k.w !== "null");
  const hasKids = kids.length > 0;
  const tog = h("span", { class: "toggle" }, hasKids ? (nd.pv || forceOpen ? "▾" : "▸") : "·");
  row.appendChild(tog);
  const mv = h("span", { class: "tmove " + (isMax ? "tmax" : "tmin") }, nd.root ? "root" : coord(nd.m));
  row.appendChild(mv);
  row.appendChild(h("span", { class: "muted" }, "d" + nd.d + " [" + fmtBound(nd.a) + "," + fmtBound(nd.b) + "]"));
  row.appendChild(h("span", { class: "tscore" }, "= " + fmtScore(nd)));
  if (nd.w === "null") row.appendChild(h("span", { class: "tag null" }, "probe"));
  if (nd.cut) row.appendChild(h("span", { class: "tag cut" }, "✂" + nd.pruned));
  if (nd.ttcut) row.appendChild(h("span", { class: "tag tt" }, "TT"));
  if (nd.leaf) row.appendChild(h("span", { class: "tag leaf" }, "leaf"));
  if (nd.term === "win") row.appendChild(h("span", { class: "tag win" }, "WIN"));
  if (nd.term === "loss") row.appendChild(h("span", { class: "tag loss" }, "LOSS"));
  row.addEventListener("click", (e) => { if (hasKids) { kidsBox.classList.toggle("hidden"); tog.textContent = kidsBox.classList.contains("hidden") ? "▸" : "▾"; } if (nd.m != null) { flashCell(nd.m); } e.stopPropagation(); });
  wrap.appendChild(row);
  const kidsBox = h("div", { class: "kids" + ((nd.pv || forceOpen) ? "" : " hidden") });
  if (hasKids) { for (const k of kids) kidsBox.appendChild(nodeEl(k, rootPlayer, false)); }
  wrap.appendChild(kidsBox);
  return wrap;
}
let flashEl = null;
function flashCell(i) { if (flashEl) flashEl.remove(); const { x, y } = cellXY(i); flashEl = el("polygon", { points: hexPts(x, y), fill: "none", stroke: "#fff", "stroke-width": 3 }); gMark.appendChild(flashEl); }

// ---------- charts ----------
function lineChart(title, series, opts) {
  opts = opts || {};
  const wrap = h("div", { class: "chart" }); wrap.appendChild(h("h4", null, title));
  const W = 380, H = 120, m = { l: 38, r: 20, t: 8, b: 18 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" });
  let lo = Infinity, hi = -Infinity;
  for (const s of series) for (const y of s.data) { if (y == null) continue; lo = Math.min(lo, y); hi = Math.max(hi, y); }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { hi = lo + 1; }
  const n = Math.max(1, (series[0] ? series[0].data.length : 1) - 1);
  const X = i => m.l + (W - m.l - m.r) * (n ? i / n : 0);
  const Y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  // zero axis
  if (lo < 0 && hi > 0) svg.appendChild(el("line", { x1: m.l, y1: Y(0), x2: W - m.r, y2: Y(0), stroke: "#33405f", "stroke-dasharray": "2,3" }));
  svg.appendChild(el("text", { x: 2, y: Y(hi) + 8, fill: "#8b96b5", "font-size": 9 })).textContent = round2(hi);
  svg.appendChild(el("text", { x: 2, y: Y(lo), fill: "#8b96b5", "font-size": 9 })).textContent = round2(lo);
  for (const s of series) {
    let d = ""; s.data.forEach((v, i) => { if (v == null) return; d += (d ? "L" : "M") + X(i) + " " + Y(v); });
    svg.appendChild(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.6 }));
  }
  // current-move marker
  const mk = el("line", { x1: X(state.mi), y1: m.t, x2: X(state.mi), y2: H - m.b, stroke: "#f6c85f", "stroke-width": 1 }); mk.dataset.role = "cursor"; svg.appendChild(mk);
  wrap.appendChild(svg); wrap._X = X; wrap._svg = svg;
  return wrap;
}
function round2(x) { return Math.abs(x) >= 100 ? Math.round(x) : Math.round(x * 100) / 100; }
let chartEls = [];
function buildCharts() {
  const p = $("panel-charts"); p.innerHTML = ""; chartEls = [];
  p.appendChild(h("h3", null, "Game-level trends (per v7 move)"));
  const evalS = MOVES.map(m => m.eval_score);
  const rMe = MOVES.map(m => { const R = m.resistance[String(m.player)].R; return R == null ? null : Math.log(R); });
  const rOpp = MOVES.map(m => { const R = m.resistance[String(1 - m.player)].R; return R == null ? null : Math.log(R); });
  const nodes = MOVES.map(m => m.nodes);
  const times = MOVES.map(m => m.time);
  const c1 = lineChart("Evaluation score (v7 perspective)", [{ data: evalS, color: "#5fd08a" }]);
  const c2 = lineChart("ln(resistance): v7 (blue) vs opponent (red) — lower is better connected", [{ data: rMe, color: "#4a90e0" }, { data: rOpp, color: "#e0574a" }]);
  const c3 = lineChart("Search nodes / move", [{ data: nodes, color: "#f6c85f" }]);
  const c4 = lineChart("Time / move (s)", [{ data: times, color: "#b48ead" }]);
  [c1, c2, c3, c4].forEach(c => { p.appendChild(c); chartEls.push(c); });
}
function markCharts() { for (const c of chartEls) { const mk = c._svg.querySelector('[data-role=cursor]'); if (mk) { const x = c._X(state.mi); mk.setAttribute("x1", x); mk.setAttribute("x2", x); } } }

// ---------- controls ----------
function buildOverlays() {
  const box = $("overlays");
  const mk = (key, label) => { const l = h("label"); const cb = h("input", { type: "checkbox" }); cb.checked = state.ov[key]; cb.addEventListener("change", () => { state.ov[key] = cb.checked; render(); }); l.appendChild(cb); l.appendChild(document.createTextNode(label)); return l; };
  box.appendChild(mk("current", "current flow"));
  box.appendChild(mk("heat", "potentials"));
  box.appendChild(mk("dijkstra", "shortest paths"));
  box.appendChild(mk("bridges", "bridges"));
  box.appendChild(mk("candidates", "candidates"));
  box.appendChild(mk("tactics", "tactics"));
  box.appendChild(mk("lastmove", "chosen move"));
  box.appendChild(mk("labels", "coords"));
  box.appendChild(h("span", { class: "sep" }));
  const l = h("label"); l.appendChild(document.createTextNode("resistance of "));
  const sel = h("select");
  sel.appendChild(h("option", { value: "0" }, "P0 (L↔R)"));
  sel.appendChild(h("option", { value: "1" }, "P1 (T↔B)"));
  sel.value = String(state.resP);
  sel.addEventListener("change", () => { state.resP = parseInt(sel.value); render(); });
  l.appendChild(sel); box.appendChild(l);
}
function buildTabs() {
  const nav = $("tabs");
  const tabs = [["move", "Move"], ["ranking", "Ranking"], ["tree", "Search tree"], ["charts", "Charts"]];
  for (const [id, label] of tabs) {
    const b = h("button", null, label);
    b.addEventListener("click", () => { state.tab = id; for (const [tid] of tabs) { $("panel-" + tid).hidden = (tid !== id); } nav.querySelectorAll("button").forEach(x => x.classList.remove("active")); b.classList.add("active"); });
    nav.appendChild(b);
    if (id === "move") b.classList.add("active");
  }
}
function buildLegend() {
  const L = $("legend");
  const item = (c, t) => { const s = h("span"); s.innerHTML = `<span class="sw" style="background:${c}"></span>${t}`; return s; };
  L.appendChild(item(P0, "Player 0 — Left↔Right"));
  L.appendChild(item(P1, "Player 1 — Top↔Bottom"));
  L.appendChild(item("#5fd08a", "candidate / immediate win"));
  L.appendChild(item("#f6c85f", "save-bridge / chosen"));
}
function buildMeta() {
  const m = T.meta;
  const res = m.winner == null ? "no result" : (m.v7_won ? `<span class="win">v7 won</span>` : `<span class="lose">v7 lost</span>`);
  $("meta").innerHTML = `<b>v7_resistance</b> (P${m.v7_side}) vs <b>${m.opponent}</b> · ${m.plies} plies · ${res} · budget depth ${m.budget.depth}/${m.budget.time}s`;
}

// ---------- navigation ----------
function go(i) { state.mi = clamp(i, 0, MOVES.length - 1); render(); }
function play() { if (state.playing) { clearInterval(state.timer); state.playing = false; $("btn-play").textContent = "▶"; return; } state.playing = true; $("btn-play").textContent = "⏸"; state.timer = setInterval(() => { if (state.mi >= MOVES.length - 1) { play(); return; } go(state.mi + 1); }, 900); }

function init() {
  buildMeta(); buildOverlays(); buildTabs(); buildLegend(); buildCharts();
  const rg = $("range"); rg.max = MOVES.length - 1; rg.addEventListener("input", () => go(parseInt(rg.value)));
  $("btn-first").onclick = () => go(0);
  $("btn-prev").onclick = () => go(state.mi - 1);
  $("btn-next").onclick = () => go(state.mi + 1);
  $("btn-last").onclick = () => go(MOVES.length - 1);
  $("btn-play").onclick = play;
  document.addEventListener("keydown", (e) => { if (e.key === "ArrowLeft") go(state.mi - 1); else if (e.key === "ArrowRight") go(state.mi + 1); });
  render();
}
init();
})();
