import WaveSurfer from "./vendor/wavesurfer.esm.js";
import RegionsPlugin from "./vendor/regions.esm.js";

const COLORS = ["#38d9f5", "#ff7ab6", "#a6e05a", "#ffc24b", "#7aa8ff", "#ff8a65", "#4fe3b0", "#c792ff"];
const WAVE = {
  waveColor: "#3b4252", progressColor: "#69718a", cursorColor: "#e9ebf1", cursorWidth: 1,
  barWidth: 2, barGap: 1, barRadius: 2, normalize: false,
};

const $ = (sel, root = document) => root.querySelector(sel);
const setlist = $("#setlist");
const mixBtn = $("#mixBtn");

const PARTS = [["drums", "ドラム"], ["bass", "ベース"], ["vocals", "ボーカル"], ["other", "その他"]];
const PART_NAME = Object.fromEntries(PARTS);

const tracks = [];   // セットリスト（並び順どおり）
const links = new Map();  // 曲と曲の間のブリッジの素材（「前の曲の番号-次の曲の番号」ごと）
let stemsAvailable = false;  // AI のパート分けが使えるか
let nextColor = 0;
let nextUid = 0;
let version = 0;     // 何か変更するたびに増える。ミックス結果が古いかどうかの判定に使う
let mix = null;      // 最後のミックス { id, result, ws, version }
let mixing = false;
let uploads = Promise.resolve();

// ---------- 小物

function fmt(sec) {
  const r = Math.round(Math.max(0, sec) * 10) / 10;
  const m = Math.floor(r / 60);
  return `${m}:${(r - m * 60).toFixed(1).padStart(4, "0")}`;
}

function parseTime(text) {
  const parts = String(text).trim().split(":");
  if (!parts[0] || parts.length > 3) return null;
  let sec = 0;
  for (const p of parts) {
    const v = Number(p);
    if (!Number.isFinite(v) || v < 0) return null;
    sec = sec * 60 + v;
  }
  return sec;
}

function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function toast(text, error = false) {
  const el = document.createElement("div");
  el.className = "toast" + (error ? " error" : "");
  el.textContent = text;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), error ? 7000 : 4000);
}

function transitionText(next) {
  if (next.kind === "echo") return "エコーアウトで切り替え";
  if (next.kind === "crossfade") {
    const b = next.beats;
    return (b % 4 === 0 ? `${b / 4}小節` : `${b}拍`) + "のクロスフェード（ベーススワップ）";
  }
  const tempo = t => (t ? `${Math.round(t)} BPM` : "テンポなし");
  const names = [...next.parts.a.map(p => `前の${PART_NAME[p]}`), ...next.parts.b.map(p => `次の${PART_NAME[p]}`)];
  if (next.synth) names.push("合成音");
  let text = `${next.bars}小節のブリッジを生成（${tempo(next.from_bpm)} → ${tempo(next.to_bpm)}、${next.from_key} → ${next.to_key}／${names.join("・")}）`;
  if (next.missing.length) text += ` ※${next.missing.join("・")}は声が見つからず使っていません`;
  return text;
}

function stopAll(except) {
  for (const t of tracks) if (t.ws && t !== except) t.ws.pause();
  if (mix && except !== "mix") mix.ws.pause();
}

const isFresh = () => mix && mix.version === version;

// ---------- セットリスト

function createTrack(props) {
  const t = { uid: ++nextUid, loops: 1, loading: true, color: COLORS[nextColor++ % COLORS.length], ...props };
  t.el = $("#track-tpl").content.firstElementChild.cloneNode(true);
  t.el.style.setProperty("--c", t.color);
  const q = sel => $(sel, t.el);
  q(".name").textContent = t.name;
  q(".name").title = t.name;
  q(".up").onclick = () => move(t, -1);
  q(".down").onclick = () => move(t, 1);
  q(".dup").onclick = () => duplicate(t);
  q(".remove").onclick = () => remove(t);
  q(".play").onclick = () => togglePreview(t);
  q(".loops").onchange = e => { t.loops = Number(e.target.value); changed(); };
  for (const [sel, key] of [[".t-start", "start"], [".t-end", "end"]]) {
    const input = q(sel);
    input.onchange = () => {
      const v = parseTime(input.value);
      if (v === null) return refresh(t);
      setRange(t, key === "start" ? v : t.start, key === "end" ? v : t.end);
    };
    input.onkeydown = e => { if (e.key === "Enter") input.blur(); };
  }
  return t;
}

function ready(t) {
  t.loading = false;
  for (const el of t.el.querySelectorAll("button, input, select")) el.disabled = false;
  $(".loops", t.el).value = String(t.loops);
  mountWave(t);
  changed();
}

function mountWave(t) {
  const box = $(".wave", t.el);
  box.replaceChildren();
  t.regions = RegionsPlugin.create();
  t.ws = WaveSurfer.create({
    ...WAVE, container: box, height: 84, plugins: [t.regions],
    url: `/api/tracks/${t.id}/audio`, peaks: [t.peaks], duration: t.duration,
  });
  const color = rgba(t.color, 0.22);
  // 曲の長さが分かる前に範囲を作ると 0 秒に丸められてしまうので、準備ができてから作る
  t.ws.once("ready", () => {
    t.region = t.regions.addRegion({ start: t.start, end: t.end, color, minLength: 1 });
    t.regions.enableDragSelection({ color, minLength: 1 });
  });
  t.regions.on("region-created", r => {  // ドラッグで選び直したら前の範囲と入れ替える
    if (!t.region || r === t.region) return;
    t.region.remove();
    t.region = r;
    fromRegion(t, true);
  });
  t.regions.on("region-update", r => r === t.region && fromRegion(t, false));
  t.regions.on("region-updated", r => r === t.region && fromRegion(t, true));
  t.ws.on("play", () => { stopAll(t); setPlaying(t, true); });
  t.ws.on("pause", () => setPlaying(t, false));
}

function fromRegion(t, done) {
  t.start = t.region.start;
  t.end = t.region.end;
  done ? changed() : refresh(t);
}

function setRange(t, start, end) {
  t.start = Math.max(0, Math.min(start, t.duration - 1));
  t.end = Math.max(t.start + 1, Math.min(end, t.duration));
  t.region?.setOptions({ start: t.start, end: t.end });
  changed();
}

function setPlaying(t, on) {
  const btn = $(".play", t.el);
  btn.textContent = on ? "⏸ 停止" : "▶ 範囲を試聴";
  btn.classList.toggle("on", on);
}

function togglePreview(t) {
  if (t.ws.isPlaying()) return t.ws.pause();
  stopAll(t);
  t.region ? t.region.play(true) : t.ws.play(t.start, t.end);
}

function move(t, dir) {
  const i = tracks.indexOf(t);
  const j = i + dir;
  if (j < 0 || j >= tracks.length) return;
  [tracks[i], tracks[j]] = [tracks[j], tracks[i]];
  changed();
}

function duplicate(t) {
  const { id, name, duration, bpm, steady, key, peaks, start, end } = t;
  const c = createTrack({ id, name, duration, bpm, steady, key, peaks, start, end });
  tracks.splice(tracks.indexOf(t) + 1, 0, c);
  render();
  ready(c);
  toast("同じ曲を追加しました。別の範囲を選んでください");
}

function remove(t) {
  const i = tracks.indexOf(t);
  if (i < 0) return;
  t.ws?.destroy();
  t.el.remove();
  tracks.splice(i, 1);
  changed();
}

function addFiles(files) {
  for (const file of files) {
    const t = createTrack({ name: file.name });
    tracks.push(t);
    uploads = uploads.then(() => upload(t, file));  // 解析は重いので1曲ずつ
  }
  changed();
}

async function upload(t, file) {
  if (!tracks.includes(t)) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/tracks", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "読み込みに失敗しました");
    const [start, end] = data.suggested;
    Object.assign(t, { id: data.id, duration: data.duration, bpm: data.bpm, steady: data.steady, key: data.key, peaks: data.peaks, start, end });
    if (tracks.includes(t)) ready(t);
  } catch (e) {
    toast(`${file.name}: ${e.message}`, true);
    remove(t);
  }
}

// ---------- 表示の更新

function changed() {
  version++;
  render();
}

// ---------- 曲と曲の間（つなぎ）とブリッジの素材

function linkState(a, b) {
  const key = `${a.uid}-${b.uid}`;
  if (!links.has(key)) links.set(key, { a: new Set(), b: new Set(), synth: true, open: false });
  return links.get(key);
}

function linkRequest(a, b) {
  const s = linkState(a, b);
  return { a: [...s.a], b: [...s.b], synth: s.synth };
}

function renderLink(a, b, text, fresh) {
  const s = linkState(a, b);
  const chosen = s.a.size + s.b.size > 0;
  const link = document.createElement("div");
  link.className = "link" + (fresh ? " done" : "");
  link.style.setProperty("--from", a.color);
  link.style.setProperty("--to", b.color);
  const body = document.createElement("div");
  body.className = "link-body";
  const label = document.createElement("span");
  label.textContent = text;
  const toggle = document.createElement("button");
  toggle.className = "link-toggle";
  toggle.setAttribute("aria-expanded", String(s.open));
  const summary = [...[...s.a].map(p => `前の${PART_NAME[p]}`), ...[...s.b].map(p => `次の${PART_NAME[p]}`)];
  toggle.textContent = `ブリッジの素材${chosen ? `: ${summary.join("・")}${s.synth ? "・合成音" : ""}` : ""} ${s.open ? "▾" : "▸"}`;
  toggle.onclick = () => { s.open = !s.open; render(); };
  body.append(label, toggle);

  if (s.open) {
    const parts = document.createElement("div");
    parts.className = "parts";
    const group = (title, side, color) => {
      const g = document.createElement("span");
      g.className = "part-group";
      g.style.setProperty("--c", color);
      g.append(Object.assign(document.createElement("span"), { className: "group-label", textContent: title }));
      for (const [part, name] of PARTS) {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.textContent = name;
        chip.disabled = !stemsAvailable;
        chip.title = stemsAvailable ? "" : "AI のパート分けが使えません";
        chip.setAttribute("aria-pressed", String(s[side].has(part)));
        chip.onclick = () => { s[side].has(part) ? s[side].delete(part) : s[side].add(part); changed(); };
        g.append(chip);
      }
      return g;
    };
    const synth = document.createElement("button");
    synth.className = "chip synth";
    synth.textContent = "合成音";
    synth.setAttribute("aria-pressed", String(s.synth));
    synth.onclick = () => { s.synth = !s.synth; changed(); };
    const note = document.createElement("span");
    note.className = "parts-note";
    note.textContent = stemsAvailable
      ? "パートを選ぶと、ここには必ずブリッジが入ります（AI のパート分けに数十秒かかります）"
      : "AI のパート分けが使えないため、合成音だけになります";
    parts.append(group("前の曲", "a", a.color), group("次の曲", "b", b.color), synth, note);
    body.append(parts);
  }
  link.append(body);
  return link;
}

function render() {
  const fresh = isFresh();
  const nodes = [];
  tracks.forEach((t, i) => {
    if (i > 0) {
      const text = fresh ? transitionText(mix.result.tracks[i - 1].next) : "つなぎ";
      nodes.push(renderLink(tracks[i - 1], t, text, fresh));
    }
    $(".num", t.el).textContent = i + 1;
    $(".up", t.el).disabled = t.loading || i === 0;
    $(".down", t.el).disabled = t.loading || i === tracks.length - 1;
    refresh(t);
    nodes.push(t.el);
  });
  setlist.replaceChildren(...nodes);
  document.body.classList.toggle("empty", !tracks.length);
  mixBtn.disabled = mixing || !tracks.length || tracks.some(t => t.loading);
  mixBtn.textContent = mix && !fresh ? "もう一度ミックス" : "ミックスする";
  $("#staleNote").hidden = !mix || fresh || mixing;
}

function refresh(t) {
  const q = sel => $(sel, t.el);
  const meta = q(".meta");
  meta.classList.toggle("free", !t.loading && !t.steady);
  if (t.loading) {
    meta.textContent = "解析中…";
    return;
  }
  meta.textContent = `${t.steady && t.bpm ? `${t.bpm.toFixed(1)} BPM` : "テンポ不定"} · ${t.key} · ${fmt(t.duration)}`;
  if (document.activeElement !== q(".t-start")) q(".t-start").value = fmt(t.start);
  if (document.activeElement !== q(".t-end")) q(".t-end").value = fmt(t.end);
  const len = t.end - t.start;
  q(".len").textContent = t.loops > 1 ? `長さ ${fmt(len)} ×${t.loops} = ${fmt(len * t.loops)}` : `長さ ${fmt(len)}`;

  const plan = q(".plan");
  plan.replaceChildren();
  const i = tracks.indexOf(t);
  if (isFresh() && mix.result.tracks[i]) {
    const r = mix.result.tracks[i];
    const stretch = document.createElement("span");
    stretch.className = "stretch";
    stretch.textContent = r.stretch == null ? " · テンポそのまま" : ` · テンポ ${r.stretch >= 0 ? "+" : ""}${(r.stretch * 100).toFixed(1)}%`;
    plan.append(`ミックス内 ${fmt(r.mix_start)}–${fmt(r.mix_end)}`, stretch);
  }
}

// ---------- ミックス

async function startMix() {
  stopAll();
  mixing = true;
  const startVersion = version;
  const colors = tracks.map(t => t.color);
  const body = {
    items: tracks.map(t => ({ id: t.id, start: t.start, end: t.end, loops: t.loops })),
    bars: Number($("#bars").value),
    bpm: parseFloat($("#bpm").value) || null,
    max_stretch: Number($("#maxStretch").value),
    bridge: $("#bridge").value,
    links: tracks.slice(1).map((t, i) => linkRequest(tracks[i], t)),
  };
  render();
  $("#deck").hidden = false;
  setProgress(0, "準備中");
  try {
    const res = await fetch("/api/mix", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    const result = await poll(data.id);
    showMix(data.id, result, colors, startVersion);
  } catch (e) {
    toast(e.message, true);
  } finally {
    mixing = false;
    $("#deckProgress").hidden = true;
    render();
  }
}

async function poll(id) {
  for (;;) {
    await new Promise(r => setTimeout(r, 400));
    const job = await (await fetch(`/api/mix/${id}`)).json();
    if (job.state === "done") return job.result;
    if (job.state === "error") throw new Error(job.error);
    setProgress(job.progress, job.message);
  }
}

function setProgress(frac, message) {
  $("#deckProgress").hidden = false;
  $(".progress > div").style.width = `${Math.round(frac * 100)}%`;
  $("#deckMessage").textContent = message;
}

function showMix(id, result, colors, startVersion) {
  mix?.ws.destroy();
  const regions = RegionsPlugin.create();
  const ws = WaveSurfer.create({
    ...WAVE, container: $("#deckWave"), height: 64, plugins: [regions],
    url: `/api/mix/${id}/audio`, peaks: [result.peaks], duration: result.duration,
  });
  ws.once("ready", () => result.tracks.forEach((r, i) => {
    regions.addRegion({
      start: r.mix_start, end: r.mix_end, color: rgba(colors[i], 0.28),
      drag: false, resize: false, content: String(i + 1),
    });
    if (r.next?.kind === "bridge") {  // ブリッジは次の曲の範囲の手前まで
      const label = document.createElement("span");
      label.className = "bridge-label";
      label.textContent = "ブリッジ";
      regions.addRegion({
        start: r.mix_end, end: result.tracks[i + 1].mix_start, color: "rgba(233, 235, 241, 0.10)",
        drag: false, resize: false, content: label,
      });
    }
  }));
  mix = { id, result, ws, version: startVersion };

  const time = () => { $("#deckTime").textContent = `${fmt(ws.getCurrentTime())} / ${fmt(result.duration)}`; };
  ws.on("timeupdate", time);
  ws.on("play", () => { stopAll("mix"); $("#deckPlay").textContent = "⏸"; });
  ws.on("pause", () => { $("#deckPlay").textContent = "▶"; });
  time();
  $("#deckInfo").textContent = `${result.bpm ? `${result.bpm.toFixed(1)} BPM` : "テンポなし"} · ${result.tracks.length}曲 · ${fmt(result.duration)}`;
  $("#deckPlay").disabled = false;
  $("#deckPlay").textContent = "▶";
  $("#saveBtn").disabled = false;
}

async function save() {
  const data = await (await fetch(`/api/mix/${mix.id}/save`, { method: "POST" })).json();
  if (data.dialog === false) {  // ブラウザで開いているときは普通のダウンロード
    const a = document.createElement("a");
    a.href = `/api/mix/${mix.id}/audio?download=1`;
    a.download = "miso-soup-mix.wav";
    a.click();
  } else if (data.saved) {
    toast(`保存しました: ${data.saved}`);
  }
}

// ---------- イベント

mixBtn.onclick = startMix;
$("#deckPlay").onclick = () => mix?.ws.playPause();
$("#saveBtn").onclick = () => save().catch(e => toast(e.message, true));
for (const sel of ["#bars", "#bpm", "#maxStretch", "#bridge"]) $(sel).addEventListener("change", changed);

const fileInput = $("#fileInput");
fileInput.onchange = () => { addFiles(fileInput.files); fileInput.value = ""; };
const drop = $("#drop");
window.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
window.addEventListener("dragleave", e => { if (!e.relatedTarget) drop.classList.remove("over"); });
window.addEventListener("drop", e => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer?.files.length) addFiles(e.dataTransfer.files);
});
window.addEventListener("keydown", e => {  // スペースでミックスを再生・停止
  if (e.code !== "Space" || !mix || ["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) return;
  e.preventDefault();
  mix.ws.playPause();
});

fetch("/api/info").then(r => r.json()).then(info => { stemsAvailable = info.stems; render(); }).catch(() => {});
render();
